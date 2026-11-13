# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native OmniSampler for ten parallel RVQ draws and one logical AR token.

Backbone forward never invokes the head. CUDA random draws use the upstream
request-seeded Gumbel implementation; CPU numerical checks use the reference
Torch multinomial stream. Both keep request RNG/history independent of batch
composition. RVQ codes leave through normal asynchronous output materialization.
"""

from __future__ import annotations

import math
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, NoReturn

import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, register_top_k_top_p_warmups
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.model_executor.output_snapshot import RequestOutputSnapshot
from vllm_omni.utils.device_copy import index_to_device
from vllm_omni.worker_v2.omni_sampler import OmniSampler, OmniSamplingOutput, StandardSample
from vllm_omni.worker_v2.warmup import is_warmup_request

from .ar_config import Zonos3Architecture
from .modeling_zonos3_mini import MultiEmbedding, MultiOutputHead
from .sampling_state import EOS, Zonos3ARSamplingState

ZONOS3_SAMPLING_KEY = "zonos3_sampling"


@dataclass(frozen=True)
class Zonos3SamplingOptions:
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    repetition_penalty: float
    repetition_window: int
    repetition_codebooks: int
    min_frames: int
    seed: int
    ignore_eos: bool
    eoa_codebook0: bool

    @classmethod
    def from_params(
        cls, params: SamplingParams, config: Zonos3Architecture, history_capacity: int
    ) -> Zonos3SamplingOptions:
        extra = (params.extra_args or {}).get(ZONOS3_SAMPLING_KEY, {})
        if not isinstance(extra, Mapping) or set(extra) - {
            "repetition_window",
            "repetition_codebooks",
            "min_frames",
            "eoa_policy",
        }:
            raise ValueError("Invalid zonos3_sampling options")
        if (
            params.logprobs is not None
            or params.prompt_logprobs is not None
            or params.logit_bias
            or params.allowed_token_ids
            or params.bad_words
            or params.structured_outputs
            or params.stop
            or params.frequency_penalty
            or params.presence_penalty
            or params.min_tokens
            or params.n != 1
            or any(token != EOS for token in params.stop_token_ids)
        ):
            raise ValueError("Unsupported logical-token sampling constraint for Zonos3 RVQ generation")
        policy = extra.get("eoa_policy", "any")
        if policy not in ("any", "codebook0"):
            raise ValueError("Zonos3 eoa_policy must be any or codebook0")
        codebooks = int(extra.get("repetition_codebooks", 6))
        seed = int(params.seed) if params.seed is not None else secrets.randbits(63)
        seed = seed & ((1 << 64) - 1)
        if seed >= 1 << 63:
            seed -= 1 << 64
        values = (params.temperature, params.top_p, params.min_p, params.repetition_penalty)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Zonos3 sampling scalars must be finite")
        return cls(
            temperature=max(float(params.temperature), 0.0),
            top_k=min(params.top_k if params.top_k > 0 else config.codebook_size, config.audio_vocab),
            top_p=float(params.top_p) if params.top_p > 0 else 1.0,
            min_p=max(float(params.min_p), 0.0),
            repetition_penalty=max(float(params.repetition_penalty), 1.0),
            repetition_window=min(max(int(extra.get("repetition_window", 96)), 0), history_capacity),
            repetition_codebooks=10 if codebooks < 0 else min(codebooks, 10),
            min_frames=max(int(extra.get("min_frames", 0)), 0),
            seed=seed,
            ignore_eos=bool(params.ignore_eos),
            eoa_codebook0=policy == "codebook0",
        )


@dataclass
class _Settings:
    temperature: UvaBackedTensor
    top_k: UvaBackedTensor
    top_p: UvaBackedTensor
    min_p: UvaBackedTensor
    repetition_penalty: UvaBackedTensor
    repetition_window: UvaBackedTensor
    repetition_codebooks: UvaBackedTensor
    min_frames: UvaBackedTensor
    seed: UvaBackedTensor


def finalize_zonos3_codes(payload: dict[str, Any], counts: list[int]) -> RequestOutputSnapshot:
    codes = payload["codes"]["audio"]
    active = payload["meta"]["codec_frame_valid"]
    eos = payload["meta"]["codec_eos_valid"]
    if codes.shape != (len(counts), 1, 10) or active.shape != (len(counts),) or eos.shape != (len(counts),):
        raise ValueError("Zonos3 output snapshot does not match the materialized request cohort")
    if codes.device.type != "cpu" or active.device.type != "cpu" or eos.device.type != "cpu":
        raise ValueError("Zonos3 finalizer requires the runner's CPU-owned output snapshot")
    result = []
    for row, count in enumerate(counts):
        if count not in (0, 1):
            raise ValueError("Zonos3 emits at most one RVQ row per logical sampling step")
        result.append(
            {"codes": {"audio": codes[row]}, "meta": {"codec_eos_valid": eos[row : row + 1]}}
            if count and bool(active[row])
            else None
        )
    return RequestOutputSnapshot(result)


class Zonos3Sampler(OmniSampler):
    def __init__(
        self,
        base_sampler: Sampler,
        config: Zonos3Architecture,
        embedder: MultiEmbedding,
        head: MultiOutputHead,
        state: Zonos3ARSamplingState,
        next_embeddings: torch.Tensor,
        history_capacity: int,
    ) -> None:
        super().__init__(base_sampler)
        if history_capacity < 1:
            raise ValueError("Zonos3 repetition history capacity must be positive")
        self.config, self.embedder, self.head = config, embedder, head
        self.state, self.next_embeddings = state, next_embeddings
        self.history_capacity = history_capacity
        capacity = state.position.shape[0]
        self.history = torch.full((capacity, 10, history_capacity), -1, dtype=torch.int64, device=state.position.device)
        self.settings = _Settings(
            **{
                name: UvaBackedTensor(capacity, dtype=dtype)
                for name, dtype in (
                    ("temperature", torch.float32),
                    ("top_k", torch.int32),
                    ("top_p", torch.float32),
                    ("min_p", torch.float32),
                    ("repetition_penalty", torch.float32),
                    ("repetition_window", torch.int32),
                    ("repetition_codebooks", torch.int32),
                    ("min_frames", torch.int32),
                    ("seed", torch.int64),
                )
            }
        )
        self.options: list[Zonos3SamplingOptions | None] = [None] * capacity
        self._pending: set[int] = set()
        self._generators: dict[int, torch.Generator] = {}
        if state.position.device.type == "cuda":
            register_top_k_top_p_warmups()

    @property
    def omni_static_staged_writes(self) -> bool:
        return not self._pending and super().omni_static_staged_writes

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        request_ids = getattr(self.base_sampler.req_states, "index_to_req_id", {})
        request_id = request_ids.get(req_idx, "")
        if isinstance(request_id, str) and is_warmup_request(request_id):
            # Upstream's generic warmup deliberately enables text-only logit
            # processors/logprobs. Warm the physical RVQ distribution instead;
            # real requests still receive strict constraint validation.
            sampling_params = self._warmup_params()
        options = Zonos3SamplingOptions.from_params(sampling_params, self.config, self.history_capacity)
        self.base_sampler.add_request(req_idx, sampling_params)
        self.options[req_idx] = options
        for field in fields(self.settings):
            getattr(self.settings, field.name).np[req_idx] = getattr(options, field.name)
        self._pending.add(req_idx)
        if self.state.position.device.type == "cpu":
            self._generators[req_idx] = torch.Generator().manual_seed(options.seed)

    @staticmethod
    def _warmup_params() -> SamplingParams:
        return SamplingParams(
            temperature=0.9,
            top_k=50,
            top_p=0.9,
            min_p=0.1,
            repetition_penalty=1.2,
            seed=0,
            detokenize=False,
            stop_token_ids=[EOS],
        )

    def warmup(self, hidden_states: torch.Tensor) -> bool:
        rows = hidden_states.shape[0]
        for slot in range(rows):
            self.add_request(slot, self._warmup_params())
        self.apply_staged_writes()
        buffers = InputBuffers(rows, rows, hidden_states.device)
        batch = InputBatch.make_dummy(rows, rows, buffers)
        # No text logits or second head call: exercise the registered custom
        # sampler exactly where serving computes the ten codebook heads.
        self.sample_step(hidden_states, batch, self.base_sampler.req_states, None, self._no_standard_warmup)
        return True

    @staticmethod
    def _no_standard_warmup(*args: object) -> NoReturn:
        raise RuntimeError("Zonos3 warmup must not call the standard text sampler")

    def apply_staged_writes(self) -> None:
        self.base_sampler.apply_staged_writes()
        if not self._pending:
            return
        for field in fields(self.settings):
            getattr(self.settings, field.name).copy_to_uva()
        slots = sorted(self._pending)
        indices = index_to_device(slots, self.state.position.device)
        prefix = self.state.has_audio_prefix.index_select(0, indices)
        self.state.reset(
            indices,
            prefix,
            index_to_device([self.options[slot].ignore_eos for slot in slots], indices.device, dtype=torch.bool),
            index_to_device([self.options[slot].eoa_codebook0 for slot in slots], indices.device, dtype=torch.bool),
        )
        self.history.index_fill_(0, indices, -1)
        self.next_embeddings.index_fill_(0, indices, 0)
        self._pending.clear()

    def probabilities(self, logits: torch.Tensor, slots: torch.Tensor, width: int) -> torch.Tensor:
        """Reference filters with per-request settings and GPU-resident history."""
        c = self.config
        logits = logits.float()
        positions = self.state.position.index_select(0, slots)
        ids = torch.arange(c.audio_vocab, device=logits.device)
        invalid = (ids >= c.codebook_size) & (ids != c.eoa_id)
        minimum = self.settings.min_frames.gpu.index_select(0, slots)
        stop_blocked = (minimum[:, None] > 0) & (positions[:, None] < minimum[:, None] + self.state.columns)
        logits = logits.masked_fill(
            invalid[None, None, :] | (stop_blocked[:, :, None] & (ids == c.eoa_id)), float("-inf")
        )
        if width:
            offsets = torch.arange(width, device=logits.device)
            indices = (positions[:, None] - 1 - offsets) % self.history_capacity
            history = self.history[slots[:, None, None], self.state.columns[None, :, None], indices[:, None, :]]
            window = self.settings.repetition_window.gpu.index_select(0, slots)
            books = self.settings.repetition_codebooks.gpu.index_select(0, slots)
            valid = (history >= 0) & (history < c.codebook_size)
            valid = valid & (offsets[None, None, :] < torch.minimum(positions, window)[:, None, None])
            valid = valid & (self.state.columns[None, :, None] < books[:, None, None])
            repeated = torch.zeros_like(logits, dtype=torch.int32)
            repeated.scatter_add_(-1, history.clamp(min=0, max=c.audio_vocab - 1), valid.to(torch.int32))
            penalty = self.settings.repetition_penalty.gpu.index_select(0, slots)[:, None, None]
            adjusted = torch.where(logits > 0, logits / penalty, logits * penalty)
            logits = torch.where(repeated > 0, adjusted, logits)
        temperature = self.settings.temperature.gpu.index_select(0, slots)[:, None, None]
        logits = logits / torch.where(temperature > 0, temperature, 1.0)
        k = self.settings.top_k.gpu.index_select(0, slots)[:, None].expand(-1, 10).reshape(-1)
        p = self.settings.top_p.gpu.index_select(0, slots)[:, None].expand(-1, 10).reshape(-1)
        logits = apply_top_k_top_p(logits.reshape(-1, c.audio_vocab), k, p, has_topp_only_rows=True).view(
            -1, 10, c.audio_vocab
        )
        probs = logits.softmax(-1)
        min_p = self.settings.min_p.gpu.index_select(0, slots)[:, None, None]
        probs = probs.masked_fill(probs < min_p * probs.amax(-1, keepdim=True), 0)
        total = probs.sum(-1, keepdim=True)
        fallback = torch.zeros_like(probs).scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
        return torch.where(total > 0, probs / total.clamp(min=1e-8), fallback)

    def _draw_cpu(self, probs: torch.Tensor, indices: list[int], active: torch.Tensor) -> torch.Tensor:
        # Explicit CPU numerical path; the CUDA path below never reads a GPU
        # active/EOA mask on the host and uses stateless native Gumbel draws.
        result = torch.full((len(indices), 10), self.config.audio_pad_id, dtype=torch.int64)
        for row, slot in enumerate(indices):
            if bool(active[row]):
                if self.options[slot].temperature == 0:
                    result[row] = probs[row].argmax(-1)
                else:
                    result[row] = torch.multinomial(
                        probs[row, :, : self.config.audio_pad_id + 1], 1, generator=self._generators[slot]
                    ).flatten()
        return result

    def sample_step(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        req_states: RequestState,
        grammar_output: GrammarOutput | None,
        standard_sample: StandardSample,
    ) -> OmniSamplingOutput:
        del standard_sample
        warmup_only = bool(input_batch.req_ids) and all(is_warmup_request(req_id) for req_id in input_batch.req_ids)
        if (
            (grammar_output is not None and not warmup_only)
            or input_batch.num_draft_tokens
            or input_batch.logits_indices.numel() != input_batch.num_reqs
        ):
            raise ValueError("Zonos3 RVQ sampling does not support grammar or speculative scheduler tokens")
        if self._pending:
            raise RuntimeError("Zonos3 sampler staged writes must be applied before sampling")
        slots = input_batch.idx_mapping
        options = [self.options[int(slot)] for slot in input_batch.idx_mapping_np]
        if any(option is None for option in options):
            raise RuntimeError("Zonos3 sampler request was not initialized")
        if hidden_states.device.type == "cuda":
            sampled, rejected = get_num_sampled_and_rejected(
                input_batch.seq_lens.new_ones(input_batch.num_reqs),
                input_batch.seq_lens,
                input_batch.cu_num_logits,
                slots,
                req_states.prefill_len.gpu,
            )
        else:
            sampled = (input_batch.seq_lens >= req_states.prefill_len.gpu.index_select(0, slots)).to(torch.int32)
            rejected = torch.zeros_like(sampled)
        active = (sampled > 0) & ~(
            (self.state.eoa_frame.index_select(0, slots) >= 0)
            & (self.state.drain_remaining.index_select(0, slots) == 0)
        )
        position = self.state.position.index_select(0, slots)
        logits = self.head(hidden_states.index_select(0, input_batch.logits_indices))
        probs = self.probabilities(logits, slots, max(option.repetition_window for option in options))
        if hidden_states.device.type == "cuda":
            noise_position = (position[:, None] * 10 + self.state.columns).reshape(-1)
            codes = gumbel_sample(
                probs.log().reshape(-1, self.config.audio_vocab),
                slots.repeat_interleave(10),
                self.settings.temperature.gpu,
                self.settings.seed.gpu,
                noise_position,
                apply_temperature=False,
                is_drafting=False,
            ).view(-1, 10)
        elif hidden_states.device.type == "cpu":
            codes = self._draw_cpu(probs, [int(slot) for slot in input_batch.idx_mapping_np], active)
        else:
            raise ValueError("Zonos3 sampler supports CUDA serving and CPU numerical checks")
        step = self.state.accept(slots, codes, active)
        ring = position[:, None] % self.history_capacity
        old = self.history[slots[:, None], self.state.columns[None, :], ring]
        self.history[slots[:, None], self.state.columns[None, :], ring] = torch.where(
            step.active[:, None], step.codes, old
        )
        frame = torch.cat((step.codes, torch.full_like(step.codes[:, :1], self.config.text_vocab)), -1)
        embeddings = self.embedder(frame)
        previous = self.next_embeddings.index_select(0, slots)
        self.next_embeddings.index_copy_(0, slots, torch.where(step.active[:, None], embeddings, previous))
        output = SamplerOutput(step.logical_token_ids.long(), None, None, sampled, rejected)
        return OmniSamplingOutput(
            output,
            sampled,
            rejected,
            multimodal_outputs={
                "codes": {"audio": step.codes[:, None, :]},
                "meta": {
                    "codec_frame_valid": step.active,
                    "codec_eos_valid": step.finished,
                },
            },
            include_hidden_states=False,
            owns_multimodal_outputs=True,
            finalize_multimodal=finalize_zonos3_codes,
        )
