# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native MRV2 request state for the Zonos3 AR and Codec stages."""

from __future__ import annotations

import math
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, apply_top_k_top_p_pytorch
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

from vllm_omni.model_executor.output_snapshot import PackedOutputSnapshot
from vllm_omni.utils.device_copy import index_to_device, to_device_nonblocking
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.warmup import is_warmup_request

from .ar_config import Zonos3Architecture
from .modeling_zonos3_mini import Zonos3Backbone, Zonos3BackboneBatch, Zonos3PrefillSpan
from .sampling_state import EOS, Zonos3ARSamplingState


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
        extra = params.extra_args or {}
        if not isinstance(extra, Mapping) or set(extra) - {
            "repetition_window",
            "repetition_codebooks",
            "min_frames",
            "eoa_policy",
            "tts_local_seed",
        }:
            raise ValueError("Invalid Zonos3 sampling options")
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
        request_seed = extra.get("tts_local_seed", params.seed)
        seed = int(request_seed) if request_seed is not None else secrets.randbits(63)
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
class _SamplingSettings:
    temperature: UvaBackedTensor
    top_k: UvaBackedTensor
    top_p: UvaBackedTensor
    min_p: UvaBackedTensor
    repetition_penalty: UvaBackedTensor
    repetition_window: UvaBackedTensor
    repetition_codebooks: UvaBackedTensor
    min_frames: UvaBackedTensor
    seed: UvaBackedTensor


class Zonos3CodeSnapshot(PackedOutputSnapshot):
    """One owned slab of per-request RVQ rows.

    Request-owned lists keep the generic runner's token-span slicing out of
    codebook rows, including mixed/partial prefills. Each row is the raw
    sampled RVQ output; the AR sampler's GPU state decides when sampling ends.
    """

    def __init__(self, codes: torch.Tensor) -> None:
        self._codes = codes
        dict.__init__(self, self._payload(codes))
        self.producer_event = None

    @staticmethod
    def _payload(codes: torch.Tensor) -> dict[str, Any]:
        return {"codes": {"audio": list(codes.split(1))}}

    def copy_to_cpu(self, copy_tensor: Callable[[torch.Tensor], torch.Tensor]) -> dict[str, Any]:
        return self._payload(copy_tensor(self._codes))


@dataclass
class _Prompt:
    embeddings: torch.Tensor
    has_audio_prefix: bool


class Zonos3ARModelState(OmniModelState):
    supports_prompt_embeds = False
    # vLLM attaches an all-valid grammar mask to startup warmup requests and
    # uses sampler-warmup SamplingParams (including logprobs > this model's
    # two logical tokens). Zonos3 determines CONTINUE/EOS in its own GPU step;
    # route only synthetic warmup requests through that result as well.
    sample_determined_tokens_during_warmup = True

    def __init__(
        self, vllm_config: VllmConfig, model: Zonos3Backbone, encoder_cache: EncoderCache | None, device: torch.device
    ) -> None:
        if vllm_config.model_config.enable_prompt_embeds or encoder_cache is not None:
            raise ValueError("Zonos3 conditioning must use its request metadata, not generic prompt/MM embeddings")
        if vllm_config.speculative_config is not None:
            raise ValueError("Zonos3 V1 does not support speculative scheduler tokens")
        super().__init__(vllm_config, model, encoder_cache, device)
        self.has_preprocess = True
        self._embed_dim = model.config.dim
        if self._static_inputs_embeds is None:
            self._static_inputs_embeds = torch.zeros(
                (self.max_num_tokens, self._embed_dim), dtype=self.dtype, device=device
            )
        self.gdn_states = model.allocate_gdn_states(2 * self.max_num_reqs)
        self.sampling_state = Zonos3ARSamplingState(
            2 * self.max_num_reqs, model.config.eoa_id, model.config.audio_pad_id, device
        )
        self.next_embeddings = torch.zeros((2 * self.max_num_reqs, self._embed_dim), dtype=self.dtype, device=device)
        self.history = torch.full((self.max_num_reqs, 10, self.max_model_len), -1, dtype=torch.int64, device=device)
        self.settings = _SamplingSettings(
            **{
                name: UvaBackedTensor(self.max_num_reqs, dtype=dtype)
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
        self.options: list[Zonos3SamplingOptions | None] = [None] * self.max_num_reqs
        self._sampling_pending: set[int] = set()
        self._generators: dict[int, torch.Generator] = {}
        self._sample_output: SamplerOutput | None = None
        self._slots = torch.empty(self.max_num_reqs, dtype=torch.int64, device=device)
        self._batches: dict[int, Zonos3BackboneBatch] = {}
        self._prompts: dict[str, _Prompt] = {}
        self._lengths: dict[str, int] = {}
        self._request_slots: dict[str, int] = {}
        self._decode_started: set[str] = set()
        self._startup = True

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        if new_req_data.num_computed_tokens:
            raise ValueError("Zonos3 cannot restore a KV-only prefix or recomputed/preempted GDN state")
        if new_req_data.req_id in self._request_slots:
            raise ValueError("Zonos3 request was reinitialized without completion cleanup")
        params = new_req_data.sampling_params
        if is_warmup_request(new_req_data.req_id):
            # Upstream startup requests can ask for text-only logprobs/grammar.
            # They are not serving requests and have no physical RVQ contract.
            # Zonos3 emits only CONTINUE/EOS logical IDs, so the upstream text
            # sampler's dummy logprob request (typically top-20) can exceed
            # this tiny logical vocabulary during GPU startup warmup.
            params = SamplingParams(temperature=0.9, top_k=50, top_p=0.9, min_p=0.1, seed=0)
            params.logprobs = None
            params.prompt_logprobs = None
        options = Zonos3SamplingOptions.from_params(params, self.model.config, self.max_model_len)
        super().add_request(req_index, new_req_data)
        req_id = new_req_data.req_id
        self._request_slots[req_id] = req_index
        self._lengths[req_id] = len(new_req_data.prompt_token_ids)
        self._startup = self._startup and is_warmup_request(req_id)
        for state in self.gdn_states.values():
            state.reset([req_index])
        slots = index_to_device([req_index], self.device)
        self.sampling_state.reset(slots, torch.zeros(1, dtype=torch.bool, device=self.device))
        self.next_embeddings.index_fill_(0, slots, 0)
        self.history.index_fill_(0, slots, -1)
        self._stage_sampling_options(req_index, options)

    def _stage_sampling_options(self, req_index: int, options: Zonos3SamplingOptions) -> None:
        self.options[req_index] = options
        for field in fields(self.settings):
            getattr(self.settings, field.name).np[req_index] = getattr(options, field.name)
        self._sampling_pending.add(req_index)
        if self.device.type == "cpu":
            self._generators[req_index] = torch.Generator().manual_seed(options.seed)

    def apply_staged_writes(self) -> None:
        super().apply_staged_writes()
        if not self._sampling_pending:
            return
        for field in fields(self.settings):
            getattr(self.settings, field.name).copy_to_uva()
        slots = sorted(self._sampling_pending)
        indices = index_to_device(slots, self.device)
        self.sampling_state.ignore_eos.index_copy_(
            0, indices, index_to_device([self.options[i].ignore_eos for i in slots], self.device, dtype=torch.bool)
        )
        self.sampling_state.eoa_codebook0.index_copy_(
            0, indices, index_to_device([self.options[i].eoa_codebook0 for i in slots], self.device, dtype=torch.bool)
        )
        self._sampling_pending.clear()

    def _drop(self, req_id: str) -> None:
        self._prompts.pop(req_id, None)
        self._lengths.pop(req_id, None)
        slot = self._request_slots.pop(req_id, None)
        if slot is not None:
            self.options[slot] = None
            self._generators.pop(slot, None)
            self._sampling_pending.discard(slot)
        self._decode_started.discard(req_id)

    def remove_request(self, req_index_or_id: int | str) -> None:
        index = self._resolve_req_index(req_index_or_id)
        if index is not None:
            req_id = self.intermediate_buffer.buffers[index].get("req_id")
            if req_id is not None:
                self._drop(req_id)
        super().remove_request(req_index_or_id)

    def on_requests_finished(self, req_ids: set[str]) -> None:
        for req_id in req_ids:
            self._drop(req_id)
        super().on_requests_finished(req_ids)

    def on_request_preempted(self, req_id: str, req_index: int) -> None:
        del req_index
        raise RuntimeError(f"Zonos3 AR does not support preemption/state recomputation: {req_id}")

    def _compile_prompt(self, req_id: str, info: dict[str, Any]) -> _Prompt:
        length = self._lengths[req_id]
        if is_warmup_request(req_id):
            return _Prompt(torch.zeros((length, self._embed_dim), dtype=self.dtype, device=self.device), False)
        payload = info
        if not payload:
            raise ValueError("Missing Zonos3 prompt conditioning metadata")
        ids = payload.get("frame_ids")
        c = self.model.config
        if (
            not isinstance(ids, torch.Tensor)
            or ids.device.type != "cpu"
            or ids.dtype not in (torch.int32, torch.int64)
            or ids.shape != (length, 11)
        ):
            raise ValueError("Zonos3 placeholder length must exactly match CPU prompt frame_ids [length, 11]")
        if bool(((ids[:, :10] < 0) | (ids[:, :10] >= c.audio_vocab)).any()) or bool(
            ((ids[:, 10] < 0) | (ids[:, 10] >= c.text_rows)).any()
        ):
            raise ValueError("Zonos3 conditioning contains out-of-vocabulary frame IDs")
        embeddings = self.model.multi_embedder(to_device_nonblocking(ids, self.device))
        speaker = payload.get("speaker_embedding")
        waveform = payload.get("reference_waveform")
        if waveform is not None:
            encoder = getattr(self.model, "speaker_encoder", None)
            if (
                speaker is not None
                or encoder is None
                or not isinstance(waveform, torch.Tensor)
                or waveform.device.type != "cpu"
            ):
                raise ValueError("Zonos3 raw reference requires the model's ownmect_s2 encoder and CPU waveform")
            sample_rate = payload.get("reference_sample_rate")
            max_samples = int(self.model.config.speaker_reference_max_seconds * int(sample_rate or 0))
            if (
                not isinstance(sample_rate, int)
                or sample_rate <= 0
                or waveform.ndim not in (1, 2)
                or waveform.numel() == 0
                or waveform.shape[-1] > max_samples
            ):
                raise ValueError(
                    "Zonos3 raw speaker reference must be non-empty and no longer than "
                    f"{self.model.config.speaker_reference_max_seconds:g} seconds"
                )
            # _compile_prompt runs once per request; chunked prefill slices the
            # cached embeddings and never repeats this expensive extraction.
            speaker = encoder(waveform, sample_rate).squeeze(0)
        if speaker is not None:
            row = payload.get("speaker_row")
            if (
                not isinstance(row, int)
                or not 0 <= row < length
                or not isinstance(speaker, torch.Tensor)
                or (waveform is None and speaker.device.type != "cpu")
                or speaker.shape != (c.speaker_embedding_dim,)
            ):
                raise ValueError("Invalid Zonos3 speaker embedding row")
            embeddings[row] = self.model.speaker_projection(to_device_nonblocking(speaker, self.device).to(self.dtype))
        buckets, start = payload.get("condition_buckets"), payload.get("condition_start")
        if (
            not isinstance(buckets, (list, tuple))
            or len(buckets) != c.condition_count
            or not isinstance(start, int)
            or not 0 <= start <= length - c.condition_count
        ):
            raise ValueError("Invalid Zonos3 condition rows")
        for i, (bucket, count) in enumerate(zip(buckets, c.condition_bucket_counts, strict=True)):
            if not isinstance(bucket, int) or not 0 <= bucket <= count:
                raise ValueError("Invalid Zonos3 condition bucket ID")
            embeddings[start + i] = self.model.condition_embeddings[i].weight[bucket]
        return _Prompt(embeddings, bool(payload.get("has_audio_prefix", False)))

    def _batch(self, tokens: int, decode: bool) -> Zonos3BackboneBatch:
        batch = self._batches.get(tokens)
        if batch is None:
            batch = Zonos3BackboneBatch(
                self.gdn_states,
                self.model.allocate_workspace(tokens),
                self._slots[:tokens],
                torch.arange(tokens, device=self.device),
                decode_only=decode,
            )
        batch.decode_only = decode
        return batch

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        if not 0 < num_reqs <= self.max_num_reqs or num_tokens > self.max_num_tokens:
            raise ValueError("Invalid Zonos3 native dummy batch shape")
        slots = list(range(self.max_num_reqs, self.max_num_reqs + num_reqs))
        for state in self.gdn_states.values():
            state.reset(slots)
        self._static_inputs_embeds[:num_tokens].zero_()
        batch = self._batch(num_tokens, num_reqs == num_tokens)
        if batch.decode_only:
            # Retain the actual capture workspace, even if upstream finalizes
            # capture sizes after ModelState initialization. Graphs keep raw
            # pointers to it; eager prefill shapes are deliberately not cached.
            self._batches[num_tokens] = batch
            self._slots[:num_tokens].copy_(index_to_device(slots, self.device))
            batch.decode_slots = self._slots[:num_tokens]
            batch.prefill_spans = ()
        else:
            lengths = [num_tokens // num_reqs] * num_reqs
            for i in range(num_tokens % num_reqs):
                lengths[-1 - i] += 1
            spans, start = [], 0
            for slot, length in zip(slots, lengths, strict=True):
                spans.append(Zonos3PrefillSpan(start, length, slot, 0))
                start += length
            batch.prefill_spans = tuple(spans)
            batch.decode_slots = torch.empty(0, dtype=torch.int64, device=self.device)
            batch.decode_rows = batch.decode_slots
        return {"inputs_embeds": self._static_inputs_embeds[:num_tokens], "zonos3_batch": batch}

    def prepare_inputs(self, input_batch: InputBatch, req_states: RequestState) -> dict[str, Any]:
        result = super().prepare_inputs(input_batch, req_states)
        if self._startup and all(req_id not in self._request_slots for req_id in input_batch.req_ids):
            result.update(self.prepare_dummy_inputs(input_batch.num_reqs, input_batch.num_tokens_after_padding))
        return result

    def run_preprocess(
        self,
        input_batch: InputBatch,
        model_inputs: dict[str, Any],
        req_states: RequestState | None = None,
        mtp_batch_descriptor_dispatcher: Callable[[int], Any] | None = None,
    ) -> None:
        tokens = input_batch.num_tokens_after_padding
        embeddings = self._static_inputs_embeds[:tokens]
        embeddings.zero_()
        one_token = bool(np.all(input_batch.num_scheduled_tokens == 1))
        decode_only = one_token and input_batch.decode_graph_eligible
        batch = self._batch(tokens, decode_only)
        spans, decode_slots, decode_rows = [], [], []
        for row, req_id in enumerate(input_batch.req_ids):
            slot = int(input_batch.idx_mapping_np[row])
            if self._request_slots.get(req_id) != slot:
                raise RuntimeError("Zonos3 request/slot identity mismatch")
            start = int(input_batch.query_start_loc_np[row])
            length = int(input_batch.num_scheduled_tokens[row])
            if input_batch.is_prefilling_np[row]:
                if req_id in self._decode_started:
                    raise RuntimeError("Zonos3 cannot resume prefill after AR decode")
                prompt = self._prompts.get(req_id)
                if prompt is None:
                    prompt = self._compile_prompt(req_id, self.intermediate_buffer.buffers[slot])
                    self._prompts[req_id] = prompt
                    self.sampling_state.has_audio_prefix[slot] = prompt.has_audio_prefix
                offset = int(input_batch.num_computed_prefill_tokens_np[row])
                if (
                    offset + length > prompt.embeddings.shape[0]
                    or int(input_batch.prefill_len_np[row]) != prompt.embeddings.shape[0]
                ):
                    raise ValueError("Zonos3 scheduled prefill differs from its exact logical prompt length")
                embeddings[start : start + length].copy_(prompt.embeddings[offset : offset + length])
                if decode_only:
                    # One-token prefills may use a native decode graph. Update
                    # the CPU prefill frontier without reading recurrent state.
                    for state in self.gdn_states.values():
                        state.prefill_tokens[slot] = offset + 1
                else:
                    spans.append(Zonos3PrefillSpan(start, length, slot, offset))
            else:
                if length != 1 or req_id not in self._prompts:
                    raise ValueError("Zonos3 AR decode needs one logical token after conditioning prefill")
                self._decode_started.add(req_id)
                decode_slots.append(slot)
                decode_rows.append(start)
        if decode_rows:
            rows = index_to_device(decode_rows, self.device)
            slots = index_to_device(decode_slots, self.device)
            embeddings.index_copy_(0, rows, self.next_embeddings.index_select(0, slots))
        if decode_only:
            count = input_batch.num_reqs
            if tokens > self.max_num_reqs:
                raise ValueError("Zonos3 decode graph padding exceeds its configured request capacity")
            slots = [int(slot) for slot in input_batch.idx_mapping_np] + list(
                range(self.max_num_reqs, self.max_num_reqs + tokens - count)
            )
            self._slots[:tokens].copy_(index_to_device(slots, self.device))
            batch.decode_slots, batch.prefill_spans = self._slots[:tokens], ()
        else:
            batch.prefill_spans = tuple(spans)
            batch.decode_slots = index_to_device(decode_slots, self.device)
            batch.decode_rows = index_to_device(decode_rows, self.device)
        model_inputs["inputs_embeds"] = embeddings
        model_inputs["zonos3_batch"] = batch

    def _setting(self, name: str, slots: torch.Tensor) -> torch.Tensor:
        backed: UvaBackedTensor = getattr(self.settings, name)
        if backed.gpu.device != slots.device:
            values = backed.np[slots.detach().cpu().numpy()]
            return torch.from_numpy(values.copy()).to(slots.device)
        return backed.gpu.index_select(0, slots)

    def probabilities(self, logits: torch.Tensor, slots: torch.Tensor, width: int) -> torch.Tensor:
        """Per-codebook filters, using the same upstream kernels as before."""
        c = self.model.config
        logits = logits.float()
        positions = self.sampling_state.position.index_select(0, slots)
        ids = torch.arange(c.audio_vocab, device=logits.device)
        invalid = (ids >= c.codebook_size) & (ids != c.eoa_id)
        minimum = self._setting("min_frames", slots)
        stop_blocked = (minimum[:, None] > 0) & (positions[:, None] < minimum[:, None] + self.sampling_state.columns)
        logits = logits.masked_fill(
            invalid[None, None, :] | (stop_blocked[:, :, None] & (ids == c.eoa_id)), float("-inf")
        )
        if width:
            offsets = torch.arange(width, device=logits.device)
            indices = (positions[:, None] - 1 - offsets) % self.max_model_len
            history = self.history[
                slots[:, None, None], self.sampling_state.columns[None, :, None], indices[:, None, :]
            ]
            window = self._setting("repetition_window", slots)
            books = self._setting("repetition_codebooks", slots)
            valid = (history >= 0) & (history < c.codebook_size)
            valid = valid & (offsets[None, None, :] < torch.minimum(positions, window)[:, None, None])
            valid = valid & (self.sampling_state.columns[None, :, None] < books[:, None, None])
            repeated = torch.zeros_like(logits, dtype=torch.int32)
            repeated.scatter_add_(-1, history.clamp(min=0, max=c.audio_vocab - 1), valid.to(torch.int32))
            penalty = self._setting("repetition_penalty", slots)[:, None, None]
            adjusted = torch.where(logits > 0, logits / penalty, logits * penalty)
            logits = torch.where(repeated > 0, adjusted, logits)
        temperature = self._setting("temperature", slots)[:, None, None]
        logits = logits / torch.where(temperature > 0, temperature, 1.0)
        # Triton expects contiguous per-codebook settings, including B=1.
        k = self._setting("top_k", slots).repeat_interleave(10)
        p = self._setting("top_p", slots).repeat_interleave(10)
        flat = logits.reshape(-1, c.audio_vocab)
        if flat.is_cuda:
            logits = apply_top_k_top_p(flat, k, p, has_topp_only_rows=True)
        else:
            logits = apply_top_k_top_p_pytorch(flat, k.to(flat.device), p.to(flat.device))
        logits = logits.view(-1, 10, c.audio_vocab)
        probs = logits.softmax(-1)
        min_p = self._setting("min_p", slots)[:, None, None]
        probs = probs.masked_fill(probs < min_p * probs.amax(-1, keepdim=True), 0)
        total = probs.sum(-1, keepdim=True)
        fallback = torch.zeros_like(probs).scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
        return torch.where(total > 0, probs / total.clamp(min=1e-8), fallback)

    def _draw_cpu(self, probs: torch.Tensor, indices: list[int], active: torch.Tensor) -> torch.Tensor:
        result = torch.full((len(indices), 10), self.model.config.audio_pad_id, dtype=torch.int64)
        for row, slot in enumerate(indices):
            if bool(active[row]):
                if self.options[slot].temperature == 0:
                    result[row] = probs[row].argmax(-1)
                else:
                    result[row] = torch.multinomial(
                        probs[row, :, : self.model.config.audio_pad_id + 1], 1, generator=self._generators[slot]
                    ).flatten()
        return result

    def postprocess_model_output(
        self, model_output: Any, input_batch: InputBatch, req_states: RequestState
    ) -> tuple[torch.Tensor, dict]:
        hidden, _ = super().postprocess_model_output(model_output, input_batch, req_states)
        if input_batch.num_draft_tokens or input_batch.logits_indices.numel() != input_batch.num_reqs:
            raise ValueError("Zonos3 RVQ sampling does not support speculative scheduler tokens")
        if self._sampling_pending:
            raise RuntimeError("Zonos3 sampling staged writes must be applied before postprocess")
        slots = input_batch.idx_mapping
        options = [self.options[int(slot)] for slot in input_batch.idx_mapping_np]
        if any(option is None for option in options):
            raise RuntimeError("Zonos3 sampling request was not initialized")
        if hidden.device.type == "cuda":
            sampled, rejected = get_num_sampled_and_rejected(
                input_batch.seq_lens.new_ones(input_batch.num_reqs),
                input_batch.seq_lens,
                input_batch.cu_num_logits,
                slots,
                req_states.prefill_len.gpu,
            )
        else:
            prefill_len = req_states.prefill_len.gpu.to(device=hidden.device)
            sampled = (input_batch.seq_lens >= prefill_len.index_select(0, slots)).to(torch.int32)
            rejected = torch.zeros_like(sampled)
        active = (sampled > 0) & ~(
            (self.sampling_state.eoa_frame.index_select(0, slots) >= 0)
            & (self.sampling_state.drain_remaining.index_select(0, slots) == 0)
        )
        position = self.sampling_state.position.index_select(0, slots)
        logits = self.model.multi_output(hidden.index_select(0, input_batch.logits_indices))
        probs = self.probabilities(logits, slots, max(option.repetition_window for option in options))
        if hidden.device.type == "cuda":
            noise_position = (position[:, None] * 10 + self.sampling_state.columns).reshape(-1)
            codes = gumbel_sample(
                probs.log().reshape(-1, self.model.config.audio_vocab),
                slots.repeat_interleave(10),
                self.settings.temperature.gpu,
                self.settings.seed.gpu,
                noise_position,
                apply_temperature=False,
                is_drafting=False,
            ).view(-1, 10)
        elif hidden.device.type == "cpu":
            codes = self._draw_cpu(probs, [int(slot) for slot in input_batch.idx_mapping_np], active)
        else:
            raise ValueError("Zonos3 sampling supports CUDA serving and CPU numerical checks")
        # Keep the producer contract simple: prefill rows and any scheduler
        # step after the GPU-side EOS drain publish no RVQ row. The EOS step
        # itself remains active and its raw codes are delivered to the processor.
        # The row is represented by padding because the generic runner still
        # partitions one output row per scheduled request token.
        codes = torch.where(active[:, None], codes, self.model.config.audio_pad_id)
        step = self.sampling_state.accept(slots, codes, active)
        ring = position[:, None] % self.max_model_len
        old = self.history[slots[:, None], self.sampling_state.columns[None, :], ring]
        self.history[slots[:, None], self.sampling_state.columns[None, :], ring] = torch.where(
            step.active[:, None], step.codes, old
        )
        frame = torch.cat((step.codes, torch.full_like(step.codes[:, :1], self.model.config.text_vocab)), -1)
        embeddings = self.model.multi_embedder(frame)
        previous = self.next_embeddings.index_select(0, slots)
        self.next_embeddings.index_copy_(0, slots, torch.where(step.active[:, None], embeddings, previous))
        self._sample_output = SamplerOutput(step.logical_token_ids.long(), None, None, sampled, rejected)
        return hidden, Zonos3CodeSnapshot(step.codes)

    def sample_determined_tokens(self, batch: InputBatch, sampler: Sampler) -> SamplerOutput:
        """Deliver the CONTINUE/EOS decided in postprocess without text sampling."""
        output = self._sample_output
        if output is None or output.sampled_token_ids.shape != (batch.num_reqs, 1):
            raise RuntimeError("Zonos3 determined tokens require the current step's postprocess output")
        return output

    def warmup_sampling(self, hidden_states: torch.Tensor, req_states: RequestState) -> None:
        """Exercise the serving sampling path before requests, then reset its state."""
        rows = min(hidden_states.shape[0], self.max_num_reqs)
        buckets = self.vllm_config.compilation_config.cudagraph_capture_sizes or ()
        sizes = sorted({1, rows, *(size for size in buckets if 0 < size <= rows)})
        options = Zonos3SamplingOptions.from_params(
            SamplingParams(temperature=0.9, top_k=50, top_p=0.9, min_p=0.1, repetition_penalty=1.2, seed=0),
            self.model.config,
            self.max_model_len,
        )
        slots = index_to_device(list(range(rows)), self.device)
        buffers = InputBuffers(rows, rows, self.device)
        try:
            for slot in range(rows):
                self._stage_sampling_options(slot, options)
            self.apply_staged_writes()
            for size in sizes:
                self.sampling_state.reset(slots, torch.zeros(rows, dtype=torch.bool, device=self.device))
                self.history.index_fill_(0, slots, -1)
                batch = InputBatch.make_dummy(size, size, buffers)
                self.postprocess_model_output(hidden_states[:size], batch, req_states)
        finally:
            self.sampling_state.reset(slots, torch.zeros(rows, dtype=torch.bool, device=self.device))
            self.history.index_fill_(0, slots, -1)
            self.next_embeddings.index_fill_(0, slots, 0)
            for slot in range(rows):
                self.options[slot] = None
                self._generators.pop(slot, None)
                self._sampling_pending.discard(slot)
            self._sample_output = None

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        warmup_only = bool(input_batch.req_ids) and all(is_warmup_request(req_id) for req_id in input_batch.req_ids)
        if (
            not for_capture
            and not warmup_only
            and not input_batch.has_prefill
            and cudagraph_mode is not CUDAGraphMode.FULL
        ):
            raise RuntimeError("Zonos3 steady-state AR decode requires one native FULL Backbone graph replay")
        return super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
            ubatch_idx=ubatch_idx,
            model_specific_attn_metadata=model_specific_attn_metadata,
        )


class Zonos3CodecModelState(OmniModelState):
    def on_request_preempted(self, req_id: str, req_index: int) -> None:
        del req_index
        raise RuntimeError(f"Zonos3 Codec does not support preemption/state recomputation: {req_id}")
