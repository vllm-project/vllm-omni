# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gepard-1.0 native-AR talker.

Single-stage AR TTS on a vLLM-native Qwen3.5 backbone. Each step samples one
32-channel FSQ frame; head0 is the vLLM-facing token and the other 31 are
side-channel. Generation ends on a learned binary stop head, not an EOS token.
The NeMo NanoCodec decodes committed frames to a waveform outside vLLM.

Zero-shot only, and ``enforce_eager``; CUDA graph is a perf follow-up.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.gepard.configuration_gepard import GepardConfig
from vllm_omni.model_executor.models.gepard.nanocodec import NanoCodec
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

# Decode cadence: a short first chunk for early audio, then larger ones to
# amortize the codec call. Does not affect the samples produced.
FIRST_CHUNK_FRAMES = int(os.environ.get("VLLM_GEPARD_FIRST_CHUNK_FRAMES", "10"))
CHUNK_FRAMES = int(os.environ.get("VLLM_GEPARD_CHUNK_FRAMES", "86"))


@dataclasses.dataclass(frozen=True)
class _ReqInfo:
    """Scheduler state the runner threads into ``preprocess``.

    The hook key strings are read only here, so a rename fails in one place.
    """

    request_id: str
    is_prefill: bool
    prompt_len: int
    num_computed_tokens: int
    max_tokens: int | None
    seed: int | None

    @classmethod
    def from_hook_kwargs(cls, kwargs: dict[str, Any], *, span: int) -> _ReqInfo:
        # Defaults cover direct calls, which thread no scheduler state.
        max_tokens = kwargs.get("_omni_max_tokens")
        seed = kwargs.get("_omni_seed")
        return cls(
            request_id=str(kwargs.get("request_id", "default")),
            is_prefill=bool(kwargs.get("_omni_is_prefill", span > 1)),
            prompt_len=int(kwargs.get("_omni_prompt_len") or 0),
            num_computed_tokens=int(kwargs.get("_omni_num_computed_tokens") or 0),
            max_tokens=None if max_tokens is None else int(max_tokens),
            seed=None if seed is None else int(seed),
        )


@dataclasses.dataclass
class _GepardState:
    """Per-request generation state, keyed by request_id."""

    request_id: str
    # Embedding of the last frame's 32 codes; the next step's input.
    curr_embed_for_next: torch.Tensor | None = None
    # Frame history (T, 32) handed to the codec.
    frames: list[torch.Tensor] = dataclasses.field(default_factory=list)
    emitted_frames: int = 0
    # None falls back to null_prefix; the cloning follow-up sets it.
    speaker_prefix: torch.Tensor | None = None
    # None means the request carried no seed and draws from the global RNG.
    generator: torch.Generator | None = None
    frame_count: int = 0
    past_first_step: bool = False
    is_stopping: bool = False


class GepardTalkerForConditionalGeneration(nn.Module):
    """Gepard native-AR TTS talker (Qwen3.5 backbone + 32 FSQ heads + stop)."""

    have_multimodal_outputs = True
    has_preprocess = True
    has_postprocess = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        cfg = vllm_config.model_config.hf_config
        if not isinstance(cfg, GepardConfig):
            # hf_overrides patches the loaded config instance rather than
            # re-classing it, so what arrives here is the backbone config.
            cfg = GepardConfig.from_checkpoint(
                vllm_config.model_config.model,
                backbone_config=cfg.to_dict(),
                revision=vllm_config.model_config.revision,
            )
        self.config = cfg
        self._device = current_omni_platform.get_torch_device()
        # Deterministic argmax sampling, for reproducible comparisons.
        self._greedy = os.environ.get("VLLM_GEPARD_GREEDY", "0") == "1"

        self.vocab_sizes: list[int] = list(cfg.audio_head_levels)  # [8,7,6,6]*8
        self.num_heads: int = cfg.num_audio_heads
        self.head0_vocab: int = cfg.head0_vocab_size
        self.stop_token: int = cfg.stop_token  # sentinel, one past head0's range
        self.total_vocab: int = sum(self.vocab_sizes)
        self.stop_threshold: float = cfg.stop_threshold
        self.temperature: float = cfg.temperature
        hidden = self.config.get_text_config().hidden_size

        self.fused_codebook_head = nn.Linear(hidden, self.total_vocab, bias=True)
        self.stop_head = nn.Linear(hidden, 1, bias=True)
        self.audio_embeddings = nn.ModuleList([nn.Embedding(v, cfg.audio_embed_dim) for v in self.vocab_sizes])
        self.audio_embed_proj = nn.Sequential(
            nn.Linear(self.num_heads * cfg.audio_embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden, elementwise_affine=False),
        )
        self.register_buffer("audio_embed_scale", torch.tensor(1.0))
        # Learned default-voice fallback.
        self.null_prefix = nn.Parameter(torch.zeros(cfg.num_speaker_prefix, hidden))

        gather_idx, mask = self._build_gather_mask(self.vocab_sizes)
        self.register_buffer("_cb_gather_idx", gather_idx, persistent=False)
        self.register_buffer("_cb_mask", mask, persistent=False)

        self.model = Qwen3_5ForCausalLM(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        # Cheap and NeMo-free to construct; load() does the heavy import.
        self._codec = NanoCodec(codec_id=cfg.codec_id, sample_rate=cfg.codec_sample_rate)

        self._max_model_len: int = int(vllm_config.model_config.max_model_len or 0)

        self._active_states: dict[str, _GepardState] = {}
        # These queues stay index-aligned across preprocess -> forward ->
        # compute_logits: preprocess appends in input_batch.req_ids order.
        # (req_id, span, samples_frame, is_last_token)
        self._pending_requests: list[tuple[str, int, bool, bool]] = []
        # (req_id, head0, do_stop); head0 < 0 marks a discarded placeholder row.
        self._results_queue: list[tuple[str, int, bool]] = []
        self._audio_queue: list[tuple[str, torch.Tensor | None]] = []
        self._deferred_cleanup_ids: set[str] = set()
        self._sample_rate: int = cfg.codec_sample_rate

    @staticmethod
    def _build_gather_mask(vocab_sizes: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather indices + -inf pad mask so one argmax samples all 31 heads."""
        num_heads = len(vocab_sizes)
        max_cb_vocab = max(vocab_sizes[1:])  # == 8 for Gepard; do NOT hardcode 7
        gather_rows, mask_rows = [], []
        cum = vocab_sizes[0]
        for i in range(1, num_heads):
            v = vocab_sizes[i]
            gather_rows.append(list(range(cum, cum + v)) + [cum] * (max_cb_vocab - v))
            mask_rows.append([0.0] * v + [float("-inf")] * (max_cb_vocab - v))
            cum += v
        return (
            torch.tensor(gather_rows, dtype=torch.long),
            torch.tensor(mask_rows, dtype=torch.float32),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Fuse codebook_heads.{0..31} into fused_codebook_head; delegate the rest."""
        cb_w: dict[str, torch.Tensor] = {}
        rest: list[tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            if name.startswith(("aligner.", "supcon_head.")):
                continue  # training-only
            if name.startswith("codebook_heads."):
                cb_w[name] = tensor
                continue
            if name.startswith(("model.", "lm_head.")):
                # We nest the backbone one level deeper than the checkpoint
                # keys assume, so prepend the extra hop.
                name = "model." + name
            rest.append((name, tensor))

        w_parts, b_parts = [], []
        for i in range(self.num_heads):
            if (wk := f"codebook_heads.{i}.weight") in cb_w:
                w_parts.append(cb_w[wk])
            if (bk := f"codebook_heads.{i}.bias") in cb_w:
                b_parts.append(cb_w[bk])
        if w_parts:
            rest.append(("fused_codebook_head.weight", torch.cat(w_parts, dim=0)))
        if b_parts:
            rest.append(("fused_codebook_head.bias", torch.cat(b_parts, dim=0)))

        # NanoCodec loads lazily on first decode, so load_format=dummy and
        # NeMo-less environments never pay for it.
        loader = AutoWeightsLoader(self, skip_prefixes=["mtp.", "ref_compressor."])
        return loader.load_weights(iter(rest))

    def _get_or_create_state(self, request_id: str) -> _GepardState:
        st = self._active_states.get(request_id)
        if st is None:
            st = _GepardState(request_id=request_id)
            self._active_states[request_id] = st
        return st

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        # Fires BEFORE forward -> defer the actual free until after forward runs,
        # else we'd drop state the in-flight forward still reads.
        self._deferred_cleanup_ids.update(str(r) for r in finished_req_ids)

    def _flush_deferred_cleanup(self) -> None:
        # Cleanup only. Audio queued here is never delivered — the id has left
        # the output batch by now — so the tail is emitted in forward instead,
        # and anything still pending here is genuinely lost. Say so.
        for req_id in self._deferred_cleanup_ids:
            state = self._active_states.pop(req_id, None)
            if state is not None and state.frame_count > state.emitted_frames:
                logger.warning(
                    "Gepard request %s ended with %d undelivered frame(s): it finished "
                    "without the stop head firing and without reaching its token budget "
                    "(aborted or preempted). Its audio is truncated.",
                    req_id,
                    state.frame_count - state.emitted_frames,
                )
        self._deferred_cleanup_ids.clear()

    def _is_last_output_token(self, info: _ReqInfo, *, span: int, samples_frame: bool) -> bool:
        """Whether the token sampled this step is the request's budget-final one.

        That step is the last one on which the request is still in the output
        batch, so it is also the last chance to ship its audio. Counted off the
        engine's cursor rather than a local tally, so a re-prefill after
        preemption cannot desync it.
        """
        if not samples_frame:
            # A partial prefill chunk's token is discarded, so it costs no budget.
            return False
        budget = info.max_tokens
        if self._max_model_len:
            len_cap = self._max_model_len - info.prompt_len
            budget = len_cap if budget is None else min(budget, len_cap)
        if budget is None or budget <= 0:
            return False
        produced = info.num_computed_tokens + span - info.prompt_len + 1
        return produced >= budget

    def _emit_audio(self, state: _GepardState, *, is_final: bool) -> None:
        """Decode this request's un-emitted frames and queue the new samples.

        Queues deltas: each call contributes only samples not queued before.
        """
        pending = state.frame_count - state.emitted_frames
        if pending <= 0:
            return
        if not is_final:
            threshold = FIRST_CHUNK_FRAMES if state.emitted_frames == 0 else CHUNK_FRAMES
            if pending < threshold:
                return

        frames = torch.stack(state.frames)  # (T, 32)
        if not self._codec.is_loaded:
            # The first decode pays the codec load; warmup is a perf follow-up.
            self._codec.load(frames.device)
        audio = self._codec.decode_stream(frames, start_idx=state.emitted_frames, is_final=is_final)
        if audio is None:
            return
        self._audio_queue.append((state.request_id, audio))
        state.emitted_frames = state.frame_count

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    @staticmethod
    def _gumbel_noise(ref: torch.Tensor, generators: list[torch.Generator | None] | None) -> torch.Tensor:
        """Gumbel noise shaped like ``ref``; row i uses ``generators[i]`` if set.

        A seeded row draws its own noise of a fixed shape, so the noise it sees
        does not depend on who shared its batch. Only the noise: the logits come
        from a bf16 matmul that is not row-wise reproducible, so a batched
        request can still sample differently from a solo one.
        """
        if not generators or all(g is None for g in generators):
            u = torch.rand_like(ref)
        else:
            u = torch.stack(
                [
                    torch.rand(ref.shape[1:], generator=g, device=ref.device, dtype=ref.dtype)
                    if g is not None
                    else torch.rand(ref.shape[1:], device=ref.device, dtype=ref.dtype)
                    for g in generators
                ]
            )
        return -torch.log(-torch.log(u.clamp_min_(1e-20)))

    def _sample_frame(
        self,
        hidden: torch.Tensor,
        generators: list[torch.Generator | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(N, hidden) -> head0 (N,), heads_1_31 (N, 31), do_stop (N,).

        ``generators`` is row-aligned with ``hidden``. The 32 heads are sampled
        here rather than by vLLM's sampler, so a request's seed only reaches
        the audio through this.
        """
        fused = self.fused_codebook_head(hidden)

        gathered = fused[:, self._cb_gather_idx].float() + self._cb_mask  # (N,31,8)
        h0_logits = fused[:, : self.head0_vocab].float()
        if self._greedy:
            heads = gathered.argmax(dim=-1)
            head0 = h0_logits.argmax(dim=-1)
        else:
            heads = (gathered / self.temperature + self._gumbel_noise(gathered, generators)).argmax(dim=-1)
            head0 = (h0_logits / self.temperature + self._gumbel_noise(h0_logits, generators)).argmax(dim=-1)

        p_stop = torch.sigmoid(self.stop_head(hidden)).squeeze(-1)
        do_stop = p_stop > self.stop_threshold
        return head0, heads, do_stop

    def _audio_frame_embed(self, head0: torch.Tensor, heads_1_31: torch.Tensor) -> torch.Tensor:
        """Rebuild the next step's input embedding from the 32 sampled codes."""
        codes = torch.cat([head0.clamp(max=self.head0_vocab - 1).unsqueeze(-1), heads_1_31], dim=-1)
        parts = [self.audio_embeddings[i](codes[:, i]) for i in range(self.num_heads)]
        return self.audio_embed_proj(torch.cat(parts, dim=-1)) * self.audio_embed_scale

    def preprocess(self, input_ids: torch.Tensor, input_embeds: torch.Tensor | None = None, **req_infos: Any):
        # Prefill injects the speaker prefix at the placeholder slots. The
        # prompt ends on SOS, so forward samples the first frame there.
        dev = input_ids.device
        span = int(input_ids.shape[0])
        info = _ReqInfo.from_hook_kwargs(req_infos, span=span)
        req_id = info.request_id
        is_prefill = info.is_prefill
        prompt_len = info.prompt_len
        computed = info.num_computed_tokens
        # A prefill chunk that does not finish the prompt commits no frame:
        # vLLM discards its sampled token.
        samples_frame = (not is_prefill) or prompt_len <= 0 or computed + span >= prompt_len

        if is_prefill:
            state = self._get_or_create_state(req_id)
            # An aborted id can be resubmitted, and the runner treats the two as
            # distinct requests. Drop the pending free, or the cleanup at the
            # end of this forward takes the new request's state with it.
            resubmitted = req_id in self._deferred_cleanup_ids
            self._deferred_cleanup_ids.discard(req_id)
            if resubmitted:
                # Stand in for the cleanup just cancelled.
                undelivered = state.frame_count - state.emitted_frames
                if undelivered > 0:
                    logger.warning(
                        "Gepard request %s ended with %d undelivered frame(s) and its id has been "
                        "resubmitted; the first request's audio is truncated.",
                        req_id,
                        undelivered,
                    )
            elif state.frame_count:
                # Not a resubmit, so this is a preemption recompute, and it
                # cannot be recovered: a frame's 31 side-channel codes are not
                # in the token stream, so the replayed prompt cannot rebuild the
                # KV that produced the audio already delivered. Generation
                # restarts from frame 0 and the caller gets two unrelated halves.
                logger.warning(
                    "Gepard request %s was preempted after %d frame(s) and is being recomputed; "
                    "its audio will not continue from where it stopped. Raise the stage's "
                    "gpu_memory_utilization or lower max_num_seqs to keep requests from being "
                    "preempted.",
                    req_id,
                    state.frame_count,
                )
            state.curr_embed_for_next = None
            state.frames.clear()
            state.emitted_frames = 0
            state.frame_count = 0
            state.past_first_step = False
            state.is_stopping = False
            # SamplingParams.seed only reaches vLLM's sampler, so it never
            # touches the audio unless it is threaded into a generator here.
            state.generator = None
            if info.seed is not None:
                state.generator = torch.Generator(device=dev)
                state.generator.manual_seed(info.seed)

            embeds = self.model.embed_input_ids(input_ids)
            prefix = state.speaker_prefix if state.speaker_prefix is not None else self.null_prefix
            base = self.config.speaker_token_base
            spk = (input_ids >= base) & (input_ids < base + self.config.num_speaker_prefix)
            if bool(spk.any()):
                embeds[spk] = prefix[input_ids[spk] - base].to(dtype=embeds.dtype)
        else:
            state = self._active_states.get(req_id)
            curr = state.curr_embed_for_next if state is not None else None
            if curr is not None:
                embeds = curr.to(device=dev).reshape(1, -1)
            else:
                # Defensive only: a decode step always follows a forward that
                # stashed curr_embed_for_next.
                hidden = self.config.get_text_config().hidden_size
                embeds = torch.zeros(1, hidden, device=dev, dtype=self.null_prefix.dtype)

        is_last_token = self._is_last_output_token(info, span=span, samples_frame=samples_frame)
        self._pending_requests.append((req_id, span, samples_frame, is_last_token))
        return input_ids, embeds, {}

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | OmniOutput:
        # positions is plain 1D: Gepard uses default RoPE, not mRoPE. input_ids
        # passes through only when no embeds were spliced (the profile run).
        hidden = self.model(
            input_ids=input_ids if inputs_embeds is None else None,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        # Empty _pending_requests means the engine-init profile run: skip.
        if self._pending_requests:
            # One sampling row per request: the last scheduled position of its
            # span. Same row set compute_logits sees at logits_indices.
            ends, generators, offset = [], [], 0
            for req_id, span, _samples_frame, _is_last in self._pending_requests:
                offset += span
                ends.append(offset - 1)
                state = self._active_states.get(req_id)
                generators.append(state.generator if state is not None else None)
            rows = hidden[torch.tensor(ends, device=hidden.device)]
            head0, heads_1_31, want_stop = self._sample_frame(rows, generators)
            frame_embeds = self._audio_frame_embed(head0, heads_1_31)
            head0_l = head0.tolist()
            stop_l = want_stop.tolist()
            for i, (req_id, _span, samples_frame, is_last_token) in enumerate(self._pending_requests):
                if not samples_frame:
                    # Partial prefill chunk: vLLM discards this row's token.
                    self._results_queue.append((req_id, -1, False))
                    continue
                state = self._get_or_create_state(req_id)
                if state.is_stopping:
                    # Async scheduling runs one more step after a stop token —
                    # only the max_tokens finish is guarded against it. This
                    # step's output is discarded, so a frame committed here
                    # would be stranded and reported as lost audio.
                    self._results_queue.append((req_id, int(head0_l[i]), True))
                    continue
                # Stop may only fire past the first audio frame, so the
                # prefill-sampled frame and the one after it always commit.
                do_stop = bool(stop_l[i]) and state.past_first_step
                if do_stop:
                    state.is_stopping = True
                else:
                    state.frames.append(torch.cat([head0[i : i + 1], heads_1_31[i]]))
                    state.frame_count += 1
                    state.curr_embed_for_next = frame_embeds[i]
                    state.past_first_step = True
                self._results_queue.append((req_id, int(head0_l[i]), do_stop))
                # Both ways a request ends have to flush here: this is the last
                # step on which it is still in the output batch, and a payload
                # queued after it is dropped by routing.
                self._emit_audio(state, is_final=do_stop or is_last_token)
            self._pending_requests.clear()
        self._flush_deferred_cleanup()
        return hidden

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None):
        # Vocab-wide logits carrying head0, or STOP for a stopped row. Requires
        # the pipeline's sampling_constraints, without which vLLM never acts on
        # the sentinel.
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        bsz = hidden_states.shape[0]
        vocab = self.config.get_text_config().vocab_size
        logits = torch.full((bsz, vocab), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype)
        if not self._results_queue:
            # Engine-init dummy sampler run: neutral one-hot logits so KV-cache
            # profiling can complete.
            logits[:, 0] = 0.0
            return logits
        # Row i belongs to _results_queue[i]. The pipeline pins temperature=0.0,
        # so vLLM's sampler argmaxes the single finite entry.
        for i, (_req_id, head0, do_stop) in enumerate(self._results_queue):
            if i >= bsz:
                break
            if head0 < 0:
                logits[i, 0] = 0.0  # partial-prefill placeholder; vLLM discards it
                continue
            logits[i, self.stop_token if do_stop else head0] = 0.0
        self._results_queue.clear()
        return logits

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        by_req: dict[str, torch.Tensor] = {}
        for req_id, audio in self._audio_queue:
            if audio is None:
                continue
            by_req[req_id] = (
                torch.cat([by_req[req_id].reshape(-1), audio.reshape(-1)], dim=0) if req_id in by_req else audio
            )
        self._audio_queue.clear()

        # The sparse marker is always carried, even on a step that decoded
        # nothing: without it the runner ships the pooler payload instead, whose
        # hidden states land under this stage's "audio" key.
        ready_req_ids = list(by_req)
        sr = torch.tensor(self._sample_rate, dtype=torch.int32)
        mm: dict[str, Any] = {
            "model_outputs": [by_req[r].reshape(-1) for r in ready_req_ids],
            "sr": [sr for _ in ready_req_ids],
            "meta": {"req_id": ready_req_ids, "sparse_audio": ["1"]},
        }
        # Deltas, not the whole history: callers concatenate across steps.
        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs=mm)
