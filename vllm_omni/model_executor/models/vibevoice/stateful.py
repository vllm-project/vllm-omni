# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-request state for VibeVoice's single-stage AR decode loop.

This module deliberately owns no scheduler or PagedAttention storage. Positive
KV remains in the standard runner and negative KV remains in a bound runner-owned
branch. This state machine owns only request-local conditions, convolution caches,
and waveform chunks.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Protocol

import torch

from .audio_decode import VibeVoiceAudioTokenDecodeOutput
from .runtime_config import (
    VIBEVOICE_MAX_GUIDANCE_SCALE,
    VIBEVOICE_MAX_NUM_DIFFUSION_STEPS,
    VIBEVOICE_MIN_GUIDANCE_SCALE,
    VIBEVOICE_RUNTIME_CONTROL_KEYS,
)


def validate_guidance_scale(value: Any) -> float:
    """Validate one bounded VibeVoice CFG guidance scale."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"VibeVoice guidance_scale must be a real number, got {value!r}.")
    scale = float(value)
    if not math.isfinite(scale):
        raise ValueError(f"VibeVoice guidance_scale must be finite, got {value!r}.")
    if not VIBEVOICE_MIN_GUIDANCE_SCALE <= scale <= VIBEVOICE_MAX_GUIDANCE_SCALE:
        raise ValueError(
            "VibeVoice guidance_scale must be between "
            f"{VIBEVOICE_MIN_GUIDANCE_SCALE} and {VIBEVOICE_MAX_GUIDANCE_SCALE}, got {value!r}."
        )
    return scale


def validate_num_diffusion_steps(value: Any) -> int:
    """Validate one bounded VibeVoice diffusion-step count."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("VibeVoice num_diffusion_steps must be a positive integer.")
    steps = int(value)
    if steps < 1:
        raise ValueError("VibeVoice num_diffusion_steps must be a positive integer.")
    if steps > VIBEVOICE_MAX_NUM_DIFFUSION_STEPS:
        raise ValueError(f"VibeVoice num_diffusion_steps cannot exceed {VIBEVOICE_MAX_NUM_DIFFUSION_STEPS}.")
    return steps


class VibeVoiceInferenceKernel(Protocol):
    """Model-side math used by :class:`VibeVoiceStatefulInference`."""

    def sample_audio_latent(
        self,
        positive_condition: torch.Tensor,
        negative_condition: torch.Tensor,
        noise: torch.Tensor,
        *,
        guidance_scale: float,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor: ...

    def decode_audio_token(
        self,
        audio_latent: torch.Tensor,
        *,
        acoustic_cache: Any = None,
        semantic_cache: Any = None,
    ) -> VibeVoiceAudioTokenDecodeOutput: ...


class VibeVoiceNegativeKVBranch(Protocol):
    """Ownership boundary for the bound negative Qwen branch.

    The implementation must own independent PagedAttention KV, advance only on
    VibeVoice audio-generation inputs, reset to a one-token audio-BOS context at
    every audio segment, and clean up with the parent request. It must not keep
    cache tensors in :class:`VibeVoiceStatefulInference`.
    """

    def reset_audio_segment(self, request_id: str) -> None: ...

    def forward_step(
        self,
        request_ids: list[str],
        input_embeddings: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Advance each negative branch by one embedding and return hidden rows."""
        ...

    def free(self, request_id: str) -> None: ...


@dataclass(slots=True)
class VibeVoiceRequestState:
    """Model-local non-Qwen state for one parent request."""

    request_id: str
    guidance_scale: float
    num_diffusion_steps: int
    acoustic_cache: Any = None
    semantic_cache: Any = None
    positive_condition: torch.Tensor | None = None
    negative_condition: torch.Tensor | None = None
    negative_input_embedding: torch.Tensor | None = None
    next_embedding: torch.Tensor | None = None
    waveform_chunks_cpu: list[torch.Tensor] = field(default_factory=list)
    in_audio_segment: bool = False
    negative_reset_pending: bool = False
    audio_token_count: int = 0
    # Pinned-D2H bookkeeping: id(buffer) -> (event, buffer)
    # for chunks copied off-device without stalling the decode pipeline, and
    # the free pool of reusable pinned buffers. Entries are consumed and the
    # buffers recycled only by drain_waveform_chunks after the copy event has
    # completed. Tests that append plain CPU tensors simply have no entry.
    _waveform_events: dict[int, tuple[Any, torch.Tensor]] = field(default_factory=dict)
    _pinned_pool: list[torch.Tensor] = field(default_factory=list)

    def clear(self) -> None:
        try:
            for event, _ in tuple(self._waveform_events.values()):
                event.synchronize()
        finally:
            self.acoustic_cache = None
            self.semantic_cache = None
            self.positive_condition = None
            self.negative_condition = None
            self.negative_input_embedding = None
            self.next_embedding = None
            self.waveform_chunks_cpu.clear()
            self._waveform_events.clear()
            self._pinned_pool.clear()
            self.in_audio_segment = False
            self.negative_reset_pending = False
            self.audio_token_count = 0


class VibeVoiceStatefulInference:
    """Request-indexed state machine around the frozen diffusion/decode kernels.

    Convolution caches and waveform chunks are parent-request state. Qwen KV is
    intentionally absent: positive KV belongs to ``GPUARModelRunner`` and the
    unresolved negative PagedAttention branch must have a separate owner.
    """

    def __init__(
        self,
        *,
        audio_bos_token_id: int,
        audio_eos_token_id: int,
        audio_token_id: int,
        eos_token_id: int,
        latent_size: int,
        condition_size: int,
        default_guidance_scale: float,
        default_num_diffusion_steps: int,
    ) -> None:
        token_ids = {
            "audio_bos_token_id": audio_bos_token_id,
            "audio_eos_token_id": audio_eos_token_id,
            "audio_token_id": audio_token_id,
            "eos_token_id": eos_token_id,
        }
        if len(set(token_ids.values())) != len(token_ids):
            raise ValueError(f"VibeVoice control token IDs must be distinct, got {token_ids}.")
        if latent_size < 1 or condition_size < 1:
            raise ValueError("VibeVoice latent_size and condition_size must be positive.")
        self.audio_bos_token_id = int(audio_bos_token_id)
        self.audio_eos_token_id = int(audio_eos_token_id)
        self.audio_token_id = int(audio_token_id)
        self.eos_token_id = int(eos_token_id)
        self.latent_size = int(latent_size)
        self.condition_size = int(condition_size)
        self.default_guidance_scale = validate_guidance_scale(default_guidance_scale)
        self.default_num_diffusion_steps = validate_num_diffusion_steps(default_num_diffusion_steps)
        self._states: dict[str, VibeVoiceRequestState] = {}
        self._deferred_cleanup_ids: set[str] = set()
        # Captured decode graphs are bound to convolution-cache addresses.
        # Recycle completed request cache pairs so continuous batching neither
        # destroys nor recaptures CUDA graphs in an unbounded lifecycle loop.
        self._decode_cache_pool: list[tuple[Any, Any]] = []
        self._negative_kv_branch: VibeVoiceNegativeKVBranch | None = None

    def bind_negative_branch(
        self,
        branch: VibeVoiceNegativeKVBranch,
    ) -> None:
        if self._negative_kv_branch is not None:
            raise RuntimeError("VibeVoice negative KV branch was bound twice.")
        self._negative_kv_branch = branch

    @property
    def active_request_ids(self) -> tuple[str, ...]:
        return tuple(self._states)

    @property
    def deferred_cleanup_ids(self) -> frozenset[str]:
        return frozenset(self._deferred_cleanup_ids)

    def get_or_create(
        self,
        request_id: str,
        *,
        reset: bool = False,
    ) -> VibeVoiceRequestState:
        if not request_id:
            raise ValueError("VibeVoice request_id must be non-empty.")
        if reset:
            self.cleanup_request(request_id)
        state = self._states.get(request_id)
        if state is None:
            state = VibeVoiceRequestState(
                request_id=request_id,
                guidance_scale=self.default_guidance_scale,
                num_diffusion_steps=self.default_num_diffusion_steps,
            )
            self._states[request_id] = state
        return state

    def get(self, request_id: str) -> VibeVoiceRequestState | None:
        return self._states.get(request_id)

    def set_runtime_controls(
        self,
        request_id: str,
        extra_args: dict[str, Any] | None,
    ) -> None:
        if not extra_args:
            return
        unknown_keys = sorted(
            (key for key in extra_args if key not in VIBEVOICE_RUNTIME_CONTROL_KEYS),
            key=str,
        )
        if unknown_keys:
            raise ValueError(f"Unsupported VibeVoice runtime controls: {unknown_keys}")

        current = self.get(request_id)
        guidance_scale = current.guidance_scale if current is not None else self.default_guidance_scale
        num_diffusion_steps = current.num_diffusion_steps if current is not None else self.default_num_diffusion_steps
        if "guidance_scale" in extra_args:
            guidance_scale = validate_guidance_scale(extra_args["guidance_scale"])
        if "num_diffusion_steps" in extra_args:
            num_diffusion_steps = validate_num_diffusion_steps(extra_args["num_diffusion_steps"])

        state = self.get_or_create(request_id)
        state.guidance_scale = guidance_scale
        state.num_diffusion_steps = num_diffusion_steps

    def _validate_condition(
        self,
        name: str,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(condition, torch.Tensor):
            raise TypeError(f"VibeVoice {name} must be a tensor.")
        expected_shape = (1, self.condition_size)
        if tuple(condition.shape) != expected_shape:
            raise ValueError(f"VibeVoice {name} must have shape {expected_shape}, got {tuple(condition.shape)}.")
        if not condition.is_floating_point():
            raise TypeError(f"VibeVoice {name} must be floating-point.")
        # Conditions survive across runner steps. Always take request-owned
        # storage: ``contiguous()`` alone aliases an already-contiguous slice
        # of GPUARModelRunner's reusable inputs/hidden-state buffers.
        return condition.detach().clone(memory_format=torch.contiguous_format)

    def record_positive_condition(
        self,
        request_id: str,
        condition: torch.Tensor,
    ) -> None:
        state = self.get_or_create(request_id)
        state.positive_condition = self._validate_condition("positive_condition", condition)

    def record_negative_input_embedding(
        self,
        request_id: str,
        input_embedding: torch.Tensor,
    ) -> None:
        state = self.get_or_create(request_id)
        state.negative_input_embedding = self._validate_condition("negative_input_embedding", input_embedding)

    def record_negative_condition(
        self,
        request_id: str,
        condition: torch.Tensor,
    ) -> None:
        state = self.get_or_create(request_id)
        state.negative_condition = self._validate_condition("negative_condition", condition)
        state.negative_reset_pending = False

    def start_audio_segment(self, request_id: str) -> None:
        state = self.get_or_create(request_id)
        self._start_audio_segment(state)

    def _start_audio_segment(self, state: VibeVoiceRequestState) -> None:
        state.in_audio_segment = True
        state.positive_condition = None
        state.negative_condition = None
        state.negative_input_embedding = None
        state.negative_reset_pending = True
        if state.acoustic_cache is None and state.semantic_cache is None and self._decode_cache_pool:
            state.acoustic_cache, state.semantic_cache = self._decode_cache_pool.pop()
        self._reset_conv_caches(state)
        if self._negative_kv_branch is not None:
            self._negative_kv_branch.reset_audio_segment(state.request_id)

    @staticmethod
    def _reset_conv_caches(state: VibeVoiceRequestState) -> None:
        """Zero the causal Conv1d padding caches at every segment boundary.

        Each audio segment (one speaker turn) must start from a zero
        left-context, matching the official VibeVoiceTokenizerStreamingCache
        ``set_to_zero`` intent that the Transformers PR batch path never
        invoked. Zeroing in place keeps buffer addresses stable so any captured
        decode graph remains valid across segments. A fresh (uninitialized)
        cache is left untouched; its first update lazily allocates zeros.
        """
        for cache in (state.acoustic_cache, state.semantic_cache):
            if cache is None:
                continue
            layers = getattr(cache, "layers", None)
            if not isinstance(layers, dict):
                continue
            for layer in layers.values():
                if getattr(layer, "is_initialized", False) and layer.cache is not None:
                    layer.cache.zero_()

    def _finish_audio_segment(
        self,
        state: VibeVoiceRequestState,
        *,
        release_negative_branch: bool,
    ) -> None:
        state.in_audio_segment = False
        state.positive_condition = None
        state.negative_condition = None
        state.negative_reset_pending = False
        if release_negative_branch:
            state.negative_input_embedding = None
            if self._negative_kv_branch is not None:
                self._negative_kv_branch.free(state.request_id)

    def process_sampled_token(
        self,
        *,
        request_id: str,
        token_id: int,
        token_embedding: torch.Tensor,
        kernel: VibeVoiceInferenceKernel,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply one sampled control-token transition before the next Qwen step."""
        state = self.get_or_create(request_id)
        if token_embedding.ndim != 2 or token_embedding.shape[0] != 1:
            raise ValueError(
                f"VibeVoice token_embedding must have shape (1, hidden_size), got {tuple(token_embedding.shape)}."
            )
        if token_embedding.shape[1] != self.condition_size:
            raise ValueError(
                f"VibeVoice token_embedding hidden size must be {self.condition_size}, got {token_embedding.shape[1]}."
            )

        token_id = int(token_id)
        if token_id == self.audio_bos_token_id:
            self._start_audio_segment(state)
            state.next_embedding = token_embedding
            return token_embedding, None
        if token_id == self.audio_eos_token_id:
            # Match the official generator: audio EOS closes the waveform
            # segment, but the negative Qwen cache is retained until the next
            # audio BOS resets it or request EOS/cleanup releases it. Keep the
            # current embedding as the preceding negative input so even a
            # model-emitted audio token without an intervening BOS cannot
            # kill the whole EngineCore.
            self._finish_audio_segment(
                state,
                release_negative_branch=False,
            )
            state.negative_input_embedding = self._validate_condition(
                "negative_input_embedding",
                token_embedding,
            )
            state.next_embedding = token_embedding
            return token_embedding, None
        if token_id == self.eos_token_id:
            self._finish_audio_segment(
                state,
                release_negative_branch=True,
            )
            state.next_embedding = token_embedding
            return token_embedding, None
        if token_id != self.audio_token_id:
            raise ValueError(
                f"Unsupported VibeVoice control token ID {token_id}; expected one of "
                f"{self.audio_bos_token_id}, {self.audio_eos_token_id}, "
                f"{self.audio_token_id}, {self.eos_token_id}."
            )
        next_embeddings, audio_chunks = self.process_audio_tokens_batch(
            request_ids=[request_id],
            token_embeddings=[token_embedding],
            kernel=kernel,
        )
        return next_embeddings[0], audio_chunks[0]

    def process_audio_tokens_batch(
        self,
        *,
        request_ids: list[str],
        token_embeddings: list[torch.Tensor],
        kernel: VibeVoiceInferenceKernel,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Batch diffusion sampling over the active subset, then decode per-request caches."""
        if not request_ids:
            return [], []
        if len(request_ids) != len(token_embeddings):
            raise ValueError("VibeVoice audio-token request/embedding batch lengths must match.")
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("VibeVoice audio-token active subset contains duplicate request IDs.")

        states: list[VibeVoiceRequestState] = []
        positive_conditions: list[torch.Tensor] = []
        negative_conditions: list[torch.Tensor] = []
        guidance_scale: float | None = None
        num_diffusion_steps: int | None = None
        for request_id, token_embedding in zip(
            request_ids,
            token_embeddings,
            strict=True,
        ):
            state = self.get_or_create(request_id)
            if token_embedding.shape != (1, self.condition_size):
                raise ValueError(
                    "VibeVoice token_embedding must have shape "
                    f"(1, {self.condition_size}), got "
                    f"{tuple(token_embedding.shape)}."
                )
            if not state.in_audio_segment:
                # The official generator applies diffusion whenever an audio
                # token is sampled and resets segment-local state only on an
                # explicit audio BOS. Preserve that behavior for malformed or
                # low-confidence model output instead of escalating one
                # request's audio-EOS -> audio-token transition to an
                # EngineCore-fatal exception.
                state.in_audio_segment = True
            if state.positive_condition is None:
                raise RuntimeError("VibeVoice audio_token has no positive Qwen condition from the preceding AR step.")
            if state.negative_condition is None or state.negative_reset_pending:
                raise RuntimeError(
                    "VibeVoice audio_token requires an independent negative Qwen "
                    "PagedAttention branch. No aligned negative condition is bound; "
                    "unguided fallback is intentionally disabled."
                )
            if guidance_scale is None:
                guidance_scale = state.guidance_scale
                num_diffusion_steps = state.num_diffusion_steps
            elif state.guidance_scale != guidance_scale or state.num_diffusion_steps != num_diffusion_steps:
                raise RuntimeError(
                    "VibeVoice active audio-token requests with different guidance_scale "
                    "or num_diffusion_steps cannot share one diffusion batch."
                )
            positive = state.positive_condition
            negative = state.negative_condition.to(positive)
            if positive_conditions:
                if positive.device != positive_conditions[0].device:
                    raise ValueError("VibeVoice active diffusion conditions must use one device.")
                if positive.dtype != positive_conditions[0].dtype:
                    raise ValueError("VibeVoice active diffusion conditions must use one dtype.")
            states.append(state)
            positive_conditions.append(positive)
            negative_conditions.append(negative)

        if guidance_scale is None or num_diffusion_steps is None:
            raise AssertionError("Non-empty VibeVoice diffusion batch has no runtime controls.")
        positive_batch = torch.cat(positive_conditions, dim=0)
        negative_batch = torch.cat(negative_conditions, dim=0)
        batch_size = len(states)
        # Preserve official active-subset RNG ordering: one [2B, latent] draw,
        # not B independent [2, latent] draws interleaved per request.
        noise = torch.randn(
            (2 * batch_size, self.latent_size),
            device=positive_batch.device,
            dtype=positive_batch.dtype,
        )
        audio_latents = kernel.sample_audio_latent(
            positive_batch,
            negative_batch,
            noise,
            guidance_scale=guidance_scale,
            num_inference_steps=num_diffusion_steps,
        )
        expected_latent_shape = (batch_size, 1, self.latent_size)
        if tuple(audio_latents.shape) != expected_latent_shape:
            raise ValueError(
                "VibeVoice stateful diffusion output must have shape "
                f"{expected_latent_shape}, got {tuple(audio_latents.shape)}."
            )

        next_embeddings: list[torch.Tensor] = []
        audio_chunks: list[torch.Tensor] = []
        for index, state in enumerate(states):
            decoded = kernel.decode_audio_token(
                audio_latents[index : index + 1],
                acoustic_cache=state.acoustic_cache,
                semantic_cache=state.semantic_cache,
            )
            state.acoustic_cache = decoded.acoustic_cache
            state.semantic_cache = decoded.semantic_cache
            # Decode-graph outputs are borrowed static buffers. The next AR
            # step outlives this replay, so retain request-owned storage.
            state.next_embedding = decoded.next_embedding.reshape(1, -1).detach().clone()
            state.waveform_chunks_cpu.append(self._stage_waveform_chunk(state, decoded.audio))
            state.audio_token_count += 1
            # Conditions are one-step values. Keeping either one would allow a
            # desynchronized branch to be reused silently on the next token.
            state.positive_condition = None
            state.negative_condition = None
            next_embeddings.append(state.next_embedding)
            audio_chunks.append(decoded.audio)
        return next_embeddings, audio_chunks

    def _stage_waveform_chunk(
        self,
        state: VibeVoiceRequestState,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """Move one decoded chunk to CPU without stalling the decode pipeline.

        CUDA path: cast to float32 on device, async-copy into a recycled
        pinned buffer, and record an event; ``drain_waveform_chunks`` is the
        only consumer and synchronizes the event before publishing. CPU/test
        path keeps the previous synchronous semantics.
        """
        chunk = audio.detach().reshape(-1)
        if not chunk.is_cuda:
            return chunk.to(device="cpu", dtype=torch.float32).contiguous()
        numel = chunk.numel()
        buffer = None
        for candidate in state._pinned_pool:
            if candidate.numel() == numel:
                buffer = candidate
                break
        if buffer is None:
            buffer = torch.empty(numel, dtype=torch.float32, pin_memory=True)
        else:
            state._pinned_pool.remove(buffer)
        buffer.copy_(chunk.float(), non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        state._waveform_events[id(buffer)] = (event, buffer)
        return buffer

    def drain_waveform_chunks(self, request_id: str) -> torch.Tensor | None:
        """Transfer unpublished CPU waveform chunks to the output channel.

        The state machine owns chunks only until ``make_omni_output`` publishes
        them. Omni's output processor then owns request-level accumulation, so
        keeping another cumulative copy here would waste host memory for long
        generations and could publish a chunk more than once.
        """
        state = self._states.get(request_id)
        if state is None or not state.waveform_chunks_cpu:
            return None
        chunks = state.waveform_chunks_cpu
        state.waveform_chunks_cpu = []
        for chunk in chunks:
            entry = state._waveform_events.pop(id(chunk), None)
            if entry is not None:
                event, buffer = entry
                event.synchronize()
                state._pinned_pool.append(buffer)
        if len(chunks) == 1:
            chunk = chunks[0]
            # A single chunk may be a recycled pinned buffer; publish an
            # owning copy so later tokens cannot overwrite published audio.
            if chunk.is_pinned():
                return chunk.clone()
            return chunk
        return torch.cat(chunks, dim=0).contiguous()

    def on_requests_finished(
        self,
        request_ids: Collection[str],
        *,
        scheduled_req_ids: Collection[str] = (),
    ) -> None:
        # A request still executing this runner step must survive through its
        # final postprocess. Every other finished request can be released now,
        # including the zero-scheduled-token early-return path.
        finished = set(request_ids)
        scheduled = set(scheduled_req_ids)
        for request_id in finished - scheduled:
            self.cleanup_request(request_id)
        self._deferred_cleanup_ids.update(finished & scheduled)

    def flush_deferred_cleanup(
        self,
        *,
        exclude_request_ids: set[str] | frozenset[str] = frozenset(),
    ) -> None:
        cleanup_ids = self._deferred_cleanup_ids - set(exclude_request_ids)
        for request_id in cleanup_ids:
            self.cleanup_request(request_id)
        self._deferred_cleanup_ids.difference_update(cleanup_ids)

    def finish_postprocess(self, request_id: str) -> None:
        if request_id in self._deferred_cleanup_ids:
            self.cleanup_request(request_id)

    def cleanup_request(self, request_id: str) -> None:
        try:
            if self._negative_kv_branch is not None:
                self._negative_kv_branch.free(request_id)
        finally:
            state = self._states.pop(request_id, None)
            try:
                if state is not None:
                    acoustic_cache = state.acoustic_cache
                    semantic_cache = state.semantic_cache
                    reusable = (
                        acoustic_cache is not None
                        and semantic_cache is not None
                        and getattr(acoustic_cache, "_vv_decode_graph", None) is not None
                    )
                    # Synchronize any pending waveform transfer before making
                    # the graph-owned cache addresses available to another
                    # request. ``clear`` then detaches them from the old state.
                    state.clear()
                    if reusable:
                        self._decode_cache_pool.append((acoustic_cache, semantic_cache))
            finally:
                self._deferred_cleanup_ids.discard(request_id)

    def clear(self) -> None:
        for request_id in set(self._states) | self._deferred_cleanup_ids:
            self.cleanup_request(request_id)
        self._deferred_cleanup_ids.clear()
        self._decode_cache_pool.clear()


__all__ = [
    "VibeVoiceInferenceKernel",
    "VibeVoiceNegativeKVBranch",
    "VibeVoiceRequestState",
    "VibeVoiceStatefulInference",
    "validate_guidance_scale",
    "validate_num_diffusion_steps",
]
