# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-request state and the per-step audio transition (synchronous scheduling).

One runner step is: ``begin_step`` → ``add_row`` per scheduled request (in
batch order) → positive forward → ``record_hidden`` → ``record_sampled`` →
``finish_step``. The audio transition runs after sampling in the same step,
like the Transformers reference loop, so a token sampled at the length cap
still produces its waveform.

Semantics follow the reference at batch size 1, per request: the negative
branch advances only on this request's ``audio_token`` steps, using that step's
positive input embedding; ``audio_bos`` restarts the negative context; causal
conv caches persist for the whole request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from transformers.models.vibevoice_acoustic_tokenizer.modeling_vibevoice_acoustic_tokenizer import (
        VibeVoiceAcousticTokenizerConv1dPaddingCache as PaddingCache,
    )

    from .negative_kv import VibeVoiceNegativeKV
    from .vibevoice import VibeVoiceModel

SAMPLE_RATE = 24000


@dataclass(slots=True)
class VibeVoiceRequestState:
    progress: int = 0
    next_embedding: torch.Tensor | None = None
    acoustic_cache: PaddingCache | None = None
    semantic_cache: PaddingCache | None = None


@dataclass(slots=True)
class _Row:
    request_id: str
    sample_valid: bool
    # Owned last-position positive input; reused as the negative input.
    input_embedding: torch.Tensor


class VibeVoiceStatefulInference:
    def __init__(
        self,
        *,
        model: VibeVoiceModel,
        negative_kv: VibeVoiceNegativeKV,
        audio_bos_token_id: int,
        audio_token_id: int,
    ) -> None:
        self._model = model
        self._negative_kv = negative_kv
        self._audio_bos_token_id = audio_bos_token_id
        self._audio_token_id = audio_token_id
        self._states: dict[str, VibeVoiceRequestState] = {}
        self.begin_step()

    @property
    def has_rows(self) -> bool:
        return bool(self._rows)

    def begin_step(self) -> None:
        self._rows: list[_Row] = []
        self._hidden: torch.Tensor | None = None
        self._sampled: list[list[int]] | None = None

    def add_row(
        self,
        request_id: str,
        *,
        num_computed: int,
        num_scheduled: int,
        prompt_len: int,
        input_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Register one scheduled row; return the embeddings to feed Qwen."""
        state = self._states.get(request_id)
        if num_computed == 0:
            if state is not None:
                raise RuntimeError(f"VibeVoice request {request_id!r} restarted; preemption is not supported.")
            self._negative_kv.acquire(request_id)
            state = self._states[request_id] = VibeVoiceRequestState()
        elif state is None or num_computed != state.progress:
            raise RuntimeError(f"VibeVoice request {request_id!r} progress is inconsistent; recompute is unsupported.")
        state.progress = num_computed + num_scheduled

        if num_computed >= prompt_len and state.next_embedding is not None:
            if num_scheduled != 1:
                raise RuntimeError("VibeVoice decode rows must schedule exactly one token.")
            input_embeds = state.next_embedding.to(input_embeds.dtype)
            state.next_embedding = None
        self._rows.append(_Row(request_id, num_computed + num_scheduled >= prompt_len, input_embeds[-1:].clone()))
        return input_embeds

    def record_hidden(self, hidden_states: torch.Tensor) -> None:
        if self._hidden is None:
            if hidden_states.shape[0] != len(self._rows):
                raise RuntimeError("VibeVoice logits rows do not match the scheduled rows.")
            self._hidden = hidden_states

    def record_sampled(self, sampled: list[list[int]]) -> None:
        if len(sampled) != len(self._rows):
            raise RuntimeError("VibeVoice sampled rows do not match the scheduled rows.")
        # Copy: the runner clears discarded rows of the returned lists in place.
        self._sampled = [list(tokens) for tokens in sampled]

    def finish_step(self, req_ids: list[str]) -> dict[str, Any]:
        """Apply sampled-token transitions; return this step's sparse audio payload."""
        rows, sampled, hidden = self._rows, self._sampled, self._hidden
        self.begin_step()
        if req_ids != [row.request_id for row in rows] or sampled is None or hidden is None:
            raise RuntimeError("VibeVoice post-sample rows do not match the scheduled rows.")
        audio_indices = []
        for index, (row, tokens) in enumerate(zip(rows, sampled, strict=True)):
            if not row.sample_valid or not tokens:
                continue
            if tokens[0] == self._audio_bos_token_id:
                self._negative_kv.reset_segment(row.request_id)
            elif tokens[0] == self._audio_token_id:
                audio_indices.append(index)

        audio_ids = [rows[index].request_id for index in audio_indices]
        negative_inputs = [rows[index].input_embedding for index in audio_indices]
        chunks = self._run_audio_tokens(audio_ids, negative_inputs, hidden[audio_indices])
        return {
            "model_outputs": chunks,
            "sr": [torch.tensor(SAMPLE_RATE, dtype=torch.int32) for _ in chunks],
            "meta": {"req_id": audio_ids, "sparse_audio": ["1"]},
        }

    def _run_audio_tokens(
        self, request_ids: list[str], negative_inputs: list[torch.Tensor], positive: torch.Tensor
    ) -> list[torch.Tensor]:
        if not request_ids:
            return []
        negative = torch.cat(self._negative_kv.forward_step(request_ids, negative_inputs), dim=0)
        # One fp32 CPU draw per active subset, cast to the conditions, as in
        # the reference; a direct BF16 CUDA draw changes values and RNG state.
        noise = torch.randn((2 * len(request_ids), self._model.config.audio_config.hidden_size)).to(positive)
        latents = self._model.sample_audio_latent(positive, negative, noise)
        states = [self._states[request_id] for request_id in request_ids]
        decoded = self._model.decode_audio_tokens(
            latents, [state.acoustic_cache for state in states], [state.semantic_cache for state in states]
        )
        for index, state in enumerate(states):
            state.acoustic_cache = decoded.acoustic_caches[index]
            state.semantic_cache = decoded.semantic_caches[index]
            state.next_embedding = decoded.next_embeddings[index].reshape(1, -1)
        # One D2H copy per subset; rows are contiguous CPU views.
        return list(decoded.audio.reshape(len(request_ids), -1).to("cpu", torch.float32).unbind(0))

    def release(self, request_ids: set[str] | list[str]) -> None:
        for request_id in request_ids:
            self._states.pop(request_id, None)
            self._negative_kv.release(request_id)


__all__ = ["VibeVoiceRequestState", "VibeVoiceStatefulInference"]
