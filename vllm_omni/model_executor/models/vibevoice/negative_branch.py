# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VibeVoice executor for the runner-owned negative causal KV branch."""

from __future__ import annotations

from typing import Any

import torch
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.sequence import IntermediateTensors

from vllm_omni.worker.named_kv_branch import NamedCausalKVBranch


class VibeVoiceNegativeBranch:
    """Advance official negative-Qwen state without owning its Paged KV."""

    def __init__(
        self,
        *,
        store: NamedCausalKVBranch,
        language_model: Qwen2Model,
        hidden_size: int,
    ) -> None:
        if store.name != "negative":
            raise ValueError(f"VibeVoice requires a named KV branch called 'negative', got {store.name!r}.")
        if hidden_size < 1:
            raise ValueError("VibeVoice negative hidden_size must be positive.")
        self.store = store
        self.language_model = language_model
        self.hidden_size = int(hidden_size)

    def reset_audio_segment(self, request_id: str) -> None:
        self.store.reset(request_id)

    def forward_step(
        self,
        request_ids: list[str],
        input_embeddings: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if not request_ids or len(request_ids) != len(input_embeddings):
            raise ValueError("VibeVoice negative KV request and embedding batches must be non-empty and aligned.")
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("VibeVoice negative KV batch contains duplicate request IDs.")

        for embedding in input_embeddings:
            if not isinstance(embedding, torch.Tensor):
                raise TypeError("VibeVoice negative input embedding must be a tensor.")
            if tuple(embedding.shape) != (1, self.hidden_size):
                raise ValueError(
                    "VibeVoice negative input embedding must have shape "
                    f"(1, {self.hidden_size}), got {tuple(embedding.shape)}."
                )
            if not embedding.is_floating_point():
                raise TypeError("VibeVoice negative input embedding must be floating-point.")

        try:
            # Advance the whole logical batch in ONE varlen decode
            # forward. The store owns one independent block table per request;
            # a single batched attention context never exposes two negative
            # kv_cache bindings at once, identical to the sequential path.
            with self.store.append_and_enter_batch(request_ids) as step:
                stacked_inputs = torch.cat(input_embeddings, dim=0)
                hidden_states: Any = self.language_model(
                    input_ids=None,
                    positions=step.position,
                    inputs_embeds=stacked_inputs,
                )
                if isinstance(hidden_states, IntermediateTensors):
                    raise RuntimeError(
                        "VibeVoice negative Qwen returned pipeline intermediate tensors; PP=1 is required."
                    )
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
                if not isinstance(hidden_states, torch.Tensor):
                    raise TypeError("VibeVoice negative Qwen must return hidden-state tensor output.")
                expected_shape = (len(request_ids), self.hidden_size)
                if tuple(hidden_states.shape) != expected_shape:
                    raise ValueError(
                        "VibeVoice negative Qwen hidden state must have shape "
                        f"{expected_shape}, got {tuple(hidden_states.shape)}."
                    )
                # The shared Qwen may reuse output storage on a later forward.
                # Own the batch until the caller binds every row to
                # request-local state (which clones again per request).
                owned = hidden_states.detach().clone(
                    memory_format=torch.contiguous_format,
                )
                conditions = [row.reshape(1, self.hidden_size) for row in owned.unbind(0)]
        except Exception:
            # A model-forward exception is fatal to the current engine. Drop
            # every request touched by this logical batch so no partially
            # advanced negative branch survives into shutdown diagnostics.
            for request_id in request_ids:
                self.free(request_id)
            raise
        return conditions

    def free(self, request_id: str) -> None:
        self.store.free(request_id)


__all__ = ["VibeVoiceNegativeBranch"]
