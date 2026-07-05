# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AURA-aware Qwen3-ASR wrapper.

When ``additional_information.omni_skip_stages`` includes stage 0 (video-only
turns), the model forces an immediate EOS on decode steps so ASR returns an
empty transcript without a full autoregressive pass. Requests without the flag
delegate to the stock Qwen3-ASR implementation unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.model_executor.models.qwen3_asr import Qwen3ASRForConditionalGeneration
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.stage_input_processors.stage_bypass import (
    should_skip_stage_from_info,
)

_ASR_STAGE_ID = 0


class AuraQwen3ASRForConditionalGeneration(Qwen3ASRForConditionalGeneration):
    """Qwen3-ASR with optional no-op decode for ``omni_skip_stages: [0]``."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._runtime_info: list[dict[str, Any]] | None = None
        self._cached_eos_token_id: int | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        runtime_info = kwargs.get("runtime_additional_information")
        if runtime_info is None:
            runtime_info = kwargs.get("model_intermediate_buffer")
        self._runtime_info = runtime_info if isinstance(runtime_info, list) else None
        return super().forward(
            input_ids,
            positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def _resolve_eos_token_id(self) -> int | None:
        if self._cached_eos_token_id is not None:
            return self._cached_eos_token_id
        eos_id = getattr(self.config, "eos_token_id", None)
        if eos_id is None:
            text_cfg = getattr(self.config, "text_config", None)
            eos_id = getattr(text_cfg, "eos_token_id", None)
        if eos_id is None:
            return None
        if isinstance(eos_id, (list, tuple)):
            eos_id = eos_id[0] if eos_id else None
        if eos_id is None:
            return None
        self._cached_eos_token_id = int(eos_id)
        return self._cached_eos_token_id

    def _noop_row_mask(self, num_rows: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(num_rows, dtype=torch.bool, device=device)
        if not self._runtime_info:
            return mask
        for row_idx in range(min(num_rows, len(self._runtime_info))):
            info = self._runtime_info[row_idx]
            if isinstance(info, dict) and should_skip_stage_from_info(info, _ASR_STAGE_ID):
                mask[row_idx] = True
        return mask

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = super().compute_logits(hidden_states)
        if logits is None or self._runtime_info is None:
            return logits

        num_rows = int(logits.shape[0])
        # Decode steps expose one row per request; prefill has more rows — skip there.
        if num_rows != len(self._runtime_info):
            return logits

        noop_mask = self._noop_row_mask(num_rows, logits.device)
        if not noop_mask.any():
            return logits

        eos_id = self._resolve_eos_token_id()
        if eos_id is None or eos_id >= logits.shape[-1]:
            return logits

        logits[noop_mask] = float("-inf")
        logits[noop_mask, eos_id] = 0.0
        return logits
