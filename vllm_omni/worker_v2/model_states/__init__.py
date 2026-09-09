# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Omni-aware ``init_model_state`` factory.

Extends the upstream v2 factory with Omni architecture dispatch.
Non-Omni architectures fall through to the upstream ``init_model_state``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states import (
    init_model_state as _upstream_init_model_state,
)
from vllm.v1.worker.gpu.model_states.interface import ModelState

# Legacy models without capability declarations remain compatible. New models
# use the existing Omni lifecycle flags; this list need not grow.
_OMNI_ARCHITECTURES: set[str] = {
    "Qwen3OmniMoeForConditionalGeneration",
    "Qwen2_5OmniForConditionalGeneration",
    "MammothModa2ForConditionalGeneration",
    "MiMoAudioForConditionalGeneration",
    "MammothModa2ARForConditionalGeneration",
    "Qwen3TTSTalkerForConditionalGeneration",
    "Qwen3TTSCode2Wav",
}


def init_omni_model_state(
    vllm_config: VllmConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
) -> ModelState:
    """Create the appropriate ``ModelState`` for *model*.

    Returns an ``OmniModelState`` when the configured architecture is a
    known legacy Omni model or declares Omni lifecycle capabilities; otherwise
    delegates to the upstream v2 factory.
    """
    archs = set(vllm_config.model_config.architectures or [])
    uses_omni_lifecycle = any(
        getattr(model, flag, False) is True for flag in ("has_preprocess", "has_postprocess", "have_multimodal_outputs")
    )
    if uses_omni_lifecycle or archs & _OMNI_ARCHITECTURES:
        from vllm_omni.worker_v2.model_states.omni_model_state import (
            OmniModelState,
        )

        return OmniModelState(vllm_config, model, encoder_cache, device)
    return _upstream_init_model_state(vllm_config, model, encoder_cache, device)
