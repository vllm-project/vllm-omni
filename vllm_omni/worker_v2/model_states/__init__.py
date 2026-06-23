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

# Keep in sync: when adding a new Omni model architecture, add it here
# AND update the corresponding test in tests/worker_v2/test_init_model_state.py.
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
    known Omni model; otherwise delegates to the upstream v2 factory.
    """
    archs = set(vllm_config.model_config.architectures or [])
    if archs & _OMNI_ARCHITECTURES:
        from vllm_omni.worker_v2.model_states.omni_model_state import (
            OmniModelState,
        )

        return OmniModelState(vllm_config, model, encoder_cache, device)
    return _upstream_init_model_state(vllm_config, model, encoder_cache, device)
