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


def init_omni_model_state(
    vllm_config: VllmConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
) -> ModelState:
    """Create the appropriate ``ModelState`` for *model*.

    Returns an ``OmniModelState`` when the model declares Omni lifecycle
    capabilities; otherwise
    delegates to the upstream v2 factory.
    """
    factory = getattr(model, "create_omni_model_state", None)
    if factory is not None:
        return factory(vllm_config, encoder_cache, device)
    uses_omni_lifecycle = any(
        getattr(model, flag, False) is True for flag in ("has_preprocess", "has_postprocess", "have_multimodal_outputs")
    )
    if uses_omni_lifecycle:
        factory = getattr(type(model), "create_mrv2_model_state", None)
        if callable(factory):
            state = factory(model, vllm_config, encoder_cache, device)
            if not isinstance(state, ModelState):
                raise TypeError("create_mrv2_model_state must return a ModelState")
            return state
        from vllm_omni.worker_v2.model_states.omni_model_state import (
            OmniModelState,
        )

        return OmniModelState(vllm_config, model, encoder_cache, device)
    return _upstream_init_model_state(vllm_config, model, encoder_cache, device)
