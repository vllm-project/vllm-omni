# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Omni-aware ``init_model_state`` factory.

Extends the upstream v2 factory with Omni architecture dispatch.
Non-Omni architectures use upstream dispatch; its default state acquires
Omni lifecycle support when owned by an Omni stage.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states import (
    init_model_state as _upstream_init_model_state,
)
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelState

from vllm_omni.config.model import OmniModelConfig


def init_omni_model_state(
    vllm_config: VllmConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
) -> ModelState:
    """Create the appropriate ``ModelState`` for *model*.

    Returns an ``OmniModelState`` when the model declares Omni lifecycle
    capabilities, and an ``OmniMambaHybridModelState`` for hybrid models of an
    Omni stage that do not supply their own state. Otherwise delegates to the
    upstream v2 factory, wrapping its default state for auxiliary Omni stages.
    """
    factory = getattr(model, "create_omni_model_state", None)
    if factory is not None:
        return factory(vllm_config, encoder_cache, device)
    model_config = getattr(vllm_config, "model_config", None)
    # get_model_state_cls and create_mrv2_model_state already name the state.
    # Drama 3's factory returns NPUFishOmniModelState on NPU; replacing it with
    # the generic hybrid state drops Ascend attention metadata and the NPU
    # runner refuses to start.
    owns_state = hasattr(model, "get_model_state_cls") or callable(
        getattr(type(model), "create_mrv2_model_state", None)
    )
    if (
        isinstance(model_config, OmniModelConfig)
        and getattr(model_config, "is_hybrid", False) is True
        and not owns_state
    ):
        # Upstream picks MambaHybridModelState for hybrid recurrent layers
        # (e.g. KDA); the Omni runner needs OmniModelState's lifecycle too.
        from vllm_omni.worker_v2.model_states.omni_mamba_hybrid import OmniMambaHybridModelState

        return OmniMambaHybridModelState(vllm_config, model, encoder_cache, device)
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
    state = _upstream_init_model_state(vllm_config, model, encoder_cache, device)
    if type(state) is DefaultModelState and isinstance(getattr(vllm_config, "model_config", None), OmniModelConfig):
        # Auxiliary stages such as the forced aligner have no model-owned
        # hooks, but Omni's runner still needs its request/output lifecycle.
        # Keep specialized upstream states and plain vLLM configs intact.
        from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

        return OmniModelState(vllm_config, model, encoder_cache, device)
    return state
