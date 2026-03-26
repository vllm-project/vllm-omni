"""Omni-aware ``init_model_state`` factory.

Always returns ``OmniModelState``.  Injected into
``GPUModelRunner.load_model`` via ``setattr`` in
``OmniGPUModelRunner.load_model``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.interface import ModelState


def init_omni_model_state(
    vllm_config: VllmConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
) -> ModelState:
    """Create an ``OmniModelState`` for *model*.

    Injected into ``GPUModelRunner.load_model`` via ``setattr`` in
    ``OmniGPUModelRunner.load_model`` so that ``OmniModelState`` is
    used instead of ``DefaultModelState``.  Handles Omni models that
    do not implement ``SupportsMRoPE`` (the MRoPE assert in
    ``DefaultModelState.__init__`` is caught inside
    ``OmniModelState.__init__``).
    """
    from vllm_omni.worker_v2.model_states.omni_model_state import (
        OmniModelState,
    )

    return OmniModelState(vllm_config, model, encoder_cache, device)
