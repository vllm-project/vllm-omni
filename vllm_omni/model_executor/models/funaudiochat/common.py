# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared registration helpers for the native FunAudioChat model."""

from __future__ import annotations

from typing import Any

from vllm.multimodal import MULTIMODAL_REGISTRY

try:
    from vllm.model_executor.models.funaudiochat import (
        FunAudioChatDummyInputsBuilder,
        FunAudioChatMultiModalProcessor,
        FunAudioChatProcessingInfo,
    )
except ImportError:
    FunAudioChatDummyInputsBuilder = None
    FunAudioChatMultiModalProcessor = None
    FunAudioChatProcessingInfo = None


def register_funaudiochat_processor(model_cls: type[Any]) -> type[Any]:
    """Register vLLM's audio-input processor for the Omni model wrapper."""
    if (
        FunAudioChatMultiModalProcessor is None
        or FunAudioChatProcessingInfo is None
        or FunAudioChatDummyInputsBuilder is None
    ):
        return model_cls

    return MULTIMODAL_REGISTRY.register_processor(
        FunAudioChatMultiModalProcessor,
        info=FunAudioChatProcessingInfo,
        dummy_inputs=FunAudioChatDummyInputsBuilder,
    )(model_cls)


__all__ = ["register_funaudiochat_processor"]
