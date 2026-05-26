# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm-omni integration for boson-ai's higgs-audio v2 (two-stage TTS)."""

from vllm_omni.model_executor.models.higgs_audio_v2.configuration_higgs_audio_v2 import (
    HiggsAudioV2Config,
)

__all__ = ["HiggsAudioV2Config"]
