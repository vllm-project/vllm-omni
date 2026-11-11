# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LFM3-Audio reuses the LFM3-Omni stages; only the checkpoint's encoders differ."""

from dataclasses import replace

from vllm_omni.model_executor.models.lfm3_omni.pipeline import LFM3_OMNI_PIPELINE, LFM3_OMNI_TEXT_PIPELINE

LFM3_AUDIO_PIPELINE = replace(
    LFM3_OMNI_PIPELINE,
    model_type="lfm3_audio",
    model_arch="Lfm3AudioForConditionalGeneration",
    default_deploy_config_name="lfm3_audio.yaml",
)

LFM3_AUDIO_TEXT_PIPELINE = replace(
    LFM3_OMNI_TEXT_PIPELINE,
    model_type="lfm3_audio_text",
    model_arch="Lfm3AudioForConditionalGeneration",
    default_deploy_config_name="lfm3_audio_text.yaml",
)
