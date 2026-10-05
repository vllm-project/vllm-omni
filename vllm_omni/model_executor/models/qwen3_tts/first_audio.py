# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-TTS first-frame decode eligibility."""

from typing import Any

from vllm_omni.model_executor.models.common.talker_first_audio import supports_talker_first_audio


def talker_first_audio_enabled(vllm_config: Any) -> bool:
    """First-frame delivery requires CUDA MRv2 streaming with an in-process Talker.

    Unsupported runners retain the regular codec path. Prefix-cache replay
    and distributed Talkers likewise retain the regular path.
    """
    from .qwen3_tts_code_predictor_vllm import Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM as Predictor

    extra = Predictor._stage_connector_extra_config(vllm_config)
    return Predictor._parse_bool_config(extra.get("talker_first_audio")) and supports_talker_first_audio(vllm_config)
