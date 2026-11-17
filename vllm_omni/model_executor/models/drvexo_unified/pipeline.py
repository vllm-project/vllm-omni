# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Share transport and speech stages, retain the distinct hybrid Thinker."""

from dataclasses import replace

from vllm_omni.model_executor.models.drvexo_speech_8b.pipeline import (
    DRVEXO_SPEECH_8B_PIPELINE,
    DRVEXO_SPEECH_8B_THINKER_ONLY_PIPELINE,
)

DRVEXO_UNIFIED_PIPELINE = replace(
    DRVEXO_SPEECH_8B_PIPELINE,
    model_type="drvexo_speech_qwen35_4b",
    model_arch="DRVEXOSpeechQwen35ForConditionalGeneration",
    default_deploy_config_name="drvexo_unified.yaml",
    stages=tuple(
        replace(stage, model_arch="DRVEXOUnifiedTalker") if stage.stage_id == 1 else stage
        for stage in DRVEXO_SPEECH_8B_PIPELINE.stages
    ),
)

DRVEXO_UNIFIED_THINKER_ONLY_PIPELINE = replace(
    DRVEXO_SPEECH_8B_THINKER_ONLY_PIPELINE,
    model_type="drvexo_speech_qwen35_4b_thinker_only",
    model_arch="DRVEXOSpeechQwen35ForConditionalGeneration",
    default_deploy_config_name="drvexo_unified_thinker_only.yaml",
)
