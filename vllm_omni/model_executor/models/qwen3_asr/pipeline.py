# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-ASR pipeline topology (frozen).

Single stage: the audio tower + Qwen3 decoder stack from upstream vLLM's
``Qwen3ASRForConditionalGeneration`` turns audio into text.  A standalone entry
is what lets ``vllm serve <Qwen3-ASR checkpoint> --omni`` resolve the model; the
multi-stage ``aura_omni`` / ``joyai_vl_interaction_asr`` pipelines keep using the
same architecture as their first stage.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

QWEN3_ASR_PIPELINE = PipelineConfig(
    model_type="qwen3_asr",
    model_arch="Qwen3ASRForConditionalGeneration",
    hf_architectures=("Qwen3ASRForConditionalGeneration",),
    default_deploy_config_name="qwen3_asr.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="asr",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="text",
            model_arch="Qwen3ASRForConditionalGeneration",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
