# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

PHONON_PIPELINE = PipelineConfig(
    model_type="gradium_phonon_tts",
    default_deploy_config_name="phonon.yaml",
    model_arch="PhononTalkerForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="phonon",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            owns_tokenizer=True,
            engine_output_type="audio",
            recompute_preemption="fail",
            sampling_constraints={"detokenize": False, "stop_token_ids": [1]},
        ),
    ),
)
