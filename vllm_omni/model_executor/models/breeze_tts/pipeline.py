# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

BREEZE_TTS_PIPELINE = PipelineConfig(
    model_type="breeze",
    model_arch="BreezeForConditionalGeneration",
    default_deploy_config_name="breeze_tts.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="breeze_tts",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            scheduler_cls="vllm_omni.core.sched.omni_cfg_ar_scheduler.OmniCFGARScheduler",
            prompt_expand_func="vllm_omni.model_executor.stage_input_processors.breeze_tts.expand_cfg_prompts",
            async_chunk_process_next_stage_input_func=(
                "vllm_omni.model_executor.stage_input_processors.breeze_tts.talker2code2wav_async_chunk"
            ),
            sampling_constraints={"detokenize": False, "stop_token_ids": [2051]},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="breeze_code2wav",
            model_arch="BreezeCode2Wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            retains_state_across_chunks=True,
            sampling_constraints={"detokenize": True},
        ),
    ),
)
