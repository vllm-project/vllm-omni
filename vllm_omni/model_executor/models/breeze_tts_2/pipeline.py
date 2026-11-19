# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import replace

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig
from vllm_omni.serving_capabilities import (
    AUDIO_RESPONSE,
    TEXT_TO_SPEECH,
    declare_capabilities,
)

BREEZE_TTS_2_PIPELINE = PipelineConfig(
    serving_capabilities=declare_capabilities(
        TEXT_TO_SPEECH,
        replace(AUDIO_RESPONSE, support="unknown", reason_code="chat_prompt_contract_undeclared"),
        unknown_reason="chat_prompt_contract_undeclared",
        sources=("vllm_omni/model_executor/models/breeze_tts_2/pipeline.py", "docs/serving/speech_api.md"),
    ),
    model_type="breeze",
    model_arch="BreezeForConditionalGeneration",
    default_deploy_config_name="breeze_tts_2.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="breeze_tts_2",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            scheduler_cls="vllm_omni.core.sched.omni_cfg_ar_scheduler.OmniCFGARScheduler",
            prompt_expand_func="vllm_omni.model_executor.stage_input_processors.breeze_tts_2.expand_cfg_prompts",
            async_chunk_process_next_stage_input_func=(
                "vllm_omni.model_executor.stage_input_processors.breeze_tts_2.talker2code2wav_async_chunk"
            ),
            sampling_constraints={"detokenize": False, "stop_token_ids": [2051], "ignore_eos": True, "min_tokens": 0},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="breeze_tts_2_codec",
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
