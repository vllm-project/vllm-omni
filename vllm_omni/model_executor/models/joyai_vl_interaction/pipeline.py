# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JoyAI-VL-Interaction -> Qwen3-TTS native all-sync pipeline."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_JOYAI_INPUT_PROCESSOR = "vllm_omni.model_executor.stage_input_processors.joyai_vl_interaction"
_QWEN3_TTS_INPUT_PROCESSOR = "vllm_omni.model_executor.stage_input_processors.qwen3_tts"


JOYAI_VL_INTERACTION_PIPELINE = PipelineConfig(
    model_type="joyai_vl_interaction",
    model_arch="Qwen3VLForConditionalGeneration",
    default_deploy_config_name="joyai_vl_interaction.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="joyvl",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="text",
            sampling_constraints={
                "detokenize": True,
                "skip_special_tokens": False,
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="qwen3_tts",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            owns_tokenizer=True,
            engine_output_type="latent",
            model_arch="Qwen3TTSTalkerForConditionalGeneration",
            custom_process_input_func=f"{_JOYAI_INPUT_PROCESSOR}.joyai_action_to_tts",
            custom_process_next_stage_input_func=f"{_QWEN3_TTS_INPUT_PROCESSOR}.talker2code2wav_full_payload",
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [2150],
            },
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="Qwen3TTSCode2Wav",
            sync_process_input_func=f"{_QWEN3_TTS_INPUT_PROCESSOR}.talker2code2wav_token_only",
            sampling_constraints={"detokenize": True},
            extras={"tts_args": {"max_instructions_length": 500}},
        ),
    ),
)
