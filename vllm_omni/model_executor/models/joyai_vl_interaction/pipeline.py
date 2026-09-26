# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""JoyAI-VL-Interaction -> Qwen3-TTS native all-sync pipelines."""

from dataclasses import replace

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
            requires_full_payload_input=True,
            sampling_constraints={"detokenize": True},
            extras={"tts_args": {"max_instructions_length": 500}},
        ),
    ),
)


_JOYAI_STAGE, *_JOYAI_TTS_STAGES = JOYAI_VL_INTERACTION_PIPELINE.stages

# Opt-in audio-input profile: Qwen3-ASR transcribes the request audio and the
# ASR-to-JoyAI processor combines the transcript with the image/video inputs
# the frontend deferred past the ASR stage. A request without audio bypasses
# stage 0 and enters at the JoyAI stage, so one deployment serves both.
JOYAI_VL_INTERACTION_ASR_PIPELINE = PipelineConfig(
    model_type="joyai_vl_interaction_asr",
    model_arch="Qwen3ASRForConditionalGeneration",
    default_deploy_config_name="joyai_vl_interaction_asr.yaml",
    stages=(
        # Not ``owns_tokenizer``: JoyAI stays the served model, so request
        # sampling params and the frontend model config target it, not ASR.
        StagePipelineConfig(
            stage_id=0,
            model_stage="asr",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            requires_multimodal_data=True,
            engine_output_type="text",
            model_arch="Qwen3ASRForConditionalGeneration",
            sampling_constraints={"detokenize": True},
            bypass_without_modalities=("audio",),
        ),
        replace(
            _JOYAI_STAGE,
            stage_id=1,
            input_sources=(0,),
            model_arch=JOYAI_VL_INTERACTION_PIPELINE.model_arch,
            custom_process_input_func=f"{_JOYAI_INPUT_PROCESSOR}.asr_to_joyai",
        ),
        *(
            replace(stage, stage_id=stage.stage_id + 1, input_sources=tuple(src + 1 for src in stage.input_sources))
            for stage in _JOYAI_TTS_STAGES
        ),
    ),
)
