"""Synchronous Breeze-TTS-2 pipeline: talker -> Mimi codec."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.breeze_tts_2"

BREEZE_TTS_2_PIPELINE = PipelineConfig(
    model_type="breeze",
    default_deploy_config_name="breeze_tts_2.yaml",
    model_arch="BreezeForConditionalGeneration",
    hf_architectures=("BreezeForConditionalGeneration",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="breeze_tts_2",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            custom_process_next_stage_input_func=f"{_PROC}.talker2codec_full_payload",
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [2051],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="breeze_tts_2_codec",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="BreezeTTS2MimiCodec",
            sync_process_input_func=f"{_PROC}.talker2codec",
            sampling_constraints={"detokenize": True},
            requires_full_payload_input=True,
        ),
    ),
)

__all__ = ["BREEZE_TTS_2_PIPELINE"]
