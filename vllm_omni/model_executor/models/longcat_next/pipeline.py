from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.longcat_next"

LONGCAT_NEXT_PIPELINE = PipelineConfig(
    model_type="longcat_next",
    model_arch="LongcatNextForCausalLM",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="image_decoder",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="image",
            engine_output_type="image",
            model_arch="LongcatNextImageDecoder",
            sync_process_input_func=f"{_PROC}.thinker2image_decoder_token_only",
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="audio_decoder",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="LongcatNextAudioDecoder",
            sync_process_input_func=f"{_PROC}.thinker2audio_decoder_token_only",
        ),
    ),
)

LONGCAT_NEXT_THINKER_ONLY_PIPELINE = PipelineConfig(
    model_type="longcat_next_thinker_only",
    model_arch="LongcatNextForCausalLM",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
