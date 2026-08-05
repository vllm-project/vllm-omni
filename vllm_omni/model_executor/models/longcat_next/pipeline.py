from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.longcat_next"

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
            # "latent", not "audio": OutputModality.AUDIO's CONCAT_LAST/"hidden"
            # remap is wrong for our [1, 8] per-step code rows (need CONCAT_DIM0,
            # no collision with "codes.audio"). See gpu_ar_model_runner.py's
            # _resolve_pooler_payload_req_ids for the matching "latent" support.
            engine_output_type="latent",
            sampling_constraints={"detokenize": True},
            prompt_expand_func=f"{_PROC}.expand_longcat_cfg_prompts",
        ),
    ),
)

# Thinker + one combined decoder stage (LongcatNextMultiDecoder) that holds
# both image and audio decode paths, dispatching per request on whichever of
# visual_token_ids/audio_token_ids talker_mtp populated -- mirrors the
# reference's PostProcessor.decode_multi. A 3-stage thinker->image->audio
# chain doesn't work here: the orchestrator only forwards stage N's output to
# N+1 (ignoring input_sources), so the audio stage would always get the image
# stage's output instead of the thinker's. 2-stage avoids that ambiguity.
#
# final_output_type is statically "audio" for schema purposes only -- the
# real modality is decided by the prompt's trigger token, not this field.
LONGCAT_NEXT_THINKER_MULTI_DECODER_PIPELINE = PipelineConfig(
    model_type="longcat_next_thinker_multi_decoder",
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
            # See LONGCAT_NEXT_THINKER_ONLY_PIPELINE's stage 0 for why this
            # stays "latent", not "audio".
            engine_output_type="latent",
            sampling_constraints={"detokenize": True},
            prompt_expand_func=f"{_PROC}.expand_longcat_cfg_prompts",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="multi_decoder",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="LongcatNextMultiDecoder",
            sync_process_input_func=f"{_PROC}.thinker2multi_decoder_token_only",
        ),
    ),
)
