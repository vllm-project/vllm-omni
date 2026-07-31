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
            # Stays "latent", NOT "audio": OutputModality.AUDIO maps to the
            # CONCAT_LAST accumulation strategy (waveform-shaped, concat along
            # the last dim) and MultimodalPayload.from_raw remaps this stage's
            # routine "hidden" pooler-payload key onto the SAME "audio" name
            # as our own data -- both wrong for our [1, 8] per-step code rows,
            # which need CONCAT_DIM0 and a "hidden" key that doesn't collide
            # with "codes.audio". (Tried "audio" first to satisfy the
            # single-stage-as-final override below; it did unblock the
            # payload but corrupted it into a garbled float tensor under a
            # bare "audio" key -- the "hidden" collision.) The actual fix for
            # that override lives in gpu_ar_model_runner.py's
            # _resolve_pooler_payload_req_ids, widened to also accept
            # "latent", so this field can keep its correct modality tag.
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
            # Stays "latent", NOT "audio": OutputModality.AUDIO maps to the
            # CONCAT_LAST accumulation strategy (waveform-shaped, concat along
            # the last dim) and MultimodalPayload.from_raw remaps this stage's
            # routine "hidden" pooler-payload key onto the SAME "audio" name
            # as our own data -- both wrong for our [1, 8] per-step code rows,
            # which need CONCAT_DIM0 and a "hidden" key that doesn't collide
            # with "codes.audio". (Tried "audio" first to satisfy the
            # single-stage-as-final override below; it did unblock the
            # payload but corrupted it into a garbled float tensor under a
            # bare "audio" key -- the "hidden" collision.) The actual fix for
            # that override lives in gpu_ar_model_runner.py's
            # _resolve_pooler_payload_req_ids, widened to also accept
            # "latent", so this field can keep its correct modality tag.
            engine_output_type="latent",
            sampling_constraints={"detokenize": True},
        ),
    ),
)

# Thinker + audio decoder only, skipping the image decoder (stage_id=1 in
# LONGCAT_NEXT_PIPELINE). Unlike PipelineConfig.validate() (which only
# requires stage_ids to be unique and resolvable via input_sources), the
# runtime orchestrator requires stage_configs[i].stage_id == i — contiguous,
# zero-based — since it indexes stage pools by stage_id directly. So the
# audio decoder here is renumbered to stage_id=1 (not kept at 2, despite
# matching LONGCAT_NEXT_PIPELINE's numbering) to be the second (index 1)
# entry in this 2-stage pipeline.
LONGCAT_NEXT_THINKER_AUDIO_PIPELINE = PipelineConfig(
    model_type="longcat_next_thinker_audio",
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
            # Stays "latent", NOT "audio": OutputModality.AUDIO maps to the
            # CONCAT_LAST accumulation strategy (waveform-shaped, concat along
            # the last dim) and MultimodalPayload.from_raw remaps this stage's
            # routine "hidden" pooler-payload key onto the SAME "audio" name
            # as our own data -- both wrong for our [1, 8] per-step code rows,
            # which need CONCAT_DIM0 and a "hidden" key that doesn't collide
            # with "codes.audio". (Tried "audio" first to satisfy the
            # single-stage-as-final override below; it did unblock the
            # payload but corrupted it into a garbled float tensor under a
            # bare "audio" key -- the "hidden" collision.) The actual fix for
            # that override lives in gpu_ar_model_runner.py's
            # _resolve_pooler_payload_req_ids, widened to also accept
            # "latent", so this field can keep its correct modality tag.
            engine_output_type="latent",
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
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

# Thinker + ONE combined decoder stage that holds both the image and audio
# decode paths internally (LongcatNextMultiDecoder), dispatching per request
# on whichever of visual_token_ids/audio_token_ids talker_mtp populated --
# mirroring the reference's own PostProcessor.decode_multi(gen_image,
# gen_audio), a single conditional dispatch in one process, not a chain of
# services.
#
# This exists instead of LONGCAT_NEXT_PIPELINE's 3-stage
# thinker->image_decoder->audio_decoder chain because the orchestrator's
# _forward_to_next_stage only ever forwards a stage's own output to
# src_stage_id + 1 -- it does NOT consult a stage's declared input_sources
# to fetch data from an earlier stage. So in the 3-stage chain, the audio
# decoder (stage 2) receives the IMAGE decoder's output (stage 1), never the
# thinker's (stage 0) -- audio is unconditionally broken there, not just
# when image-gen happened to run. A 2-stage chain has no such ambiguity:
# stage 1 is always the immediate successor of stage 0, so it always
# receives the thinker's real output regardless of which modality it
# generated.
#
# final_output_type is statically "audio" here for schema purposes even
# though a given response may actually be an image -- this only affects (a)
# the rarely-hit terminal-empty-output shape when a request finishes with
# no downstream input at all, and (b) get_final_stage_id_for_e2e's
# output_modalities matching heuristic (a client hint, not authoritative
# here since which modality a request produces is decided by the prompt's
# trigger token, not by client-requested modalities). Neither affects
# whether the real audio/image is produced or which key it lands under.
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
            # See LONGCAT_NEXT_PIPELINE's stage 0 for why this stays
            # "latent", not "audio".
            engine_output_type="latent",
            sampling_constraints={"detokenize": True},
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
