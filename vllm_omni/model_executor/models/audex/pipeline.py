# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audex (Nemotron-Labs-Audex-2B) TTS pipeline topology.

Stage 0: Thinker  — ChatML text prompt → <speechcodec_N> tokens (LLM AR).
Stage 1: Code2Wav — streaming causal speech decoder → 16 kHz waveform.

Users pass the HF repo ROOT (its manifest config.json carries
``model_type: nemotron_labs_audex``); per-stage checkpoints resolve through
``model_subdir``/``tokenizer_subdir``. The decoder subfolder has no tokenizer
files, so stage 1's tokenizer also points at the thinker checkpoint.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.audex"

# <speechgen_end> token id in checkpoint_folder_audiogen's tokenizer; pinned
# by a unit test against the real tokenizer fixture.
AUDEX_SPEECHGEN_END_TOKEN_ID = 131076

# <audiogen_end> stops TTA generation (pinned in tests via models/audex/tta.py).
AUDEX_AUDIOGEN_END_TOKEN_ID = 131074

# No pipeline-level model_arch: stage-0 classes resolve from each
# checkpoint's own ``architectures`` (dense on the 2B, NemotronH on the
# 30B-A3B); forcing one name via hf_overrides would misbind the other
# size. Decoder stages keep their explicit per-stage model_arch.
AUDEX_TTS_PIPELINE = PipelineConfig(
    model_type="audex_tts",
    default_deploy_config_name="audex_tts.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="audex_thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            model_subdir="checkpoint_folder_audiogen",
            tokenizer_subdir="checkpoint_folder_audiogen",
            # Emits the unconditional CFG companion for cfg_scale > 1.0
            # requests; a no-op for unguided requests.
            prompt_expand_func=f"{_PROC}.expand_cfg_prompts",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.thinker2code2wav_async_chunk"),
            custom_process_next_stage_input_func=f"{_PROC}.thinker2code2wav_full_payload",
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [AUDEX_SPEECHGEN_END_TOKEN_ID],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="audex_code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="AudexCode2Wav",
            model_subdir="audex_causal_speech_decoder",
            tokenizer_subdir="checkpoint_folder_audiogen",
            sync_process_input_func=f"{_PROC}.thinker2code2wav_token_only",
            sampling_constraints={"detokenize": True},
        ),
    ),
)


# Text-to-audio (caption → general audio): same audiogen thinker checkpoint,
# but the output rides the interleaved 4-codebook <audiocodec_N> RVQ block
# and decodes through the EXTERNAL XCodec1 checkpoint. XCodec1 is a CNN codec
# decoded over the full sequence, so this pipeline is sync full-payload only
# (deploy yaml sets async_chunk: false). CFG is effectively mandatory for TTA
# quality (official default scale 3.0).
# Cascaded speech-to-speech: the audio-capable full checkpoint serves all
# three official passes (ASR → chat → TTS) in one deployment; only the TTS
# pass (output modality "audio", final_stage_id 1) traverses the streaming
# speech decoder — text-modality passes finish at stage 0 and never reach
# the codec path. The cascade itself is orchestrated by the client/example.
AUDEX_S2S_PIPELINE = PipelineConfig(
    model_type="audex_s2s",
    default_deploy_config_name="audex_s2s.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="audex_omni",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            model_subdir="checkpoint_folder_full",
            tokenizer_subdir="checkpoint_folder_full",
            prompt_expand_func=f"{_PROC}.expand_cfg_prompts",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.thinker2code2wav_async_chunk"),
            custom_process_next_stage_input_func=f"{_PROC}.thinker2code2wav_full_payload",
            # <speechgen_end> additionally stops the TTS pass; text passes
            # stop at their natural eos first. detokenize stays on for the
            # text-final passes.
            sampling_constraints={
                "detokenize": True,
                "stop_token_ids": [AUDEX_SPEECHGEN_END_TOKEN_ID],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="audex_code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="AudexCode2Wav",
            model_subdir="audex_causal_speech_decoder",
            # The decoder folder has no tokenizer files. Borrow the audiogen
            # tokenizer (not checkpoint_folder_full): this stage resolves its
            # snapshot with the "tts" download profile, which guarantees
            # checkpoint_folder_audiogen but not the full folder's tokenizer,
            # so per-stage / clean-cache launches would otherwise fail.
            tokenizer_subdir="checkpoint_folder_audiogen",
            sync_process_input_func=f"{_PROC}.thinker2code2wav_token_only",
            sampling_constraints={"detokenize": True},
        ),
    ),
)


# Audio understanding (ASR / audio QA): single-stage deployment of the
# audio-capable full checkpoint — speech (+ text instruction) in, text out.
# Modeled on the ming_flash_omni / qwen2_5_omni thinker-only pipelines.
AUDEX_THINKER_ONLY_PIPELINE = PipelineConfig(
    model_type="audex_thinker_only",
    default_deploy_config_name="audex_thinker_only.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="audex_omni",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="text",
            model_subdir="checkpoint_folder_full",
            tokenizer_subdir="checkpoint_folder_full",
            sampling_constraints={"detokenize": True},
        ),
    ),
)


AUDEX_TTA_PIPELINE = PipelineConfig(
    model_type="audex_tta",
    default_deploy_config_name="audex_tta.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="audex_tta_thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            model_subdir="checkpoint_folder_audiogen",
            tokenizer_subdir="checkpoint_folder_audiogen",
            prompt_expand_func=f"{_PROC}.expand_cfg_prompts",
            custom_process_next_stage_input_func=f"{_PROC}.thinker2xcodec_full_payload",
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [AUDEX_AUDIOGEN_END_TOKEN_ID],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="audex_xcodec",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="AudexXCodec1",
            # Length placeholder for the stage-1 request; the bulk codec
            # payload ships via thinker2xcodec_full_payload.
            sync_process_input_func=f"{_PROC}.thinker2code2wav_token_only",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
