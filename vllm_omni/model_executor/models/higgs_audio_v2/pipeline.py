# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""higgs-audio v2 pipeline: Talker (text -> 8-codebook codec) -> Code2Wav (codec -> 24 kHz PCM)."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.higgs_audio_v2"

HIGGS_AUDIO_V2_PIPELINE = PipelineConfig(
    model_type="higgs_audio_v2",
    default_deploy_config_name="higgs_audio_v2.yaml",
    model_arch="HiggsAudioV2ForConditionalGeneration",
    hf_architectures=("HiggsAudioV2ForConditionalGeneration",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="higgs_audio_v2",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.talker2code2wav_async_chunk"),
            sampling_constraints={
                "detokenize": False,
                # Two stop signals for Stage-0:
                #   * 128009 = standard LM eos_token_id (sequence-level stop)
                #   * 128012 = audio_eos_token_id (forced by the talker's
                #     _apply_audio_mode_bias at audio-ramp completion; matches
                #     upstream HiggsAudioModel._sample_audio_tokens override).
                "stop_token_ids": [128009, 128012],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="HiggsAudioV2Code2WavForConditionalGeneration",
            sync_process_input_func=f"{_PROC}.talker2code2wav",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
