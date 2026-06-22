# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""StepAudioEditx pipeline topology (frozen).

Stage 0: Tokenizer+AR  — text prompt + audio prompt → speech tokens (LLM autoregressive).
Stage 1: Code2Wav — flow-matching decoder → acoustic features → waveform.
  * ``sync_process_input_func`` runs when ``deploy.async_chunk=false``:
    stage 1 builds full-sequence flow input via ``ar2decoder``.
  * ``async_chunk_process_next_stage_input_func`` runs when
    ``deploy.async_chunk=true``: stage 0 streams codec chunks to stage 1
    through the shared-memory connector.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.step_audio_editx"

STEP_AUDIO_EDITX_PIPELINE = PipelineConfig(
    model_type="step_audio_editx",
    hf_architectures=("Step1ForCausalLM",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="step_audio_editx_ar",
            model_arch="StepAudioAR",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.talker2code2wav_async_chunk"),
            custom_process_next_stage_input_func=f"{_PROC}.talker2code2wav_full_payload",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="step_audio_editx_code2wav",
            model_arch="StepAudioCode2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="latent",
            sync_process_input_func=f"{_PROC}.talker2code2wav_token_only",
        ),
    ),
)
