# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Covo-Audio pipeline topology (frozen).

Stage 0: fused_thinker_talker — multimodal understanding + interleaved text/audio
         token generation (AR).
Stage 1: code2wav              — audio codes → waveform (generation).
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.covo_audio"

COVO_AUDIO_PIPELINE = PipelineConfig(
    model_type="covo_audio",
    default_deploy_config_name="covo_audio.yaml",
    model_arch="CovoAudioForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="fused_thinker_talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            custom_process_next_stage_input_func=f"{_PROC}.llm2code2wav_full_payload",
            sampling_constraints={
                "detokenize": True,
                "stop_token_ids": [151645],
                "ignore_eos": True,
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
            sync_process_input_func=f"{_PROC}.llm2code2wav_token_only",
            sampling_constraints={"detokenize": False},
            requires_full_payload_input=True,
        ),
    ),
)

# ASR / audio-understanding only: fused_thinker_talker (stage 0) without the
# code2wav waveform stage. Text-only decoding. This is an evaluation topology for
# the encoder-bucketing benchmarks (single sequence, greedy), so it is NOT
# registered in the production pipeline_registry and carries no default deploy
# YAML under vllm_omni/deploy/. A local WER measurement script registers it via
# ``register_pipeline`` and passes an explicit evaluation deploy config.
COVO_AUDIO_THINKER_ONLY_PIPELINE = PipelineConfig(
    model_type="covo_audio_thinker_only",
    model_arch="CovoAudioForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="fused_thinker_talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="text",
            sampling_constraints={
                "detokenize": True,
                "stop_token_ids": [151645],
                "ignore_eos": True,
            },
        ),
    ),
)
