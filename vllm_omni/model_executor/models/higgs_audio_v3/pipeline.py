# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""higgs-audio v3 pipeline: Talker (text -> 8-codebook codec) -> Code2Wav (codec -> 24 kHz PCM).

Sync-only in this phase (no async_chunk streaming).
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.higgs_audio_v3"

HIGGS_AUDIO_V3_PIPELINE = PipelineConfig(
    model_type="higgs_multimodal_qwen3",
    model_arch="HiggsAudioV3TalkerForConditionalGeneration",
    hf_architectures=("HiggsMultimodalQwen3ForConditionalGeneration",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="higgs_audio_v3",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            # No async_chunk in this phase (sync-only)
            sampling_constraints={
                "detokenize": False,
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
            model_arch="HiggsAudioV3Code2WavForConditionalGeneration",
            sync_process_input_func=f"{_PROC}.talker2code2wav",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
