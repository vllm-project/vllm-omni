# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiMo Audio pipeline topology (frozen).

Stage 0: fused thinker+talker — multimodal understanding + text + RVQ codes.
Stage 1: Code2Wav — RVQ codes → waveform.

MiMoAudioConfig inherits from Qwen2Config, so the HF ``model_type`` field
reports ``qwen2`` — the registry's model_type-based auto-detect can't
disambiguate. ``hf_architectures`` lets ``StageConfigFactory`` fall back to
matching ``hf_config.architectures`` instead.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.mimo_audio"

MIMO_AUDIO_PIPELINE = PipelineConfig(
    model_type="mimo_audio",
    # HF ``architectures: ["MiMoAudioModel"]`` is also the registry key in
    # ``model_executor/models/registry.py``; it resolves to the internal
    # class ``MiMoAudioForConditionalGeneration`` in ``mimo_audio.py``.
    model_arch="MiMoAudioModel",
    hf_architectures=("MiMoAudioModel", "MiMoV2ASRForCausalLM"),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="fused_thinker_talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.llm2code2wav_async_chunk"),
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            sync_process_input_func=f"{_PROC}.llm2code2wav",
            sampling_constraints={"detokenize": False},
        ),
    ),
)
