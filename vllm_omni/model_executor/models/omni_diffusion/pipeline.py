# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Omni-Diffusion pipeline topologies.

Omni-Diffusion is one DreamModel checkpoint with several task modes.  The
serving output type is still fixed when a stage starts, so expose separate
variants for image, text, audio, and the S2I diffusion wrapper.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.model_executor.models.omni_diffusion.utils import OMNI_DIFFUSION_END_OF_TEXT_TOKEN_ID

_OMNI_DIFFUSION_MODEL_ARCH = "OmniDiffusionForConditionalGeneration"
_OMNI_DIFFUSION_TEXT_ADAPTER_ARCH = "OmniDiffusionTextAdapterForConditionalGeneration"
_OMNI_DIFFUSION_PROC = "vllm_omni.model_executor.stage_input_processors.omni_diffusion"
_OMNI_DIFFUSION_HF_ARCHS = (
    "DreamModel",
    _OMNI_DIFFUSION_MODEL_ARCH,
)


def _llm_generation_stage(
    *,
    engine_output_type: str,
    final_output_type: str,
) -> StagePipelineConfig:
    return StagePipelineConfig(
        stage_id=0,
        model_stage="omni_diffusion",
        execution_type=StageExecutionType.LLM_GENERATION,
        input_sources=(),
        final_output=True,
        final_output_type=final_output_type,
        owns_tokenizer=True,
        requires_multimodal_data=True,
        model_arch=_OMNI_DIFFUSION_MODEL_ARCH,
        engine_output_type=engine_output_type,
    )


OMNI_DIFFUSION_T2I_PIPELINE = PipelineConfig(
    model_type="omni_diffusion_t2i",
    model_arch=_OMNI_DIFFUSION_MODEL_ARCH,
    hf_architectures=_OMNI_DIFFUSION_HF_ARCHS,
    stages=(
        _llm_generation_stage(
            engine_output_type="latent",
            final_output_type="image",
        ),
    ),
)

OMNI_DIFFUSION_TEXT_PIPELINE = PipelineConfig(
    model_type="omni_diffusion_text",
    model_arch=_OMNI_DIFFUSION_MODEL_ARCH,
    hf_architectures=(),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="omni_diffusion",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(),
            final_output=False,
            final_output_type=None,
            owns_tokenizer=True,
            requires_multimodal_data=True,
            model_arch=_OMNI_DIFFUSION_MODEL_ARCH,
            engine_output_type="latent",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="omni_diffusion_text_adapter",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=False,
            requires_multimodal_data=False,
            model_arch=_OMNI_DIFFUSION_TEXT_ADAPTER_ARCH,
            engine_output_type="text",
            sampling_constraints={
                "stop_token_ids": [OMNI_DIFFUSION_END_OF_TEXT_TOKEN_ID],
            },
            custom_process_input_func=f"{_OMNI_DIFFUSION_PROC}.text_tokens_to_ar_text_adapter",
        ),
    ),
)

OMNI_DIFFUSION_AUDIO_PIPELINE = PipelineConfig(
    model_type="omni_diffusion_audio",
    model_arch=_OMNI_DIFFUSION_MODEL_ARCH,
    hf_architectures=(),
    stages=(
        _llm_generation_stage(
            engine_output_type="audio",
            final_output_type="audio",
        ),
    ),
)

OMNI_DIFFUSION_S2I_PIPELINE = PipelineConfig(
    model_type="omni_diffusion_s2i",
    model_arch="OmniDiffusionS2IPipeline",
    hf_architectures=(),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="omni_diffusion_s2i",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="image",
            owns_tokenizer=False,
            requires_multimodal_data=True,
            model_arch="OmniDiffusionS2IPipeline",
        ),
    ),
)
