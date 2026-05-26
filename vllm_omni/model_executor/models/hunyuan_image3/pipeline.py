# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HunyuanImage3 pipeline topology."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_HUNYUAN_IMAGE3_HF_ARCHS = (
    "HunyuanImage3ForConditionalGeneration",
    "HunyuanImage3ForCausalMM",
)
_HUNYUAN_IMAGE3_MODEL_ARCH = "HunyuanImage3ForCausalMM"
_HUNYUAN_IMAGE3_INPUT_PROCESSOR = "vllm_omni.model_executor.stage_input_processors.hunyuan_image3"


HUNYUAN_IMAGE3_PIPELINE = PipelineConfig(
    model_type="hunyuan_image_3_moe",
    model_arch=_HUNYUAN_IMAGE3_MODEL_ARCH,
    hf_architectures=_HUNYUAN_IMAGE3_HF_ARCHS,
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="AR",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=False,
            owns_tokenizer=False,
            requires_multimodal_data=True,
            model_arch=_HUNYUAN_IMAGE3_MODEL_ARCH,
            engine_output_type="latent",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=True,
            final_output_type="image",
            requires_multimodal_data=True,
            model_arch=_HUNYUAN_IMAGE3_MODEL_ARCH,
            custom_process_input_func=f"{_HUNYUAN_IMAGE3_INPUT_PROCESSOR}.ar2diffusion",
        ),
    ),
)


HUNYUAN_IMAGE3_AR_PIPELINE = PipelineConfig(
    model_type="hunyuan_image3_ar",
    model_arch=_HUNYUAN_IMAGE3_MODEL_ARCH,
    hf_architectures=(),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="AR",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=False,
            requires_multimodal_data=True,
            model_arch=_HUNYUAN_IMAGE3_MODEL_ARCH,
            engine_output_type="latent",
        ),
    ),
)


HUNYUAN_IMAGE3_DIT_PIPELINE = PipelineConfig(
    model_type="hunyuan_image3_dit",
    model_arch=_HUNYUAN_IMAGE3_MODEL_ARCH,
    hf_architectures=(),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="image",
            requires_multimodal_data=True,
            model_arch=_HUNYUAN_IMAGE3_MODEL_ARCH,
        ),
    ),
)
