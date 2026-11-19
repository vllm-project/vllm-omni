# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in Qwen-Image 2.1 diffusion stage topologies."""

from vllm_omni.config.stage_config import (
    DiffusionStageRole,
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_MODEL_ARCH = "QwenImage21Pipeline"
_DIFFUSION_HANDOFF = "vllm_omni.model_executor.stage_input_processors.diffusion_disagg.diffusion_stage_handoff"
_ENCODE_PAYLOAD_KEYS = ("qwen_image_21_conditioning",)
_DENOISE_PAYLOAD_KEYS = ("latents",)


QWEN_IMAGE_21_EG_PIPELINE = PipelineConfig(
    model_type="qwen_image_21_eg",
    model_arch=_MODEL_ARCH,
    default_deploy_config_name="qwen_image_21_eg.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="encode",
            execution_type=StageExecutionType.DIFFUSION,
            stage_role=DiffusionStageRole.ENCODE,
            stage_output_payload_keys=_ENCODE_PAYLOAD_KEYS,
            input_sources=(),
            final_output=False,
            model_arch=_MODEL_ARCH,
            engine_output_type="custom",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="denoise_decode",
            execution_type=StageExecutionType.DIFFUSION,
            stage_role=DiffusionStageRole.DENOISE_DECODE,
            stage_input_payload_keys=_ENCODE_PAYLOAD_KEYS,
            input_sources=(0,),
            final_output=True,
            final_output_type="image",
            model_arch=_MODEL_ARCH,
            custom_process_input_func=_DIFFUSION_HANDOFF,
        ),
    ),
)


QWEN_IMAGE_21_EGD_PIPELINE = PipelineConfig(
    model_type="qwen_image_21_egd",
    model_arch=_MODEL_ARCH,
    default_deploy_config_name="qwen_image_21_egd.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="encode",
            execution_type=StageExecutionType.DIFFUSION,
            stage_role=DiffusionStageRole.ENCODE,
            stage_output_payload_keys=_ENCODE_PAYLOAD_KEYS,
            input_sources=(),
            final_output=False,
            model_arch=_MODEL_ARCH,
            engine_output_type="custom",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="denoise",
            execution_type=StageExecutionType.DIFFUSION,
            stage_role=DiffusionStageRole.DENOISE,
            stage_input_payload_keys=_ENCODE_PAYLOAD_KEYS,
            stage_output_payload_keys=_DENOISE_PAYLOAD_KEYS,
            input_sources=(0,),
            final_output=False,
            model_arch=_MODEL_ARCH,
            engine_output_type="custom",
            custom_process_input_func=_DIFFUSION_HANDOFF,
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="decode",
            execution_type=StageExecutionType.DIFFUSION,
            stage_role=DiffusionStageRole.DECODE,
            stage_input_payload_keys=_DENOISE_PAYLOAD_KEYS,
            input_sources=(1,),
            final_output=True,
            final_output_type="image",
            model_arch=_MODEL_ARCH,
            custom_process_input_func=_DIFFUSION_HANDOFF,
        ),
    ),
)
