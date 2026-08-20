# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.model_extras.registry import (
    build_image_to_image_prompt,
    build_image_to_video_prompt,
    build_text_to_image_prompt,
    build_x_to_text_prompt,
    get_extra_body_params,
    get_extra_output_params,
    get_model_class_name,
    get_output_tensor_range,
    get_transformer_config_subfolder,
    get_x_to_text_model_family,
    should_init_extra_args_for_non_diffusion_stages,
    should_preserve_reference_image_size,
)

__all__ = [
    "build_image_to_image_prompt",
    "build_image_to_video_prompt",
    "build_text_to_image_prompt",
    "build_x_to_text_prompt",
    "get_extra_body_params",
    "get_extra_output_params",
    "get_model_class_name",
    "get_output_tensor_range",
    "get_transformer_config_subfolder",
    "get_x_to_text_model_family",
    "should_init_extra_args_for_non_diffusion_stages",
    "should_preserve_reference_image_size",
]
