# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.model_extras.registry import (
    build_image_to_image_prompt,
    build_image_to_video_prompt,
    build_robot_observations,
    build_text_to_image_prompt,
    finalize_robot_run,
    get_extra_body_params,
    get_extra_output_params,
    get_model_class_name,
    get_worker_extension_class,
    process_robot_actions,
    should_init_extra_args_for_non_diffusion_stages,
)

__all__ = [
    "build_image_to_image_prompt",
    "build_image_to_video_prompt",
    "build_robot_observations",
    "build_text_to_image_prompt",
    "finalize_robot_run",
    "get_extra_body_params",
    "get_extra_output_params",
    "get_model_class_name",
    "get_worker_extension_class",
    "process_robot_actions",
    "should_init_extra_args_for_non_diffusion_stages",
]
