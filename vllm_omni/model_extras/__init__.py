# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.model_extras.registry import (
    build_image_to_image_prompt,
    build_image_to_video_prompt,
    build_observations,
    build_text_to_image_prompt,
    get_extra_body_params,
    get_extra_output_params,
    get_model_class_name,
    load_eval_context,
    process_actions,
    run_open_loop,
    should_init_extra_args_for_non_diffusion_stages,
)

__all__ = [
    "build_image_to_image_prompt",
    "build_image_to_video_prompt",
    "build_observations",
    "build_text_to_image_prompt",
    "get_extra_body_params",
    "get_extra_output_params",
    "get_model_class_name",
    "load_eval_context",
    "process_actions",
    "run_open_loop",
    "should_init_extra_args_for_non_diffusion_stages",
]
