# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OpenAI diffusion-stage request helpers.

These helpers are stage-based rather than image-format-specific."""

import json
from http import HTTPStatus
from typing import Any, cast

from fastapi import HTTPException

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.stage_params import (
    build_stage_sampling_params_list,
    get_default_sampling_params_list,
)

MAX_UINT32_SEED = 2**32 - 1


async def _generate_with_async_omni(
    engine_client: AsyncOmni | Any,
    gen_params: Any,
    stage_configs: list[Any],
    **kwargs,
):
    engine_client = cast(AsyncOmni, engine_client)
    result = None
    normalized_stage_configs = list(stage_configs)
    if not normalized_stage_configs:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Stage configs not found. Start server with a multi-stage omni model.",
        )
    sampling_params_list = build_stage_sampling_params_list(
        normalized_stage_configs,
        get_default_sampling_params_list(engine_client),
        diffusion_params=gen_params,
        replace_diffusion_params=True,
    )

    async for output in engine_client.generate(
        sampling_params_list=sampling_params_list,
        **kwargs,
    ):
        result = output

    if result is None:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="No output generated from multi-stage pipeline.",
        )
    return result


def apply_stage_default_sampling_params(
    default_params_json: str | None,
    sampling_params: Any,
    stage_key: str,
) -> None:
    """
    Update a stage's sampling parameters with vLLM-Omni defaults.

    Args:
        default_params_json: JSON string of stage-keyed default parameters
        sampling_params: The sampling parameters object to update
        stage_key: The stage ID/key in the pipeline
    """
    if default_params_json is not None:
        default_params_dict = json.loads(default_params_json)
        if stage_key in default_params_dict:
            stage_defaults = default_params_dict[stage_key]
            for param_name, param_value in stage_defaults.items():
                if hasattr(sampling_params, param_name):
                    setattr(sampling_params, param_name, param_value)
