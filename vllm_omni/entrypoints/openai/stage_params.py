# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
from typing import Any

from vllm import SamplingParams
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.utils import get_stage_type
from vllm_omni.inputs.data import OmniSamplingParams

logger = init_logger(__name__)


def clone_sampling_params(params: OmniSamplingParams) -> OmniSamplingParams:
    """Clone request sampling params without sharing mutable request state."""
    if hasattr(params, "clone"):
        try:
            return params.clone()
        except Exception as exc:
            logger.warning("Failed to clone sampling params with clone(): %s", exc)

    try:
        return copy.deepcopy(params)
    except Exception as exc:
        logger.warning("Failed to deepcopy sampling params; reusing original object: %s", exc)
        return params


def get_default_sampling_params_list(engine_client: Any) -> list[OmniSamplingParams]:
    """Return a mutable copy of an engine client's default sampling params."""
    default_params = getattr(engine_client, "default_sampling_params_list", None)
    if isinstance(default_params, list):
        return list(default_params)
    return []


def resolve_stage_sampling_params(
    stage_cfg: Any,
    stage_index: int,
    default_sampling_params_list: list[OmniSamplingParams],
    *,
    diffusion_params: OmniSamplingParams | None = None,
) -> OmniSamplingParams:
    """Resolve one stage's effective sampling params from stage defaults."""
    if stage_index < len(default_sampling_params_list):
        return clone_sampling_params(default_sampling_params_list[stage_index])

    if get_stage_type(stage_cfg) == "diffusion" and diffusion_params is not None:
        return clone_sampling_params(diffusion_params)

    return SamplingParams()


def build_stage_sampling_params_list(
    stage_configs: list[Any],
    default_sampling_params_list: list[OmniSamplingParams],
    *,
    diffusion_params: OmniSamplingParams | None = None,
    replace_diffusion_params: bool = False,
) -> list[OmniSamplingParams]:
    """Build effective sampling params for a multi-stage request.

    When ``replace_diffusion_params`` is set, diffusion stages receive cloned
    request-level diffusion params. That preserves existing image and video
    endpoint behavior where request params replace diffusion defaults without
    sharing mutable state across stages.
    """
    sampling_params_list: list[OmniSamplingParams] = []
    for idx, stage_cfg in enumerate(stage_configs):
        if replace_diffusion_params and get_stage_type(stage_cfg) == "diffusion" and diffusion_params is not None:
            sampling_params_list.append(clone_sampling_params(diffusion_params))
        else:
            sampling_params_list.append(
                resolve_stage_sampling_params(
                    stage_cfg,
                    idx,
                    default_sampling_params_list,
                    diffusion_params=diffusion_params,
                )
            )
    return sampling_params_list
