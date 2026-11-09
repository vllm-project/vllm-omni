# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

from vllm import SamplingParams
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.utils import get_stage_type
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniSamplingParams

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


def to_sampling_params_list(
    engine_client: Any,
    sampling_params_list: list[dict],
    *,
    request_overrides: dict[str, Any] | None = None,
    merge_llm_defaults: bool = True,
) -> list[Any]:
    """Convert request dicts to stage-typed sampling params objects.

    Partial LLM dictionaries inherit deploy defaults, then ``request_overrides``
    (explicit top-level request parameters) on the comprehension stage. Explicit stage fields win.
    Video sessions set ``merge_llm_defaults=False`` so provided stage dicts
    use constructor defaults instead, preserving their replacement semantics.

    For diffusion stages, build ``OmniDiffusionSamplingParams`` so
    downstream ``StageDiffusionClient._sampling_params_to_dict`` (which
    requires a dataclass) works. For LLM stages build ``SamplingParams``.
    If callers provide params for fewer stages than the native pipeline has
    (for example AURA has three semantic models but four engine stages),
    append cloned deploy defaults for the omitted tail stages.
    """
    stage_configs = list(getattr(engine_client, "stage_configs", []) or [])
    default_params_list = list(getattr(engine_client, "default_sampling_params_list", []) or [])
    final_sampling_params_list: list[Any] = []
    for idx, sampling_params in enumerate(sampling_params_list):
        stage_config = stage_configs[idx] if idx < len(stage_configs) else None
        stage_type = get_stage_type(stage_config)
        target_cls = OmniDiffusionSamplingParams if stage_type == "diffusion" else SamplingParams
        if isinstance(sampling_params, dict):
            if target_cls is SamplingParams and merge_llm_defaults:
                default = default_params_list[idx] if idx < len(default_params_list) else SamplingParams()
                final_sampling_params_list.append(
                    merge_llm_sampling_params(default, stage_config, sampling_params, request_overrides)
                )
            else:
                final_sampling_params_list.append(target_cls(**sampling_params))
        elif isinstance(sampling_params, target_cls):
            final_sampling_params_list.append(sampling_params)
        elif isinstance(sampling_params, SamplingParams | OmniDiffusionSamplingParams):
            # Cross-typed (e.g. user passed SamplingParams but this is a
            # diffusion stage) — rebuild via a dict round-trip so we end
            # up with the correct target class.
            as_dict = {
                f.name: getattr(sampling_params, f.name)
                for f in (fields(sampling_params) if is_dataclass(sampling_params) else [])
            } or sampling_params.__dict__
            final_sampling_params_list.append(target_cls(**as_dict))
        else:
            raise ValueError(f"Invalid sampling params: {sampling_params}")
    for idx in range(len(final_sampling_params_list), len(stage_configs)):
        stage_config = stage_configs[idx]
        is_comprehension = (
            stage_config.get("is_comprehension", False)
            if isinstance(stage_config, Mapping)
            else getattr(stage_config, "is_comprehension", False)
        )
        if request_overrides is not None and is_comprehension and get_stage_type(stage_config) != "diffusion":
            default = default_params_list[idx] if idx < len(default_params_list) else SamplingParams()
            final_sampling_params_list.append(merge_llm_sampling_params(default, stage_config, {}, request_overrides))
        elif idx < len(default_params_list):
            final_sampling_params_list.append(clone_sampling_params(default_params_list[idx]))
        else:
            final_sampling_params_list.append(SamplingParams())
    return final_sampling_params_list


def merge_llm_sampling_params(
    default_params: SamplingParams | dict[str, Any],
    stage_config: Any,
    overrides: dict[str, Any],
    request_overrides: dict[str, Any] | None,
) -> SamplingParams:
    """Merge raw configuration before constructing normalized runtime params."""
    model_config = getattr(stage_config, "model_config", stage_config)
    if isinstance(model_config, Mapping):
        raw_defaults = model_config.get("default_sampling_params", default_params)
    else:
        raw_defaults = getattr(model_config, "default_sampling_params", default_params)
    if raw_defaults is None:
        merged = {}
    elif isinstance(raw_defaults, Mapping):
        merged = dict(raw_defaults)
    elif isinstance(default_params, dict):
        merged = dict(default_params)
    else:
        # Compatibility for clients without raw stage configuration. Keep
        # their runtime policies; explicit and inferred values are ambiguous.
        merged = {
            name: getattr(default_params, name)
            for name in SamplingParams.__struct_fields__
            if not name.startswith("_") and name != "output_text_buffer_length"
        }
    is_comprehension = (
        stage_config.get("is_comprehension", False)
        if isinstance(stage_config, Mapping)
        else getattr(stage_config, "is_comprehension", False)
    )
    if request_overrides is not None and is_comprehension:
        merged.update(request_overrides)
    extra_args = merged.get("extra_args")
    merged.update(overrides)
    if isinstance(extra_args, dict) and isinstance(overrides.get("extra_args"), dict):
        merged["extra_args"] = {**extra_args, **overrides["extra_args"]}
    # Constructors derive cache/stop/greedy state from the final values.
    # Nested dictionaries must never alias deploy defaults or request data.
    return SamplingParams(**copy.deepcopy(merged))
