import math

from .logger import get_logger
from .models import lookup_model_spec
from .types import (
    MINIMAX_H3_CONTROL_TYPES,
    AutoregressionSamplingParams,
    DiffusionSamplingParams,
    MiniMaxH3Control,
)

logger = get_logger(__name__)

_MINIMAX_H3_CONTROL_KEYS = frozenset(MiniMaxH3Control.__annotations__)


def validate_minimax_h3_control(control: MiniMaxH3Control) -> None:
    """Validate the proposed CTRL-01 H3 control contract without mutating it."""
    unknown_keys = set(control) - _MINIMAX_H3_CONTROL_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown MiniMax-H3 control fields: {', '.join(sorted(unknown_keys))}.")

    control_type = control.get("control_type")
    if control_type not in MINIMAX_H3_CONTROL_TYPES:
        supported = ", ".join(MINIMAX_H3_CONTROL_TYPES)
        raise ValueError(f"Unsupported MiniMax-H3 control_type {control_type!r}; expected one of: {supported}.")

    strength = control.get("control_context_scale")
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise ValueError("control_context_scale must be a finite, non-negative number.")
    try:
        normalized_strength = float(strength)
    except (OverflowError, ValueError):
        raise ValueError("control_context_scale must be a finite, non-negative number.") from None
    if not math.isfinite(normalized_strength) or normalized_strength < 0:
        raise ValueError("control_context_scale must be a finite, non-negative number.")

    mask = control.get("mask")
    mask_video = control.get("mask_video")
    has_mask = mask is not None or mask_video is not None
    if mask is not None and mask_video is not None:
        raise ValueError("Provide only one of mask or mask_video, not both.")
    if control.get("source_video") is not None and not has_mask:
        raise ValueError("source_video requires mask or mask_video.")
    if control_type == "inpaint" and not has_mask:
        raise ValueError("inpaint requires mask or mask_video.")
    if control_type != "inpaint" and control.get("control_video") is None:
        raise ValueError(f"{control_type} requires control_video.")


def validate_model_and_sampling_params_types(
    model_name: str,
    sampling_param_list: dict | list[dict] | None = None,
):
    # Check if model name exists
    if not model_name:
        raise ValueError("Model name must not be empty.")

    # Skip if no spec or no sampling params
    pipeline_spec, _ = lookup_model_spec(model_name)
    if pipeline_spec is None:
        logger.info(f"skipping sampling params check because spec for {model_name} is not found")
        return
    if sampling_param_list is None:
        return

    # Check the number of stages and their data types
    stages = pipeline_spec["stages"]
    if isinstance(sampling_param_list, list):
        # Check that the lengths match
        if len(stages) != len(sampling_param_list):
            raise ValueError(
                f"Sampling parameter list length {len(sampling_param_list)} does not match "
                f"number of stages {len(stages)} for model {model_name}."
            )
        # Check that each stage's type match
        for i, sp in enumerate(sampling_param_list):
            if not _check_sampling_param_matches_stage(sp, stages[i]):
                raise ValueError(
                    f"Sampling parameter type ({sp.__class__.__name__}) does not match "
                    f"stage type ({stages[i]}) at index {i} for model {model_name}."
                )
    elif isinstance(sampling_param_list, dict):
        # Check that the provided single sampling param matches all stages
        for i, stage in enumerate(stages):
            if not _check_sampling_param_matches_stage(sampling_param_list, stage):
                raise ValueError(
                    f"Provided single sampling parameter type ({sampling_param_list.__class__.__name__}) must match "
                    f"the types of all stages of the model. "
                    f"However, stage {i} of model {model_name} is of type {stage}."
                )


def add_sampling_parameters_to_stage(
    model_name: str,
    sampling_param_list: dict | list[dict] | None,
    stage_type: str,
    /,
    **params_to_add,
) -> dict | list[dict]:
    """
    Given a model's name and the sampling parameter list to query this model,
    add arbitrary additional parameters to the sampling parameters of all stages of the given type.
    """
    pipeline_spec, _ = lookup_model_spec(model_name)
    if not pipeline_spec:
        logger.warning(
            f"Since the model {model_name} is not in our list, we cannot ensure if "
            f"the fields ({tuple(params_to_add.keys())}) are added to the correct stage's sampling params. "
            f"We will do it heuristically."
        )
        pipeline_spec = {"stages": ["diffusion"]}

    stages = pipeline_spec["stages"]
    if isinstance(sampling_param_list, dict):
        sampling_param_list = sampling_param_list.__class__(sampling_param_list)
        sampling_param_list.update(params_to_add)
    elif sampling_param_list is None:
        sampling_param_list = params_to_add.copy()
    else:
        for i, stage in enumerate(stages):
            if stage == stage_type:
                stage_param = sampling_param_list[i]
                stage_param = stage_param.__class__(stage_param)
                stage_param.update(params_to_add)
                sampling_param_list[i] = stage_param

    return sampling_param_list


def _check_sampling_param_matches_stage(sampling_param: dict, stage_type: str) -> bool:
    if stage_type == "autoregression":
        return isinstance(sampling_param, AutoregressionSamplingParams)
    if stage_type == "diffusion":
        return isinstance(sampling_param, DiffusionSamplingParams)
    raise RuntimeError(f"Internal error: unknown stage type {stage_type}.")
