from .logger import get_logger
from .models import lookup_model_spec

logger = get_logger(__name__)


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
        logger.info("skipping sampling params check because spec is not found")
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
            if "type" not in sp:
                raise RuntimeError("Internal error: unknown sampling parameter type")
            if sp["type"] != stages[i]:
                raise ValueError(
                    f"Sampling parameter type ({sp['type']}) does not match "
                    f"stage type ({stages[i]}) at index {i} for model {model_name}."
                )
    elif isinstance(sampling_param_list, dict):
        if "type" not in sampling_param_list:
            raise RuntimeError("Internal error: unknown sampling parameter type")
        # Check that the provided single sampling param matches all stages
        elif any(stage != sampling_param_list["type"] for stage in stages):
            raise ValueError(
                f"When passing a single sampling parameter node, all stages of the model must match "
                f"the provided sampling parameter's type. "
                f"However, the stages of model {model_name} are: {stages}. "
                f"The provided sampling parameter is {sampling_param_list['type']}"
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
        sampling_param_list.update(params_to_add)
    elif sampling_param_list is None:
        sampling_param_list = params_to_add.copy()
    else:
        for i, stage in enumerate(stages):
            if stage == stage_type:
                sampling_param_list[i].update(params_to_add)

    return sampling_param_list
