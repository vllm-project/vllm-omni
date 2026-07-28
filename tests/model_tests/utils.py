"""
Utilities for resolving real models to their tiny model equivalents.
"""

import logging

from tests.model_tests.config_types import ACC_DESCRIPTORS, DiffusionAccs
from tests.model_tests.model_settings import DIFFUSION_TEST_SETTINGS
from vllm_omni.diffusion.data import DiffusionParallelConfig, resolve_model_class_name
from vllm_omni.entrypoints.omni import Omni

logger = logging.getLogger(__name__)


def resolve_tiny_model_path(model: str) -> str:
    """Given a real model name/path, resolve it to a tiny model path.

    Raises ValueError if the pipeline class cannot be determined (invalid
    model). Returns the original model path if no tiny builder exists yet."""
    pipeline_class = resolve_model_class_name(model)
    if pipeline_class is None:
        raise ValueError(
            f"Cannot resolve pipeline class for model: {model}. The model path may be invalid or its config unreadable."
        )

    test_opts = DIFFUSION_TEST_SETTINGS.get(pipeline_class)
    if test_opts is None:
        logger.warning(
            "No tiny model builder for pipeline %s (model: %s). Using original model.",
            pipeline_class,
            model,
        )
        return model

    return test_opts.builder()


### helpers for building Diffusion Models
def get_required_device_count(accelerations: list[DiffusionAccs] | None) -> int:
    """Compute the minimum number of devices needed for a set of accelerations.
    The total is the product of all parallel dimensions (defaulting to 1).

    If not enough devices are available for a test group's accelerations,
    that test will be skipped."""
    count = 1
    if accelerations is None:
        return count

    for acc in accelerations:
        descriptor = ACC_DESCRIPTORS[acc]
        if descriptor.device_count_key is not None:
            count *= descriptor.omni_parallel_kwargs[descriptor.device_count_key]
    return count


def build_parallel_config_from_diff_accelerations(accelerations: list[DiffusionAccs]) -> DiffusionParallelConfig | None:
    """Given a list of accelerations pertaining to the current test group,
    build the parallel config needed for the Omni() object (if any)."""
    config_kwargs = {}
    for acc in accelerations:
        update_dict = ACC_DESCRIPTORS[acc].omni_parallel_kwargs
        config_kwargs.update(update_dict)
    if config_kwargs:
        return DiffusionParallelConfig(**config_kwargs)
    return None


### Offline Omni() object builder
def build_omni_from_diff_accelerations(accelerations: list[DiffusionAccs] | None, **kwargs) -> Omni:
    """Given one or more acceleration types, build the corresponding Omni() object."""
    # Coerce to a list and build the parallel config, since that depends on the accelerations
    if accelerations is None:
        accelerations = []
    parallel_config = build_parallel_config_from_diff_accelerations(accelerations)

    # Then add anything else that's a top-level kwarg
    acc_kwargs = {}
    if parallel_config is not None:
        acc_kwargs["parallel_config"] = parallel_config
    for acc in accelerations:
        update_dict = ACC_DESCRIPTORS[acc].omni_kwargs
        acc_kwargs.update(update_dict)

    # Keys passed through should mostly be things like enforce_eager;
    # if there's overlap, it's probably due to a misconfiguration
    shared_keys = acc_kwargs.keys() & kwargs.keys()
    if shared_keys:
        raise ValueError(f"Explicit Omni kwargs and inferred Omni kwargs for accelerations overlap: {shared_keys}")
    omni_kwargs = {**acc_kwargs, **kwargs}
    return Omni(**omni_kwargs)


### Online server flag builder
def build_server_args_from_diff_accelerations(accelerations: list[DiffusionAccs] | None) -> list[str]:
    """Given one or more acceleration types, build the corresponding CLI args
    for launching an OmniServer subprocess."""
    if accelerations is None:
        return []
    args = []
    for acc in accelerations:
        acc_cli_args = ACC_DESCRIPTORS[acc].cli_args
        args.extend(acc_cli_args)
    return args


### TODO - fix the qwen3omni preproc layer index
#  - fix slow start
#  - fix audio out
