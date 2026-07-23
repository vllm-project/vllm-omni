"""
Common definitions for controlling what tests run where.
"""

from collections.abc import Callable
from enum import StrEnum, auto
from typing import NamedTuple, TypeAlias

from pytest import MarkDecorator

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni

# All builder funcs take no params and return a path
TinyDiffusionBuilder: TypeAlias = Callable[[], str]


class DiffusionAccs(StrEnum):
    """Supported acceleration types / test settings for Diffusion Models."""

    HSDP = auto()
    TEA_CACHE = auto()
    CACHE_DIT = auto()
    SEQUENCE_PARALLEL = auto()
    CFG_PARALLEL = auto()
    TENSOR_PARALLEL = auto()
    CPU_OFFLOAD = auto()
    LAYERWISE_OFFLOAD = auto()
    VAE_PATCH_PARALLEL = auto()


class DiffusionTasks(StrEnum):
    """Supported tasks for Diffusion Models."""

    TEXT_TO_IMAGE = auto()
    IMAGE_TO_IMAGE = auto()
    TEXT_TO_VIDEO = auto()
    # Text to audio, etc should be added here as needed


class DiffusionModelTestOpts(NamedTuple):
    """Configuration for one Diffusion model's tests."""

    # HF model name for real-weight tests (advanced_model / full_model level).
    # For now, whether we use the real weights vs tiny weights in the common tests
    # depends on the run level.
    model: str

    # Creates a tiny model for the given architecture. We should always use tiny
    # model weights for tests that do not require us to check the model quality.
    builder: TinyDiffusionBuilder

    # Actual tasks which controls the tests actually run
    supported_tasks: list[DiffusionTasks]

    # Additional acceleration groups to run beyond the base case (no acceleration).
    # The base case is always run for every model in the test settings. None means
    # we only run the base case.
    extra_test_groups: list[list[DiffusionAccs]] | None = None

    # Pytest Marks for this model. This may be useful for selecting which models
    # we want to run where, similar to the way vLLM's multimodal tests mark some
    # as core models to always run in the CI.
    # Example: https://github.com/vllm-project/vllm/blob/v0.23.0/tests/models/multimodal/generation/test_common.py#L131
    marks: list[MarkDecorator] | None = None

    # When True (default), online tests only run the base case (no accelerations).
    # When False, online tests run all test_groups, same as offline. This should be
    # True unless there is a good reason for it not to be, because the execution of
    # the acceleration should be the same on both codepaths, and CLI parsing etc should
    # be tested by tests adding the acceleration, and not per model.
    online_base_only: bool = True

    # Additional checks to run for the base case.
    check_multi_output: bool = True  # Runs multiple generations in one request
    check_determinism: bool = True  # Runs 2 generations with the same seed and check determinism


### Mappings & utils for building offline Omni() instances given a list of enabled accelerations
ACC_OMNI_KWARGS = {
    DiffusionAccs.VAE_PATCH_PARALLEL: {"vae_use_tiling": True},
    DiffusionAccs.CPU_OFFLOAD: {"enable_cpu_offload": True},
    DiffusionAccs.LAYERWISE_OFFLOAD: {"enable_layerwise_offload": True},
    DiffusionAccs.CACHE_DIT: {"cache_backend": "cache_dit"},
    DiffusionAccs.TEA_CACHE: {"cache_backend": "tea_cache"},
}

ACC_PARALLEL_KWARGS = {
    DiffusionAccs.HSDP: {"use_hsdp": True, "hsdp_shard_size": 2},
    DiffusionAccs.TENSOR_PARALLEL: {"tensor_parallel_size": 2},
    DiffusionAccs.CFG_PARALLEL: {"cfg_parallel_size": 2},
    DiffusionAccs.VAE_PATCH_PARALLEL: {"vae_patch_parallel_size": 2},
    # For SP, we don't run ring here to conserve devices, since it is easy
    # to blow up the number of needed devices fast. Compatibility for ring
    # and ulysses together should be tested generically outside of these tests.
    DiffusionAccs.SEQUENCE_PARALLEL: {"ulysses_degree": 2},
}

ACC_DEVICE_COUNT_KEYS = {
    DiffusionAccs.TENSOR_PARALLEL: "tensor_parallel_size",
    DiffusionAccs.CFG_PARALLEL: "cfg_parallel_size",
    DiffusionAccs.SEQUENCE_PARALLEL: "ulysses_degree",
    DiffusionAccs.VAE_PATCH_PARALLEL: "vae_patch_parallel_size",
    DiffusionAccs.HSDP: "hsdp_shard_size",
}

### CLI args for launching an OmniServer subprocess per acceleration.
# Follows the same pattern as e2e online serving tests, which hand-code
# their server_args lists in OmniServerParams.
ACC_SERVER_ARGS: dict[DiffusionAccs, list[str]] = {
    DiffusionAccs.HSDP: ["--use-hsdp", "--hsdp-shard-size", "2"],
    DiffusionAccs.TEA_CACHE: ["--cache-backend", "tea_cache"],
    DiffusionAccs.CACHE_DIT: ["--cache-backend", "cache_dit"],
    DiffusionAccs.SEQUENCE_PARALLEL: ["--usp", "2"],
    DiffusionAccs.CFG_PARALLEL: ["--cfg-parallel-size", "2"],
    DiffusionAccs.TENSOR_PARALLEL: ["--tensor-parallel-size", "2"],
    DiffusionAccs.CPU_OFFLOAD: ["--enable-cpu-offload"],
    DiffusionAccs.LAYERWISE_OFFLOAD: ["--enable-layerwise-offload"],
    DiffusionAccs.VAE_PATCH_PARALLEL: ["--vae-use-tiling", "--vae-patch-parallel-size", "2"],
}

## TODO ^ These should be in the same object, it's getting messy.


def get_required_device_count(accelerations: list[DiffusionAccs] | None) -> int:
    """Compute the minimum number of devices needed for a set of accelerations.
    The total is the product of all parallel dimensions (defaulting to 1).

    If not enough devices are available for a test group's accelerations,
    that test will be skipped."""
    count = 1
    if accelerations is None:
        return count

    for acc in accelerations:
        key = ACC_DEVICE_COUNT_KEYS.get(acc)
        if key is not None:
            count *= ACC_PARALLEL_KWARGS[acc][key]
    return count


def build_parallel_config_from_diff_accelerations(accelerations: list[DiffusionAccs]) -> DiffusionParallelConfig | None:
    """Given a list of accelerations pertaining to the current test group,
    build the parallel config needed for the Omni() object (if any)."""
    config_kwargs = {}
    for acc in accelerations:
        update_dict = ACC_PARALLEL_KWARGS.get(acc, {})
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
        update_dict = ACC_OMNI_KWARGS.get(acc, {})
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
        args.extend(ACC_SERVER_ARGS.get(acc, []))
    return args
