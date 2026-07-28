from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any, TypeAlias

from pytest import MarkDecorator

# All builder funcs take no params and return a path
TinyModelBuilder: TypeAlias = Callable[[], str]


class ModelTasks(StrEnum):
    """Supported model tasks."""

    TEXT_TO_TEXT = auto()
    MM_TO_TEXT = auto()  # Text + Other Modalities
    TEXT_TO_IMAGE = auto()
    IMAGE_TO_IMAGE = auto()
    TEXT_TO_VIDEO = auto()
    TEXT_TO_AUDIO = auto()


@dataclass
class ModelTestOpts:
    """Test fields that apply to all model types."""

    # HF model name for real-weight tests (advanced_model / full_model level).
    # For now, whether we use the real weights vs tiny weights in the common tests
    # depends on the run level.
    model: str

    # Creates a tiny model for the given architecture. We should always use tiny
    # model weights for tests that do not require us to check the model quality.
    builder: TinyModelBuilder

    # Actual tasks which controls the tests actually run. These are agnostic
    # to the implementation details / engine type used by the model.
    supported_tasks: list[ModelTasks]

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


@dataclass
class DiffusionModelTestOpts(ModelTestOpts):
    """Configuration for one Diffusion model's tests."""

    # Additional acceleration groups to run beyond the base case (no acceleration).
    # The base case is always run for every model in the test settings. None means
    # we only run the base case.
    extra_test_groups: list[list[DiffusionAccs]] | None = None


class OmniModelTestOpts(ModelTestOpts):
    # The number of devices required to run these tests.
    # TODO (Alex): We need a clean way of handling things like TP
    # analogous to what we do for single stage diffusion. This should
    # probably be moved into ModelTestOpts and implemented generically,
    # since it would also be useful in considering models like Bagel.
    num_devices_required: int = 1

    # Testing options for omni
    supports_prefix_caching: bool
    supports_async_chunking: bool


@dataclass
class AccelerationDescriptor:
    """Describes how to enable an acceleration for both offline / online cases,
    as well as how many devices will be leveraged."""

    omni_kwargs: dict[str, Any]
    omni_parallel_kwargs: dict[str, Any]
    device_count_key: str | None
    cli_args: list[str]


ACC_DESCRIPTORS: dict[DiffusionAccs, AccelerationDescriptor] = {
    DiffusionAccs.HSDP: AccelerationDescriptor(
        omni_kwargs={},
        omni_parallel_kwargs={"use_hsdp": True, "hsdp_shard_size": 2},
        device_count_key="hsdp_shard_size",
        cli_args=["--use-hsdp", "--hsdp-shard-size", "2"],
    ),
    DiffusionAccs.TEA_CACHE: AccelerationDescriptor(
        omni_kwargs={"cache_backend": "tea_cache"},
        omni_parallel_kwargs={},
        device_count_key=None,
        cli_args=["--cache-backend", "tea_cache"],
    ),
    DiffusionAccs.CACHE_DIT: AccelerationDescriptor(
        omni_kwargs={"cache_backend": "cache_dit"},
        omni_parallel_kwargs={},
        device_count_key=None,
        cli_args=["--cache-backend", "cache_dit"],
    ),
    DiffusionAccs.SEQUENCE_PARALLEL: AccelerationDescriptor(
        omni_kwargs={},
        omni_parallel_kwargs={"ulysses_degree": 2},
        device_count_key="ulysses_degree",
        cli_args=["--usp", "2"],
    ),
    DiffusionAccs.CFG_PARALLEL: AccelerationDescriptor(
        omni_kwargs={},
        omni_parallel_kwargs={"cfg_parallel_size": 2},
        device_count_key="cfg_parallel_size",
        cli_args=["--cfg-parallel-size", "2"],
    ),
    DiffusionAccs.TENSOR_PARALLEL: AccelerationDescriptor(
        omni_kwargs={},
        omni_parallel_kwargs={"tensor_parallel_size": 2},
        device_count_key="tensor_parallel_size",
        cli_args=["--tensor-parallel-size", "2"],
    ),
    DiffusionAccs.CPU_OFFLOAD: AccelerationDescriptor(
        omni_kwargs={"enable_cpu_offload": True},
        omni_parallel_kwargs={},
        device_count_key=None,
        cli_args=["--enable-cpu-offload"],
    ),
    DiffusionAccs.LAYERWISE_OFFLOAD: AccelerationDescriptor(
        omni_kwargs={"enable_layerwise_offload": True},
        omni_parallel_kwargs={},
        device_count_key=None,
        cli_args=["--enable-layerwise-offload"],
    ),
    DiffusionAccs.VAE_PATCH_PARALLEL: AccelerationDescriptor(
        omni_kwargs={"vae_use_tiling": True},
        omni_parallel_kwargs={"vae_patch_parallel_size": 2},
        device_count_key="vae_patch_parallel_size",
        cli_args=["--vae-use-tiling", "--vae-patch-parallel-size", "2"],
    ),
}
