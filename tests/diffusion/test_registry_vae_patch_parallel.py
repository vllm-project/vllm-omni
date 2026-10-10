# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""VAE patch-parallel selection in the diffusion model registry."""

from typing import ClassVar

import pytest
from torch import nn

from vllm_omni.diffusion import registry
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import DistributedAutoencoderKL_base
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import DistributedVaeMixin
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DeclaredPipeline(nn.Module, SupportsComponentDiscovery):
    _dit_modules: ClassVar[list[str]] = []
    _encoder_modules: ClassVar[list[str]] = []
    _vae_modules: ClassVar[list[str]] = []


class _RecordingVae(nn.Module, DistributedVaeMixin):
    def __init__(self):
        super().__init__()
        self.use_slicing = False
        self.use_tiling = False
        self.parallel_settings = None

    def set_parallel_size(self, parallel_size: int, mode: str = "tile") -> None:
        self.parallel_settings = (parallel_size, mode)


def _initialize(
    mocker, *, attr: str, pp_size: int, use_tiling: bool = False, use_slicing: bool = False, mode: str = "tile"
):
    class _RecordingBatchVae(_RecordingVae, DistributedAutoencoderKL_base):
        pass

    class _Pipeline(_DeclaredPipeline):
        _vae_modules = [attr]

        def __init__(self, *, od_config):
            super().__init__()
            setattr(self, attr, _RecordingBatchVae() if mode == "batch" else _RecordingVae())

    mocker.patch.object(registry.DiffusionModelRegistry, "_try_load_model_cls", return_value=_Pipeline)
    mocker.patch.object(registry, "_apply_sequence_parallel_if_enabled")
    config = OmniDiffusionConfig(
        model_class_name="RecordingPipeline",
        parallel_config=DiffusionParallelConfig(vae_patch_parallel_size=pp_size, vae_parallel_mode=mode),
        vae_use_tiling=use_tiling,
        vae_use_slicing=use_slicing,
    )
    return registry.initialize_model(config), config


def test_declared_gen_vae_receives_patch_parallel_configuration(mocker):
    pipeline, config = _initialize(mocker, attr="gen_vae", pp_size=2)

    assert config.vae_use_tiling is True
    assert pipeline.gen_vae.use_tiling is True
    assert pipeline.gen_vae.parallel_settings == (2, "tile")
    assert not hasattr(pipeline, "vae")


def test_conventional_vae_still_receives_patch_parallel_configuration(mocker):
    pipeline, config = _initialize(mocker, attr="vae", pp_size=2)

    assert config.vae_use_tiling is True
    assert pipeline.vae.use_tiling is True
    assert pipeline.vae.parallel_settings == (2, "tile")


def test_declared_gen_vae_is_untouched_when_patch_parallel_disabled(mocker):
    pipeline, config = _initialize(mocker, attr="gen_vae", pp_size=1)

    assert config.vae_use_tiling is False
    assert pipeline.gen_vae.use_tiling is False
    assert pipeline.gen_vae.parallel_settings is None


def test_declared_gen_vae_honors_explicit_tiling_with_one_rank(mocker):
    pipeline, config = _initialize(mocker, attr="gen_vae", pp_size=1, use_tiling=True)

    assert config.vae_use_tiling is True
    assert pipeline.gen_vae.use_tiling is True


@pytest.mark.parametrize("attr", ["vae", "gen_vae"])
@pytest.mark.parametrize("pp_size", [1, 2])
def test_slicing_only_initialization_configures_discovered_vae(mocker, attr, pp_size):
    pipeline, config = _initialize(mocker, attr=attr, pp_size=pp_size, use_slicing=True)
    vae = getattr(pipeline, attr)

    assert vae.use_slicing is True
    assert vae.use_tiling is (pp_size > 1)
    assert config.vae_use_tiling is (pp_size > 1)


def test_declared_gen_vae_keeps_batch_mode_without_forcing_tiling(mocker):
    pipeline, config = _initialize(mocker, attr="gen_vae", pp_size=2, mode="batch")

    assert pipeline.gen_vae.parallel_settings == (2, "batch")
    assert pipeline.gen_vae.use_tiling is False
    assert config.vae_use_tiling is False


@pytest.mark.parametrize("pp_size", [1, 2])
def test_plain_vae_warns_only_for_unsupported_parallelism(mocker, pp_size):
    class _Pipeline(nn.Module):
        def __init__(self, *, od_config):
            super().__init__()
            self.vae = nn.Module()
            self.vae.use_tiling = False

    mocker.patch.object(registry.DiffusionModelRegistry, "_try_load_model_cls", return_value=_Pipeline)
    mocker.patch.object(registry, "_apply_sequence_parallel_if_enabled")
    warning = mocker.patch.object(registry.logger, "warning")
    config = OmniDiffusionConfig(
        model_class_name="PlainVaePipeline",
        vae_use_tiling=True,
        parallel_config=DiffusionParallelConfig(vae_patch_parallel_size=pp_size),
    )

    pipeline = registry.initialize_model(config)

    assert pipeline.vae.use_tiling is True
    if pp_size == 1:
        warning.assert_not_called()
    else:
        warning.assert_called_once()


@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("memory_mode", ["slicing", "tiling"])
def test_multiple_declared_vaes_are_not_configured_ambiguously(mocker, pp_size, memory_mode):
    class _Pipeline(_DeclaredPipeline):
        _vae_modules = ["gen_vae", "preview_vae"]

        def __init__(self, *, od_config):
            super().__init__()
            self.gen_vae = _RecordingVae()
            self.preview_vae = _RecordingVae()

    mocker.patch.object(registry.DiffusionModelRegistry, "_try_load_model_cls", return_value=_Pipeline)
    mocker.patch.object(registry, "_apply_sequence_parallel_if_enabled")
    warning = mocker.patch.object(registry.logger, "warning")
    config = OmniDiffusionConfig(
        model_class_name="RecordingPipeline",
        parallel_config=DiffusionParallelConfig(vae_patch_parallel_size=pp_size),
        vae_use_slicing=memory_mode == "slicing",
        vae_use_tiling=memory_mode == "tiling",
    )

    pipeline = registry.initialize_model(config)

    assert config.vae_use_tiling is (memory_mode == "tiling")
    assert pipeline.gen_vae.parallel_settings is None
    assert pipeline.preview_vae.parallel_settings is None
    assert pipeline.gen_vae.use_slicing is False
    assert pipeline.preview_vae.use_slicing is False
    warning.assert_called_once()


@pytest.mark.parametrize("mode", ["spatial_shard_height", "spatial_shard_width"])
def test_spatial_mode_only_discovers_vae_and_reaches_mode_validation(mocker, mode):
    class _GuardedVae(nn.Module, DistributedAutoencoderKL_base):
        pass

    class _Pipeline(_DeclaredPipeline):
        _vae_modules = ["gen_vae"]

        def __init__(self, *, od_config):
            super().__init__()
            self.gen_vae = _GuardedVae()

    mocker.patch.object(registry.DiffusionModelRegistry, "_try_load_model_cls", return_value=_Pipeline)
    mocker.patch.object(registry, "_apply_sequence_parallel_if_enabled")
    config = OmniDiffusionConfig(
        model_class_name="RecordingPipeline",
        parallel_config=DiffusionParallelConfig(vae_patch_parallel_size=1, vae_parallel_mode=mode),
    )

    assert config.vae_use_tiling is False
    assert config.vae_use_slicing is False
    with pytest.raises(ValueError, match="supports only.*tile.*batch"):
        registry.initialize_model(config)


@pytest.mark.parametrize("mode", ["spatial_shard_height", "spatial_shard_width"])
def test_spatial_mode_only_warns_when_no_compatible_vae_is_discovered(mocker, mode):
    class _Pipeline(_DeclaredPipeline):
        _vae_modules = ["gen_vae"]

        def __init__(self, *, od_config):
            super().__init__()
            self.gen_vae = nn.Identity()

    mocker.patch.object(registry.DiffusionModelRegistry, "_try_load_model_cls", return_value=_Pipeline)
    mocker.patch.object(registry, "_apply_sequence_parallel_if_enabled")
    warning = mocker.patch.object(registry.logger, "warning")
    config = OmniDiffusionConfig(
        model_class_name="PlainDeclaredVaePipeline",
        parallel_config=DiffusionParallelConfig(vae_patch_parallel_size=1, vae_parallel_mode=mode),
    )

    registry.initialize_model(config)

    warning.assert_called_once()


def test_duplicate_declared_path_to_same_vae_is_configured_once(mocker):
    class _Pipeline(_DeclaredPipeline):
        _vae_modules = ["gen_vae", "wrapper.gen_vae"]

        def __init__(self, *, od_config):
            super().__init__()
            self.gen_vae = _RecordingVae()
            self.wrapper = nn.Module()
            self.wrapper.gen_vae = self.gen_vae

    mocker.patch.object(registry.DiffusionModelRegistry, "_try_load_model_cls", return_value=_Pipeline)
    mocker.patch.object(registry, "_apply_sequence_parallel_if_enabled")
    config = OmniDiffusionConfig(
        model_class_name="RecordingPipeline",
        parallel_config=DiffusionParallelConfig(vae_patch_parallel_size=2),
    )

    pipeline = registry.initialize_model(config)

    assert pipeline.gen_vae.parallel_settings == (2, "tile")
