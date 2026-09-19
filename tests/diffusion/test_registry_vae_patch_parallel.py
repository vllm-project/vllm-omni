# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""VAE patch-parallel selection in the diffusion model registry."""

import pytest
from torch import nn

from vllm_omni.diffusion import registry
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import DistributedVaeMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _RecordingVae(nn.Module, DistributedVaeMixin):
    def __init__(self):
        super().__init__()
        self.use_slicing = False
        self.use_tiling = False
        self.parallel_settings = None

    def set_parallel_size(self, parallel_size: int, mode: str = "tile") -> None:
        self.parallel_settings = (parallel_size, mode)


def _initialize(mocker, *, attr: str, pp_size: int, use_tiling: bool = False):
    class _Pipeline(nn.Module):
        _vae_modules = [attr]

        def __init__(self, *, od_config):
            super().__init__()
            setattr(self, attr, _RecordingVae())

    mocker.patch.object(registry.DiffusionModelRegistry, "_try_load_model_cls", return_value=_Pipeline)
    mocker.patch.object(registry, "_apply_sequence_parallel_if_enabled")
    config = OmniDiffusionConfig(
        model_class_name="RecordingPipeline",
        parallel_config=DiffusionParallelConfig(vae_patch_parallel_size=pp_size),
        vae_use_tiling=use_tiling,
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


def test_multiple_declared_vaes_are_not_configured_ambiguously(mocker):
    class _Pipeline(nn.Module):
        _vae_modules = ["gen_vae", "preview_vae"]

        def __init__(self, *, od_config):
            super().__init__()
            self.gen_vae = _RecordingVae()
            self.preview_vae = _RecordingVae()

    mocker.patch.object(registry.DiffusionModelRegistry, "_try_load_model_cls", return_value=_Pipeline)
    mocker.patch.object(registry, "_apply_sequence_parallel_if_enabled")
    config = OmniDiffusionConfig(
        model_class_name="RecordingPipeline",
        parallel_config=DiffusionParallelConfig(vae_patch_parallel_size=2),
    )

    pipeline = registry.initialize_model(config)

    assert config.vae_use_tiling is False
    assert pipeline.gen_vae.parallel_settings is None
    assert pipeline.preview_vae.parallel_settings is None


def test_duplicate_declared_path_to_same_vae_is_configured_once(mocker):
    class _Pipeline(nn.Module):
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
