# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import inspect
import weakref
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

import vllm_omni.diffusion.models.helios.pipeline_helios as pipeline_module
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.helios.pipeline_helios import (
    DEFAULT_NUM_LATENT_FRAMES_PER_CHUNK,
    HeliosPipeline,
)
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _DecodedOutput:
    pass


class _FakeVAE:
    device = torch.device("cpu")
    dtype = torch.float32
    config = SimpleNamespace(z_dim=4)

    def __init__(self):
        self.shapes = []
        self.return_dict_values = []
        self.output_ref = None
        self.latent_ref = None

    def decode(self, latents, *, return_dict):
        assert torch.is_inference_mode_enabled()
        assert latents.device == self.device
        assert latents.dtype == self.dtype
        self.shapes.append(tuple(latents.shape))
        self.return_dict_values.append(return_dict)
        self.latent_ref = weakref.ref(latents)
        output = _DecodedOutput()
        self.output_ref = weakref.ref(output)
        return (output,)


def _pipeline():
    pipeline = object.__new__(HeliosPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.vae = _FakeVAE()
    pipeline.vae_scale_factor_spatial = 8
    pipeline.vae_scale_factor_temporal = 4
    pipeline.transformer = Mock()
    return pipeline


def _runner(additional_config, pipeline=None):
    runner = object.__new__(DiffusionModelRunner)
    runner.od_config = OmniDiffusionConfig.__new__(OmniDiffusionConfig)
    runner.od_config.additional_config = additional_config
    runner.pipeline = _pipeline() if pipeline is None else pipeline
    return runner


def test_missing_none_and_empty_profiles_keep_warmup_disabled():
    additional_configs: tuple[Any, ...] = (
        None,
        {},
        {"helios_vae_warmup_profiles": None},
        {"helios_vae_warmup_profiles": []},
    )
    for additional_config in additional_configs:
        runner = _runner(additional_config)

        runner.run_helios_vae_warmup()

        assert runner.pipeline.vae.shapes == []


def test_profiles_derive_exact_resolution_shapes_from_shared_chunk_default():
    pipeline = _pipeline()
    runner = _runner(
        {
            "helios_vae_warmup_profiles": [
                {"height": 384, "width": 640},
                {"height": 640, "width": 384},
            ]
        },
        pipeline,
    )

    runner.run_helios_vae_warmup()

    assert DEFAULT_NUM_LATENT_FRAMES_PER_CHUNK == 9
    assert pipeline.vae.shapes == [(1, 4, 9, 48, 80), (1, 4, 9, 80, 48)]
    assert pipeline.vae.return_dict_values == [False, False]
    assert inspect.signature(HeliosPipeline.forward).parameters["num_latent_frames_per_chunk"].default == 9
    pipeline.transformer.assert_not_called()


def test_duplicate_resolutions_are_deduplicated_in_input_order():
    pipeline = _pipeline()
    profile = {"height": 384, "width": 640}

    pipeline.warmup_vae_profiles([profile, profile])

    assert pipeline.vae.shapes == [(1, 4, 9, 48, 80)]


@pytest.mark.parametrize(
    "profile",
    [{"height": 0, "width": 640}, {"height": 384, "width": 0}, {"height": 385, "width": 640}],
)
def test_invalid_geometry_is_rejected_without_coercion(profile):
    pipeline = _pipeline()

    with pytest.raises(ValueError):
        pipeline.warmup_vae_profiles([profile])

    assert pipeline.vae.shapes == []


@pytest.mark.parametrize(
    "profile",
    [
        {"height": 384},
        {"width": 640},
        {"height": 384, "width": 640, "num_frames": 33},
    ],
)
def test_profile_requires_exact_height_width_keys(profile):
    with pytest.raises(ValueError, match="exactly height and width"):
        _pipeline().warmup_vae_profiles([profile])


@pytest.mark.parametrize("profiles", [{}, "384x640"])
def test_malformed_configured_profiles_raise(profiles):
    runner = _runner({"helios_vae_warmup_profiles": profiles})

    with pytest.raises(TypeError):
        runner.run_helios_vae_warmup()


def test_configured_profiles_on_unsupported_pipeline_raise_clear_error():
    runner = _runner({"helios_vae_warmup_profiles": [{"height": 384, "width": 640}]}, object())

    with pytest.raises(ValueError, match="only supported by HeliosPipeline"):
        runner.run_helios_vae_warmup()


def test_warmup_discards_latent_and_decode_output():
    pipeline = _pipeline()

    pipeline.warmup_vae_profiles([{"height": 384, "width": 640}])

    assert pipeline.vae.output_ref() is None
    assert pipeline.vae.latent_ref() is None
    pipeline.transformer.assert_not_called()


def test_platform_synchronization_brackets_decode(monkeypatch):
    pipeline = _pipeline()
    events = []
    monkeypatch.setattr(
        pipeline_module,
        "current_omni_platform",
        SimpleNamespace(is_available=lambda: True, synchronize=lambda: events.append("sync")),
    )
    decode = pipeline.vae.decode

    def recording_decode(*args, **kwargs):
        events.append("decode")
        return decode(*args, **kwargs)

    pipeline.vae.decode = recording_decode
    pipeline.warmup_vae_profiles([{"height": 384, "width": 640}])

    assert events == ["sync", "decode", "sync"]


def test_cpu_warmup_skips_platform_sync_when_unavailable(monkeypatch):
    pipeline = _pipeline()
    monkeypatch.setattr(pipeline_module, "current_omni_platform", SimpleNamespace(is_available=lambda: False))

    pipeline.warmup_vae_profiles([{"height": 384, "width": 640}])

    assert pipeline.vae.shapes == [(1, 4, 9, 48, 80)]


def test_diffusion_worker_delegates_warmup_to_runner():
    worker = object.__new__(DiffusionWorker)
    worker.model_runner = SimpleNamespace(run_helios_vae_warmup=Mock())

    worker.run_helios_vae_warmup()

    worker.model_runner.run_helios_vae_warmup.assert_called_once_with()
