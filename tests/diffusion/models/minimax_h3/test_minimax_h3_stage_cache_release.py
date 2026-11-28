# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

import vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 as pipeline_module
import vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3_decoder as decoder_module
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.offloader import BoundedAllocatorCache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_ENV = pipeline_module.MINIMAX_H3_STAGE_CACHE_RELEASE_ENV
_GIB = 1 << 30
_CUDA = torch.device("cuda:1")


class _Device:
    def __init__(self, monkeypatch):
        self.memory_gib = (58, 54, 40)
        self.is_cuda = True
        self.integrated = False
        self.fraction = 1.0
        self.released = 0
        platform = pipeline_module.current_omni_platform
        monkeypatch.setattr(platform, "is_cuda", lambda: self.is_cuda)
        monkeypatch.setattr(platform, "is_integrated_gpu", lambda index: self.integrated and index == _CUDA.index)
        monkeypatch.setattr(platform, "empty_cache", self._empty_cache, raising=False)
        monkeypatch.setattr(platform, "get_device_memory", self._device_memory)
        monkeypatch.setattr(torch.accelerator, "memory_reserved", lambda _device: self.memory_gib[0] * _GIB)
        monkeypatch.setattr(torch.accelerator, "memory_allocated", lambda _device: self.memory_gib[1] * _GIB)
        monkeypatch.setattr(
            torch.cuda, "get_per_process_memory_fraction", lambda device: self.fraction if device == _CUDA else 1.0
        )

    def _empty_cache(self):
        self.released += 1

    def _device_memory(self, _device):
        if self.memory_gib[2] is None:
            raise RuntimeError("no device memory telemetry")
        return self.memory_gib[2] * _GIB, 100 * _GIB


@pytest.fixture
def device(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    return _Device(monkeypatch)


def _policy(target=_CUDA, **fields):
    return pipeline_module._resident_stage_allocator_cache(OmniDiffusionConfig(**fields), target)


def _pipeline(stage_cache, dlo_cache=None):
    pipeline = object.__new__(pipeline_module.MiniMaxH3Pipeline)
    pipeline._dlo_component_cache = dlo_cache
    pipeline._stage_allocator_cache = stage_cache
    return pipeline


def _decoder(monkeypatch, tmp_path):
    monkeypatch.setattr(decoder_module, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(decoder_module, "resolve_minimax_h3_decoder_model_path", lambda *_args: str(tmp_path))
    monkeypatch.setattr(decoder_module, "resolve_h3_lightvae_decoder_path", lambda _config: None)
    monkeypatch.setattr(decoder_module, "MiniMaxH3VideoVAE", lambda *_args, **_kwargs: torch.nn.Identity())
    monkeypatch.setattr(decoder_module, "MiniMaxH3AudioVAE", lambda *_args, **_kwargs: torch.nn.Identity())
    od_config = OmniDiffusionConfig(model=str(tmp_path), task_type="ref2va")
    return decoder_module.MiniMaxH3DecoderPipeline(od_config=od_config)


@pytest.mark.parametrize(
    ("memory_gib", "released"),
    [
        pytest.param((58, 54, 40), 0, id="within-bounds"),
        pytest.param((75, 50, 25), 0, id="at-bounds"),
        pytest.param((60, 34, 40), 1, id="26pct-idle"),
        pytest.param((70, 66, 24), 1, id="24pct-free"),
        pytest.param((58, 54, None), 1, id="no-telemetry"),
    ],
)
def test_resident_stage_cache_releases_past_bounds(device, memory_gib, released):
    device.memory_gib = memory_gib
    _pipeline(_policy())._release_stage_cache()
    assert device.released == released


def test_component_cache_release_stays_forced(device):
    _pipeline(_policy(), BoundedAllocatorCache(_CUDA))._release_stage_cache()
    assert device.released == 1


@pytest.mark.parametrize(("fraction", "released"), [(1.0, 0), (0.99, 1)], ids=["uncapped", "capped"])
def test_capped_process_keeps_forced_release(device, fraction, released):
    pipeline = _pipeline(_policy())
    device.fraction = fraction
    pipeline._release_stage_cache()
    assert device.released == released


@pytest.mark.parametrize(("mode", "resident"), [(None, True), (" bounded ", True), ("force", False)])
def test_release_mode(monkeypatch, device, mode, resident):
    if mode is not None:
        monkeypatch.setenv(_ENV, mode)
    cache = _policy()
    assert (cache.device == _CUDA) if resident else (cache is None)


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"enable_cpu_offload": True}, id="offload"),
        pytest.param({"diffusion_offload_config": {"mode": "layer", "components": ["text_encoder"]}}, id="te-offload"),
        pytest.param({"parallel_config": DiffusionParallelConfig(use_hsdp=True, hsdp_shard_size=1)}, id="hsdp"),
    ],
)
def test_resident_policy_declines_offload_and_hsdp(device, fields):
    assert _policy(**fields) is None


@pytest.mark.parametrize(
    ("is_cuda", "integrated", "target"),
    [
        pytest.param(True, True, _CUDA, id="integrated"),
        pytest.param(False, False, _CUDA, id="non-cuda-platform"),
        pytest.param(True, False, torch.device("cpu"), id="cpu-device"),
    ],
)
def test_resident_policy_declines_off_discrete_cuda(device, is_cuda, integrated, target):
    device.is_cuda, device.integrated = is_cuda, integrated
    assert _policy(target) is None


@pytest.mark.parametrize("fields", [{}, {"enable_cpu_offload": True}], ids=["resident", "offload"])
def test_resident_policy_rejects_unknown_mode(monkeypatch, device, fields):
    monkeypatch.setenv(_ENV, "Force")
    with pytest.raises(ValueError, match=_ENV):
        _policy(**fields)


def test_decoder_stage_keeps_forced_release(monkeypatch, device, tmp_path):
    _decoder(monkeypatch, tmp_path)._release_stage_cache()
    assert device.released == 1


def test_decoder_stage_rejects_unknown_mode(monkeypatch, tmp_path):
    monkeypatch.setenv(_ENV, "Force")
    with pytest.raises(ValueError, match=_ENV):
        _decoder(monkeypatch, tmp_path)
