# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Encoder/VAE placement under cpu_offload; DiT follows the loader device context."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_DEVICE_STACK: list[str] = ["cpu"]
_REAL_TORCH_DEVICE = torch.device


class _FakeDevice:
    def __init__(self, type_: str):
        self.type = type_
        self.index = 0

    def __enter__(self):
        _DEVICE_STACK.append(self.type)
        return self

    def __exit__(self, *args):
        _DEVICE_STACK.pop()
        return False


def _fake_torch_device(spec):
    if isinstance(spec, _FakeDevice):
        return spec
    if isinstance(spec, _REAL_TORCH_DEVICE):
        return _FakeDevice(spec.type)
    return _FakeDevice(str(spec).split(":")[0])


class _RecordingTransformer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.probe_device_type = _DEVICE_STACK[-1]


class _MoveableModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.placed_device = _FakeDevice("cpu")
        self.weight = nn.Parameter(torch.zeros(1))

    def to(self, device, *args, **kwargs):
        if isinstance(device, _FakeDevice):
            self.placed_device = device
        elif isinstance(device, _REAL_TORCH_DEVICE):
            self.placed_device = _FakeDevice(device.type)
        else:
            self.placed_device = _FakeDevice(str(device).split(":")[0])
        return self

    @property
    def temperal_downsample(self):
        return [True, True, True]


def _run_pipeline_init(monkeypatch, *, enable_cpu_offload, loader_device):
    import vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image as pipe_mod
    from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline

    enc = _MoveableModule()
    vae = _MoveableModule()
    _DEVICE_STACK.clear()
    _DEVICE_STACK.append(loader_device)

    monkeypatch.setattr(pipe_mod, "get_local_device", lambda: _REAL_TORCH_DEVICE("cuda"))
    monkeypatch.setattr(pipe_mod.torch, "device", _fake_torch_device)
    monkeypatch.setattr(pipe_mod, "prefetch_subfolders", lambda *a, **k: None)
    monkeypatch.setattr(pipe_mod, "get_transformer_config_kwargs", lambda *a, **k: {})
    monkeypatch.setattr(pipe_mod, "QwenImageTransformer2DModel", _RecordingTransformer)
    monkeypatch.setattr(
        pipe_mod.FlowMatchEulerDiscreteScheduler,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: MagicMock()),
    )
    monkeypatch.setattr(
        pipe_mod.Qwen2Tokenizer,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: MagicMock()),
    )

    def _prefetch(factory, *args, **kwargs):
        subfolder = kwargs.get("subfolder")
        if subfolder == "text_encoder":
            return enc
        if subfolder == "vae":
            return vae
        raise AssertionError(f"unexpected prefetch subfolder={subfolder!r}")

    monkeypatch.setattr(pipe_mod, "from_pretrained_with_prefetch", _prefetch)
    monkeypatch.setattr(
        QwenImagePipeline,
        "setup_diffusion_pipeline_profiler",
        lambda self, **kwargs: None,
    )

    od_config = SimpleNamespace(
        model="/tmp/unused-qwen-image",
        parallel_config=SimpleNamespace(),
        tf_model_config={},
        quantization_config=None,
        enable_cpu_offload=enable_cpu_offload,
        enable_diffusion_pipeline_profiler=False,
    )
    return QwenImagePipeline(od_config=od_config)


def test_pipeline_cpu_offload_parks_encoder_vae_on_cpu(monkeypatch):
    """#7555: cpu_offload keeps BF16 encoder/VAE off the GPU during DiT init."""
    pipe = _run_pipeline_init(monkeypatch, enable_cpu_offload=True, loader_device="cuda")
    assert pipe.text_encoder.placed_device.type == "cpu"
    assert pipe.vae.placed_device.type == "cpu"
    assert pipe.transformer.probe_device_type == "cuda"


def test_pipeline_dit_follows_loader_cpu_without_override(monkeypatch):
    """Layerwise / HSDP: DiT must not be forced onto CUDA by the pipeline."""
    pipe = _run_pipeline_init(monkeypatch, enable_cpu_offload=False, loader_device="cpu")
    assert pipe.transformer.probe_device_type == "cpu"
    assert pipe.text_encoder.placed_device.type == "cuda"
    assert pipe.vae.placed_device.type == "cuda"
