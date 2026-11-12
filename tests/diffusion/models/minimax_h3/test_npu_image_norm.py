# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
_PATH = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/minimax_h3/ops/vae/group_norm_silu.py"
_SPEC = importlib.util.spec_from_file_location("h3_npu_group_norm_silu_test", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
ops = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ops)


@dataclass(frozen=True)
class Device:
    type: str = "npu"


def test_non_npu_is_not_installed():
    model = nn.Module()
    model.encoder = nn.GroupNorm(4, 8)
    assert ops.install_npu_group_norm_silu(model, device=torch.device("cpu")) == 0
    assert not hasattr(model.encoder, "_omni_group_norm_silu_supported")


def test_scope_and_cpu_fallback():
    x = torch.randn(1, 8, 1, 3, 5)
    norm = nn.GroupNorm(4, 8).eval()
    assert not ops._eligible(x, norm, None)
    with pytest.raises(ValueError), ops.image_encode_context():
        assert ops._IMAGE_ENCODE.get()
        with torch.inference_mode():
            assert not ops._eligible(x, norm, None)
        raise ValueError("expected")
    assert not ops._IMAGE_ENCODE.get()


def norm_silu(x, norm, cond=None):
    if cond is None:
        return F.silu(norm(x))
    else:
        return F.silu(norm(x, cond))


class ResnetBlock3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.GroupNorm(4, 8)

    def forward(self, x):
        return norm_silu(x, self.norm) + x


class EncoderFCN3D(ResnetBlock3D):
    pass


def get_group_norm_3d():
    return nn.GroupNorm


def _merge_time_to_batch(x):
    batch, channels, frames, height, width = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, 1, height, width)


def _split_time_from_batch(x, batch):
    _, channels, _, height, width = x.shape
    frames = x.shape[0] // batch
    return x.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()


class TemporalIsolatedSpatialParallelGroupNorm(nn.GroupNorm):
    def forward(self, x):
        return _split_time_from_batch(super().forward(_merge_time_to_batch(x)), x.shape[0])


def test_private_dispatch_keeps_decoder_and_other_instances_unchanged(monkeypatch):
    calls = []

    def fused(x, weight, bias, groups, eps):
        calls.append(x)
        return (F.silu(F.group_norm(x, groups, weight, bias, eps)),)

    backend = ModuleType("torch_npu")
    monkeypatch.setattr(backend, "npu_group_norm_silu", fused, raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", backend)
    model = nn.Module()
    model.encoder = EncoderFCN3D().eval()
    model.decoder = ResnetBlock3D().eval()
    other = ResnetBlock3D().eval()
    global_helper = ResnetBlock3D.forward.__globals__["norm_silu"]
    device = Device()
    assert ops.install_npu_group_norm_silu(model, device=device) == 1
    assert ops.install_npu_group_norm_silu(model, device=device) == 1
    assert ResnetBlock3D.forward.__globals__["norm_silu"] is global_helper
    assert "forward" not in model.decoder.__dict__
    assert "forward" not in other.__dict__
    x = torch.randn(1, 8, 1, 3, 5)
    expected = global_helper(x, model.encoder.norm) + x
    assert torch.equal(model.encoder(x), expected)
    assert not calls
    # Exercise the installed dispatcher on CPU without pretending this is
    # numerical qualification of the actual NPU kernel.
    monkeypatch.setattr(ops, "_eligible", lambda *_: True)
    assert torch.equal(model.encoder(x), expected)
    assert len(calls) == 1
    model.decoder(x)
    other(x)
    assert len(calls) == 1


def test_unknown_encoder_and_norm_subclasses_keep_original_forward(monkeypatch):
    # Class-name equality is insufficient: it must be the checkpoint's type.
    impostor = type("EncoderFCN3D", (EncoderFCN3D,), {})
    model = nn.Module()
    model.encoder = impostor().eval()
    assert ops.install_npu_group_norm_silu(model, device=Device()) == 0
    assert "forward" not in model.encoder.__dict__

    backend = ModuleType("torch_npu")
    monkeypatch.setattr(backend, "npu_group_norm_silu", lambda *_: (), raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", backend)
    model.encoder = EncoderFCN3D().eval()
    other_norm = type("SpatialParallelGroupNorm", (nn.GroupNorm,), {})
    model.encoder.norm = other_norm(4, 8).eval()
    assert ops.install_npu_group_norm_silu(model, device=Device()) == 1
    assert not model.encoder.norm._omni_group_norm_silu_supported


def test_video_temporal_isolation_and_independent_scope(monkeypatch):
    calls = []

    def fused(x, weight, bias, groups, eps):
        calls.append(tuple(x.shape))
        return (F.silu(F.group_norm(x, groups, weight, bias, eps)),)

    backend = ModuleType("torch_npu")
    monkeypatch.setattr(backend, "npu_group_norm_silu", fused, raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", backend)
    model = nn.Module()
    model.encoder = EncoderFCN3D().eval()
    model.encoder.norm = TemporalIsolatedSpatialParallelGroupNorm(4, 8).eval()
    assert ops.install_npu_group_norm_silu(model, device=Device()) == 1
    monkeypatch.setattr(ops, "_eligible", lambda *_: ops._VIDEO_ENCODE.get())

    x = torch.randn(2, 8, 3, 3, 5)
    expected = norm_silu(x, model.encoder.norm) + x
    with torch.inference_mode(), ops.image_encode_context():
        assert torch.equal(model.encoder(x), expected)
    assert calls == []

    with torch.inference_mode(), ops.video_encode_context():
        assert torch.equal(model.encoder(x), expected)
    assert calls == [(6, 8, 1, 3, 5)]


def test_missing_backend_does_not_install(monkeypatch):
    backend = ModuleType("torch_npu")
    monkeypatch.setitem(sys.modules, "torch_npu", backend)
    model = nn.Module()
    model.encoder = EncoderFCN3D().eval()
    assert ops.install_npu_group_norm_silu(model, device=Device()) == 0
    assert "forward" not in model.encoder.__dict__


def test_video_ordinary_group_norm_keeps_ncthw(monkeypatch):
    calls = []

    def fused(x, weight, bias, groups, eps):
        calls.append(tuple(x.shape))
        return (F.silu(F.group_norm(x, groups, weight, bias, eps)),)

    backend = ModuleType("torch_npu")
    monkeypatch.setattr(backend, "npu_group_norm_silu", fused, raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", backend)
    model = nn.Module()
    model.encoder = EncoderFCN3D().eval()
    assert ops.install_npu_group_norm_silu(model, device=Device()) == 1
    monkeypatch.setattr(ops, "_eligible", lambda *_: ops._VIDEO_ENCODE.get())

    x = torch.randn(2, 8, 3, 3, 5)
    expected = norm_silu(x, model.encoder.norm) + x
    with torch.inference_mode(), ops.video_encode_context():
        assert torch.equal(model.encoder(x), expected)
    assert calls == [(2, 8, 3, 3, 5)]


def test_temporal_helper_missing_falls_back_to_original_ncthw(monkeypatch):
    calls = []

    def fused(*args):
        calls.append(args[0])
        return (args[0],)

    backend = ModuleType("torch_npu")
    monkeypatch.setattr(backend, "npu_group_norm_silu", fused, raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", backend)
    namespace = get_group_norm_3d.__globals__
    merge_time = namespace["_merge_time_to_batch"]
    monkeypatch.setitem(namespace, "_merge_time_to_batch", None)
    model = nn.Module()
    model.encoder = EncoderFCN3D().eval()
    model.encoder.norm = TemporalIsolatedSpatialParallelGroupNorm(4, 8).eval()
    assert ops.install_npu_group_norm_silu(model, device=Device()) == 1
    monkeypatch.setitem(namespace, "_merge_time_to_batch", merge_time)
    monkeypatch.setattr(ops, "_eligible", lambda *_: True)

    x = torch.randn(2, 8, 3, 3, 5)
    expected = norm_silu(x, model.encoder.norm) + x
    with torch.inference_mode(), ops.video_encode_context():
        assert torch.equal(model.encoder(x), expected)
    assert calls == []
