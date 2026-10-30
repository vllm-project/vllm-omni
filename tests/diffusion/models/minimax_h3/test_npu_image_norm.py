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
_PATH = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/minimax_h3/ops/vae/npu_image_norm.py"
_SPEC = importlib.util.spec_from_file_location("h3_npu_image_norm_test", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
ops = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ops)


@dataclass(frozen=True)
class Device:
    type: str = "npu"


def test_default_off_and_non_npu(monkeypatch):
    model = nn.Module()
    model.encoder = nn.GroupNorm(4, 8)
    monkeypatch.delenv(ops._ENV, raising=False)
    assert ops.install_npu_image_norm(model, device=torch.device("cpu")) == 0
    monkeypatch.setenv(ops._ENV, "1")
    assert ops.install_npu_image_norm(model, device=torch.device("cpu")) == 0
    assert not hasattr(model.encoder, "_omni_image_norm_supported")


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


def test_private_dispatch_keeps_decoder_and_other_instances_unchanged(monkeypatch):
    calls = []

    def fused(x, weight, bias, groups, eps):
        calls.append(x)
        return (F.silu(F.group_norm(x, groups, weight, bias, eps)),)

    backend = ModuleType("torch_npu")
    monkeypatch.setattr(backend, "npu_group_norm_silu", fused, raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", backend)
    monkeypatch.setenv(ops._ENV, "1")
    model = nn.Module()
    model.encoder = EncoderFCN3D().eval()
    model.decoder = ResnetBlock3D().eval()
    other = ResnetBlock3D().eval()
    global_helper = ResnetBlock3D.forward.__globals__["norm_silu"]
    device = Device()
    assert ops.install_npu_image_norm(model, device=device) == 1
    assert ops.install_npu_image_norm(model, device=device) == 1
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
    monkeypatch.setenv(ops._ENV, "1")
    # Class-name equality is insufficient: it must be the checkpoint's type.
    impostor = type("EncoderFCN3D", (EncoderFCN3D,), {})
    model = nn.Module()
    model.encoder = impostor().eval()
    assert ops.install_npu_image_norm(model, device=Device()) == 0
    assert "forward" not in model.encoder.__dict__

    backend = ModuleType("torch_npu")
    monkeypatch.setattr(backend, "npu_group_norm_silu", lambda *_: (), raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", backend)
    model.encoder = EncoderFCN3D().eval()
    other_norm = type("SpatialParallelGroupNorm", (nn.GroupNorm,), {})
    model.encoder.norm = other_norm(4, 8).eval()
    assert ops.install_npu_image_norm(model, device=Device()) == 1
    assert not model.encoder.norm._omni_image_norm_supported
