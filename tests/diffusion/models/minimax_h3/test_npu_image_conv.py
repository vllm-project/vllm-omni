# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU dispatch, padding and fallback checks; NPU benchmarks run separately."""

import importlib.util
import sys
import weakref
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
_PATH = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/minimax_h3/ops/vae/npu_image_conv.py"
_SPEC = importlib.util.spec_from_file_location("h3_npu_image_conv_test", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
ops = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ops)


@dataclass(frozen=True)
class Device:
    type: str = "npu"


def test_non_npu_keeps_original_modules():
    model = nn.Module()
    model.encoder = nn.Conv3d(4, 8, 1)
    model.quant_conv = nn.Conv3d(8, 8, 1)
    model.decoder = nn.Conv3d(8, 4, 1)
    assert ops.install_npu_image_conv(model, device=torch.device("cpu")) == 0
    assert not hasattr(model.encoder, "_omni_image_conv_forward")


def test_scope_resets_after_exception():
    assert ops._IMAGE_ENCODE.get() is None
    with pytest.raises(ValueError), ops.image_encode_context():
        outer = ops._IMAGE_ENCODE.get()
        assert outer is not None
        with ops.image_encode_context():
            inner = ops._IMAGE_ENCODE.get()
            assert inner is not None and inner is not outer
        assert ops._IMAGE_ENCODE.get() is outer
        raise ValueError("expected")
    assert ops._IMAGE_ENCODE.get() is None


def test_unsupported_conv_contract():
    assert not ops._supported_layout(nn.Conv3d(4, 8, 3, padding=1))


def _checkpoint_model(monkeypatch):
    class SpatialParallelConv3d(nn.Conv3d):
        def __init__(self):
            super().__init__(4, 8, 3, padding=1)
            self.causal = True
            self.pad_mode_t = "constant"
            self.pad_mode = "reflect"

    class EncoderFCN3D(nn.Module):
        pass

    namespace = sys.modules[__name__]
    monkeypatch.setattr(namespace, "SpatialParallelConv3d", SpatialParallelConv3d, raising=False)
    monkeypatch.setattr(namespace, "EncoderFCN3D", EncoderFCN3D, raising=False)
    model = nn.Module()
    model.encoder = EncoderFCN3D()
    model.encoder.conv = SpatialParallelConv3d()
    return model


def test_checkpoint_type_identity_rejects_same_name_subclass(monkeypatch):
    model = _checkpoint_model(monkeypatch)
    impostor = type("SpatialParallelConv3d", (type(model.encoder.conv),), {})
    model.quant_conv = impostor()
    assert ops.install_npu_image_conv(model, device=Device()) == 1
    assert hasattr(model.encoder.conv, "_omni_image_conv_forward")
    assert not hasattr(model.quant_conv, "_omni_image_conv_forward")


@pytest.mark.parametrize("attribute", ["causal", "pad_mode_t", "pad_mode"])
def test_missing_checkpoint_attribute_keeps_original_forward(monkeypatch, attribute):
    model = _checkpoint_model(monkeypatch)
    conv = model.encoder.conv
    original = conv.forward
    delattr(conv, attribute)
    assert ops.install_npu_image_conv(model, device=Device()) == 0
    assert conv.forward == original


@pytest.mark.parametrize("padding_mode", ["reflect", "replicate", "circular"])
def test_nonzero_padding_mode_keeps_original_forward(monkeypatch, padding_mode):
    model = nn.Module()
    model.encoder = nn.Conv3d(4, 8, 1, padding_mode=padding_mode).eval()
    original = model.encoder.forward
    assert ops.install_npu_image_conv(model, device=Device()) == 0
    assert model.encoder.forward == original


def test_installer_is_scoped_and_preserves_overrides(monkeypatch):
    model = nn.Module()
    model.encoder = nn.Sequential(nn.Conv3d(4, 8, 1), nn.Conv3d(8, 8, 1)).eval()
    model.quant_conv = nn.Conv3d(8, 8, 1).eval()
    model.decoder = nn.Conv3d(8, 4, 1).eval()

    def custom(x):
        return x

    model.encoder[1].forward = custom
    device = Device()
    assert ops.install_npu_image_conv(model, device=device) == 2
    original = model.encoder[0]._omni_image_conv_forward
    assert ops.install_npu_image_conv(model, device=device) == 2
    assert model.encoder[0]._omni_image_conv_forward is original
    assert model.encoder[1].forward is custom
    assert "forward" not in model.decoder.__dict__
    x = torch.randn(1, 4, 2, 3, 5)
    # CPU and multi-frame inputs retain the original forward even in scope.
    with torch.inference_mode(), ops.image_encode_context():
        assert torch.equal(model.encoder[0](x), original(x))


def _checkpoint_causal_conv(conv: nn.Conv3d, x: torch.Tensor) -> torch.Tensor:
    # Mirrors BaseConv3d.forward for T=1 in the public checkpoint source
    # (MiniMaxAI/MiniMax-H3@42ed227, Ref2VA/video_vae/conv.py): spatial
    # padding first, then K_t - 1 zero frames before the single frame.
    ph, pw = conv.padding[1:]
    x = F.pad(x, (pw, pw, ph, ph, 0, 0), mode=conv.pad_mode)
    x = torch.cat([x.new_zeros(*x.shape[:2], conv.kernel_size[0] - 1, *x.shape[3:]), x], dim=2)
    return F.conv3d(x, conv.weight, conv.bias, conv.stride)


# padding/stride pairs used by the checkpoint encoder (vae_cnn.py): ResNet and
# conv_in/conv_out use padding=1; the downsample conv uses padding=(1, 0, 0)
# with a spatial stride.
@pytest.mark.parametrize(("padding", "stride"), [(1, 1), ((1, 0, 0), (1, 2, 2))])
@pytest.mark.parametrize("pad_mode", ["constant", "reflect", "replicate"])
@pytest.mark.parametrize("spatial", [(3, 5), (8, 9)])
def test_reduction_matches_checkpoint_causal_conv(padding, stride, pad_mode, spatial, monkeypatch):
    conv = nn.Conv3d(4, 8, 3, padding=padding, stride=stride).eval()
    conv.pad_mode = pad_mode
    with torch.no_grad():
        # Small integers keep every sum exact in FP32. Each temporal slice
        # gets a different scale, so selecting the wrong slice changes the
        # result instead of matching by construction.
        base = (torch.arange(conv.weight[:, :, 0].numel()) % 7 - 3).view_as(conv.weight[:, :, 0]).float()
        for t, scale in enumerate((1.0, 16.0, 256.0)):
            conv.weight[:, :, t] = base * scale
        conv.bias.copy_(torch.arange(8, dtype=torch.float32))
    x = (torch.arange(4 * spatial[0] * spatial[1]) % 5 - 2).float().view(1, 4, 1, *spatial)
    expected = _checkpoint_causal_conv(conv, x)
    monkeypatch.setattr(ops, "_eligible", lambda *_: True)
    actual = ops._forward(conv, x)
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)
    with torch.no_grad():
        conv.weight[:, :, -1] = conv.weight[:, :, 0]
    assert not torch.equal(ops._forward(conv, x), expected)


@pytest.mark.parametrize("fail_encode", [False, True])
def test_packed_weight_is_reused_and_released_with_image_scope(fail_encode):
    conv = nn.Conv3d(2, 3, 3).eval()
    error_context = pytest.raises(RuntimeError, match="synthetic encode failure") if fail_encode else nullcontext()
    with error_context, ops.image_encode_context():
        packed = ops._weight(conv)
        assert ops._weight(conv) is packed
        assert torch.equal(packed, conv.weight[:, :, -1:].contiguous())
        packed_ref = weakref.ref(packed)
        del packed
        if fail_encode:
            raise RuntimeError("synthetic encode failure")
    assert packed_ref() is None
    with torch.no_grad():
        conv.weight.add_(1)
    with ops.image_encode_context():
        assert torch.equal(ops._weight(conv), conv.weight[:, :, -1:].contiguous())


def test_packed_weight_is_not_shared_across_npu_streams(monkeypatch):
    # A meta tensor exercises device stream lookup without NPU hardware.
    conv = nn.Conv3d(2, 3, 3, device="meta")
    stream = SimpleNamespace(npu_stream=1)
    backend = SimpleNamespace(current_stream=lambda _: stream)
    monkeypatch.setattr(torch, "get_device_module", lambda _: backend)

    with ops.image_encode_context():
        first = ops._weight(conv)
        assert ops._weight(conv) is first
        stream.npu_stream = 2
        second = ops._weight(conv)
        assert second is not first
        assert ops._weight(conv) is second
        del stream.npu_stream
        assert ops._weight(conv) is not second
        assert ops._weight(conv) is not ops._weight(conv)
