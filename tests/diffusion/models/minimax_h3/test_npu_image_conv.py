# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU dispatch, padding and fallback checks; NPU benchmarks run separately."""

import importlib.util
from dataclasses import dataclass
from pathlib import Path

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
    assert not ops._IMAGE_ENCODE.get()
    with pytest.raises(ValueError), ops.image_encode_context():
        assert ops._IMAGE_ENCODE.get()
        with ops.image_encode_context():
            assert ops._IMAGE_ENCODE.get()
        raise ValueError("expected")
    assert not ops._IMAGE_ENCODE.get()


def test_unsupported_conv_contract():
    class OtherConv(nn.Conv3d):
        pass

    assert not ops._supported_module(OtherConv(4, 8, 3))
    assert not ops._supported_module(nn.Conv3d(4, 8, 3, padding=1))


def test_checkpoint_type_identity_rejects_same_name_subclass():
    class SpatialParallelConv3d(nn.Conv3d):
        pass

    impostor = type("SpatialParallelConv3d", (SpatialParallelConv3d,), {})
    assert ops._supported_module(SpatialParallelConv3d(4, 8, 3), SpatialParallelConv3d)
    assert not ops._supported_module(impostor(4, 8, 3), SpatialParallelConv3d)
    assert not ops._supported_module(SpatialParallelConv3d(4, 8, 3))


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


def _cached_model(monkeypatch):
    model = nn.Module()
    model.encoder = nn.Conv3d(2, 3, 3).eval()
    model.decoder = nn.Conv3d(2, 3, 3).eval()
    # CPU tests isolate cache semantics; the remote-code allowlist and NPU
    # numerical dispatch are validated separately against real checkpoint code.
    monkeypatch.setattr(ops, "_supported_module", lambda *_: True)
    assert ops.install_npu_image_conv(model, device=Device()) == 1
    return model


def test_npu_installer_enables_residency_scoped_cache():
    model = nn.Module()
    model.encoder = nn.Conv3d(2, 3, 1).eval()
    assert ops.install_npu_image_conv(model, device=Device()) == 1
    assert model.encoder._omni_image_conv_cache_enabled
    assert model.encoder._omni_image_conv_weight is None


def test_cache_reuses_slice_and_rebuilds_after_updates(monkeypatch):
    model = _cached_model(monkeypatch)
    conv = model.encoder
    keys = set(model.state_dict())
    with torch.inference_mode():
        first = ops._weight(conv)
        assert ops._weight(conv) is first
        assert torch.equal(first, conv.weight[:, :, -1:].contiguous())
        conv.weight.add_(0.25)
        updated = ops._weight(conv)
        assert updated is not first
        assert torch.equal(updated, conv.weight[:, :, -1:].contiguous())
    conv.weight = nn.Parameter(conv.weight.detach().clone())
    replaced = ops._weight(conv)
    assert replaced is not updated
    # Storage rebinding need not increment the Parameter version (DLO uses it).
    conv.weight.data = conv.weight.detach().clone() + 0.5
    rebound = ops._weight(conv)
    assert rebound is not replaced
    assert torch.equal(rebound, conv.weight[:, :, -1:].contiguous())
    assert set(model.state_dict()) == keys
    assert not hasattr(model.decoder, "_omni_image_conv_weight")


def test_cache_releases_on_state_load_conversion_and_explicit_offload(monkeypatch):
    model = _cached_model(monkeypatch)
    conv = model.encoder
    ops._weight(conv)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(state)
    assert conv._omni_image_conv_weight is None
    ops._weight(conv)
    model.double()
    assert conv._omni_image_conv_weight is None
    assert ops._weight(conv).dtype == torch.float64
    ops.clear_npu_image_conv_cache(model)
    assert conv._omni_image_conv_weight is None


def test_unversioned_weights_never_reuse_cache(monkeypatch):
    model = _cached_model(monkeypatch)
    with torch.inference_mode():
        model.encoder.weight = nn.Parameter(model.encoder.weight.clone())
        before = ops._weight(model.encoder)
        model.encoder.weight.add_(1)
        after = ops._weight(model.encoder)
        assert torch.equal(after, before + 1)
        assert model.encoder._omni_image_conv_weight is None


class _Stream:
    def __init__(self, handle):
        if handle is not None:
            self.npu_stream = handle


class _Backend:
    def __init__(self):
        self.stream = _Stream("A")
        self.devices = []

    def current_stream(self, device):
        self.devices.append(device)
        return self.stream


def test_stream_switch_rebuilds_and_unknown_stream_skips_cache(monkeypatch):
    # Meta weights take the device-stream branch of _weight_key on CPU hosts.
    model = _cached_model(monkeypatch)
    conv = model.encoder.to("meta")
    backend = _Backend()
    monkeypatch.setattr(torch, "get_device_module", lambda device: backend)

    def weight_on(handle):
        backend.stream = _Stream(handle)
        return ops._weight(conv)

    a1 = weight_on("A")
    assert weight_on("A") is a1
    b = weight_on("B")
    assert b is not a1
    a2 = weight_on("A")
    assert a2 is not a1 and a2 is not b
    assert conv._omni_image_conv_weight[1][-1] == "A"
    assert backend.devices and all(device == conv.weight.device for device in backend.devices)
    # A stream without a native handle cannot order reuse, so never cache.
    weight_on(None)
    assert conv._omni_image_conv_weight is None


def test_fallback_releases_existing_cache(monkeypatch):
    model = _cached_model(monkeypatch)
    conv = model.encoder
    ops._weight(conv)
    x = torch.randn(1, 2, 3, 4, 5)
    expected = conv._omni_image_conv_forward(x)
    # CPU / multi-frame fallback must retain the original convolution.
    assert torch.equal(conv(x), expected)
    assert conv._omni_image_conv_weight is None
