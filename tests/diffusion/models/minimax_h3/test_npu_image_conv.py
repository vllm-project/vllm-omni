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


def test_default_off_and_non_npu(monkeypatch):
    model = nn.Module()
    model.encoder = nn.Conv3d(4, 8, 1)
    model.quant_conv = nn.Conv3d(8, 8, 1)
    model.decoder = nn.Conv3d(8, 4, 1)
    monkeypatch.delenv(ops._ENV, raising=False)
    assert ops.install_npu_image_conv(model, device=torch.device("privateuseone")) == 0
    monkeypatch.setenv(ops._ENV, "1")
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
    monkeypatch.setenv(ops._ENV, "1")
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
    monkeypatch.setenv(ops._ENV, "1")
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


@pytest.mark.parametrize("spatial", [(1, 1), (3, 5), (8, 9)])
def test_reduction_preserves_causal_padding_math(spatial, monkeypatch):
    conv = nn.Conv3d(4, 8, 3, padding=1).eval()
    conv.pad_mode = "constant"
    with torch.no_grad():
        conv.weight.fill_(0.125)
        conv.bias.fill_(0.25)
    x = torch.arange(4 * spatial[0] * spatial[1], dtype=torch.float32).view(1, 4, 1, *spatial)
    expected = F.conv3d(F.pad(x, (1, 1, 1, 1, 2, 0)), conv.weight, conv.bias)
    monkeypatch.setattr(ops, "_eligible", lambda *_: True)
    actual = ops._forward(conv, x)
    assert torch.equal(actual, expected)


def _cached_model(monkeypatch):
    model = nn.Module()
    model.encoder = nn.Conv3d(2, 3, 3).eval()
    model.decoder = nn.Conv3d(2, 3, 3).eval()
    monkeypatch.setenv(ops._ENV, "1")
    monkeypatch.setenv(ops._CACHE_ENV, "1")
    # CPU tests isolate cache semantics; the remote-code allowlist and NPU
    # numerical dispatch are validated separately against real checkpoint code.
    monkeypatch.setattr(ops, "_supported_module", lambda *_: True)
    assert ops.install_npu_image_conv(model, device=Device()) == 1
    return model


def test_cache_is_separately_opt_in(monkeypatch):
    monkeypatch.setenv(ops._ENV, "1")
    monkeypatch.delenv(ops._CACHE_ENV, raising=False)
    model = nn.Module()
    model.encoder = nn.Conv3d(2, 3, 1).eval()
    assert ops.install_npu_image_conv(model, device=Device()) == 1
    assert not hasattr(model.encoder, "_omni_image_conv_weight")
    monkeypatch.delenv(ops._ENV)
    monkeypatch.setenv(ops._CACHE_ENV, "1")
    assert ops.install_npu_image_conv(model, device=Device()) == 0


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


def test_stream_change_and_unknown_stream_invalidate(monkeypatch):
    model = _cached_model(monkeypatch)
    conv = model.encoder
    original = ops._weight_key
    first = ops._weight(conv)
    monkeypatch.setattr(ops, "_weight_key", lambda w: (*original(w), "other-stream"))
    assert ops._weight(conv) is not first
    monkeypatch.setattr(ops, "_weight_key", lambda w: None)
    ops._weight(conv)
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
