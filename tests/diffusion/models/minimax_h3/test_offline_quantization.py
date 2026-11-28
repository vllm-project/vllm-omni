# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for ModelSlim offline MiniMax-H3 MX checkpoints."""

import json

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.models.minimax_h3.offline_quantization import (
    OfflineComponentCheckpoint,
    load_offline_vae_weights,
    npu_quant_family,
    reconstruct_mx_weight,
    resolve_offline_components,
    unpack_fp4,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _checkpoint(path, tensors, description, *, shard_scales=False):
    path.mkdir(parents=True, exist_ok=True)
    (path / "quant_model_description.json").write_text(json.dumps(description))
    if shard_scales:
        first = {name: tensor for name, tensor in tensors.items() if name.endswith(".weight")}
        second = {name: tensor for name, tensor in tensors.items() if name not in first}
        save_file(first, str(path / "weights.safetensors"))
        save_file(second, str(path / "scales.safetensors"))
        index = {name: "weights.safetensors" if name in first else "scales.safetensors" for name in tensors}
        (path / "quant_model_weights.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    else:
        save_file(tensors, str(path / "weights.safetensors"))
    return path


def _mx8(path, *, shard_scales=False):
    tensors = {
        "proj.weight": torch.ones(2, 32, dtype=torch.float8_e4m3fn),
        "proj.weight_scale": torch.tensor([[126], [128]], dtype=torch.uint8),
        "proj.bias": torch.tensor([1.0, 2.0]),
    }
    desc: dict[str, object] = {name: "W8A8_MXFP8" if name != "proj.bias" else "FLOAT" for name in tensors}
    desc.update(version="1.0.0", metadata={}, optional={})
    return _checkpoint(path, tensors, desc, shard_scales=shard_scales)


def test_fp4_all_codes_and_low_nibble_first():
    packed = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=torch.uint8)
    expected = torch.tensor([[0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]])
    torch.testing.assert_close(unpack_fp4(packed), expected)
    assert torch.signbit(unpack_fp4(packed)[0, 8])


@pytest.mark.parametrize("dual_groups", [1, 16])
def test_fp4_retains_dual_scales_at_512_and_32_granularity(dual_groups):
    # E2M1 1.0 in both nibbles, E8M0 exponent -1, and nonuniform FP32 dual scales.
    weight = torch.full((2, 256), 0x22, dtype=torch.uint8)
    scale = torch.full((2, 16), 126, dtype=torch.uint8)
    dual = torch.arange(1, 2 * dual_groups + 1, dtype=torch.float32).reshape(2, dual_groups, 1)
    actual = reconstruct_mx_weight(weight, scale, dual)
    expected = dual.squeeze(-1).repeat_interleave(512 // dual_groups, dim=1) * 0.5
    torch.testing.assert_close(actual, expected)


def test_mxfp8_exponent_scale_is_not_uint8_numeric_value():
    weight = torch.ones(2, 64, dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[127, 128], [126, 125]], dtype=torch.uint8)
    actual = reconstruct_mx_weight(weight, scale)
    expected = torch.tensor([[1.0, 2.0], [0.5, 0.25]]).repeat_interleave(32, dim=1)
    torch.testing.assert_close(actual, expected)


def test_device_family_accepts_a5_and_rejects_unknown():
    assert npu_quant_family("Ascend950PR_958b") == "mx"
    with pytest.raises(ValueError, match="unsupported"):
        npu_quant_family("Ascend910B2")


def test_native_auto_selects_mx_for_a5(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.minimax_h3.offline_quantization.resolve_npu_quant_family", lambda *_: "mx"
    )
    path = _mx8(tmp_path / "mx", shard_scales=True)
    native, reconstruct = resolve_offline_components({"path": str(path)})
    assert native["transformer"].execution_mode == "native_mx" and reconstruct
    checkpoints, dequantize = resolve_offline_components({"path": str(path), "execution_mode": "dequantize"})
    assert dequantize
    assert checkpoints["transformer"].schemes == {"W8A8_MXFP8"}


def test_missing_dual_scale_falls_back_to_single_scale(tmp_path):
    tensors = {
        "proj.weight": torch.full((1, 16), 0x22, dtype=torch.uint8),
        "proj.weight_scale": torch.full((1, 1), 126, dtype=torch.uint8),
    }
    path = _checkpoint(tmp_path / "mx4", tensors, {name: "W4A4_MXFP4_DUALSCALE" for name in tensors})
    checkpoint = OfflineComponentCheckpoint(path)
    actual = dict(checkpoint.iter_weights(dequantize=True))
    torch.testing.assert_close(actual["proj.weight"], torch.full((1, 32), 0.5))


def test_declared_missing_dual_scale_is_rejected(tmp_path):
    tensors = {
        "proj.weight": torch.full((1, 16), 0x22, dtype=torch.uint8),
        "proj.weight_scale": torch.full((1, 1), 126, dtype=torch.uint8),
    }
    description = {name: "W4A4_MXFP4_DUALSCALE" for name in tensors}
    description["proj.weight_dual_scale"] = "W4A4_MXFP4_DUALSCALE"
    path = _checkpoint(tmp_path / "mx4", tensors, description)
    with pytest.raises(ValueError, match="weight_dual_scale"):
        OfflineComponentCheckpoint(path)


def test_single_scale_mxfp4_is_accepted_and_reconstructed(tmp_path):
    tensors = {
        "proj.weight": torch.full((1, 16), 0x22, dtype=torch.uint8),
        "proj.weight_scale": torch.full((1, 1), 126, dtype=torch.uint8),
    }
    path = _checkpoint(tmp_path / "mx4-single", tensors, {name: "W4A4_MXFP4" for name in tensors})
    checkpoint = OfflineComponentCheckpoint(path)
    actual = dict(checkpoint.iter_weights(dequantize=True))
    torch.testing.assert_close(actual["proj.weight"], torch.full((1, 32), 0.5))


def test_single_scale_mxfp4_rejects_unexpected_dual_scale(tmp_path):
    tensors = {
        "proj.weight": torch.full((1, 16), 0x22, dtype=torch.uint8),
        "proj.weight_scale": torch.full((1, 1), 126, dtype=torch.uint8),
        "proj.weight_dual_scale": torch.ones(1, 1, 1),
    }
    path = _checkpoint(tmp_path / "mx4-single", tensors, {name: "W4A4_MXFP4" for name in tensors})
    with pytest.raises(ValueError, match="weight_dual_scale"):
        OfflineComponentCheckpoint(path)


def test_native_mx_loading_applies_all_delivery_scales_before_conversion(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.minimax_h3.offline_quantization.resolve_npu_quant_family", lambda *_: "mx"
    )
    mode = "native_mx"
    tensors = {
        "proj.weight": torch.full((2, 32), 0x22, dtype=torch.uint8),
        "proj.weight_scale": torch.tensor([[126, 127], [128, 125]], dtype=torch.uint8),
        "proj.weight_dual_scale": torch.tensor([[[0.3], [0.7]], [[1.1], [1.9]]]),
    }
    path = _checkpoint(tmp_path / "mx4", tensors, {name: "W4A4_MXFP4_DUALSCALE" for name in tensors})
    checkpoints, reconstruct = resolve_offline_components({"path": str(path), "execution_mode": mode})
    checkpoint = checkpoints["transformer"]
    assert reconstruct and checkpoint.execution_mode == mode
    actual = dict(checkpoint.iter_weights(dequantize=reconstruct))
    assert set(actual) == {"proj.weight"}
    expected = torch.tensor([[0.15, 0.7], [2.2, 0.475]]).repeat_interleave(32, dim=1)
    torch.testing.assert_close(actual["proj.weight"], expected)


@pytest.mark.parametrize("scheme", ["W8A8", "W8A8_DYNAMIC", "W4A4_DYNAMIC", "W8A8_UNKNOWN"])
def test_unsupported_scheme_is_rejected(tmp_path, scheme):
    path = _checkpoint(tmp_path / "bad", {"proj.weight": torch.ones(2, 2)}, {"proj.weight": scheme})
    with pytest.raises(ValueError, match="unsupported offline MX scheme"):
        OfflineComponentCheckpoint(path)


def test_index_cannot_escape_checkpoint_directory(tmp_path):
    path = _mx8(tmp_path / "mx8")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"proj.weight": "../other.safetensors"}})
    )
    with pytest.raises(ValueError, match="invalid or missing"):
        OfflineComponentCheckpoint(path)


def test_vae_overlay_is_strict_and_uses_reconstructed_weights(tmp_path):
    checkpoint = OfflineComponentCheckpoint(_mx8(tmp_path / "mx8"))
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(32, 2)
    load_offline_vae_weights(model, checkpoint)
    torch.testing.assert_close(model.proj.weight, torch.tensor([[0.5], [2.0]]).expand(2, 32))
    model.register_buffer("required", torch.ones(1))
    with pytest.raises(ValueError, match="incomplete"):
        load_offline_vae_weights(model, checkpoint)


@pytest.mark.parametrize("shard_scales", [False, True])
def test_vae_overlay_accepts_only_exact_duplicate_quantizer_bias(tmp_path, shard_scales):
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(32, 2)
    tensors = {
        "proj.weight": torch.ones(2, 32, dtype=torch.float8_e4m3fn),
        "proj.weight_scale": torch.tensor([[126], [128]], dtype=torch.uint8),
        "proj.bias": torch.tensor([1.0, 2.0]),
        "proj.weight_quantizer.bias": torch.tensor([1.0, 2.0]),
    }
    desc = {name: "W8A8_MXFP8" if name in {"proj.weight", "proj.weight_scale"} else "FLOAT" for name in tensors}
    checkpoint = OfflineComponentCheckpoint(_checkpoint(tmp_path / "vae", tensors, desc, shard_scales=shard_scales))
    load_offline_vae_weights(model, checkpoint)
    torch.testing.assert_close(model.proj.bias, tensors["proj.bias"], rtol=0, atol=0)
    torch.testing.assert_close(model.proj.weight, torch.tensor([[0.5], [2.0]]).expand(2, 32))


@pytest.mark.parametrize("invalid", ["value", "dtype", "shape", "missing_canonical", "unknown_target"])
def test_vae_overlay_rejects_conflicting_or_orphan_quantizer_bias(tmp_path, invalid):
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(32, 2)
    tensors = {
        "proj.weight": torch.ones(2, 32, dtype=torch.float8_e4m3fn),
        "proj.weight_scale": torch.full((2, 1), 127, dtype=torch.uint8),
        "proj.bias": torch.tensor([1.0, 2.0]),
        "proj.weight_quantizer.bias": torch.tensor([1.0, 2.0]),
    }
    if invalid == "value":
        tensors["proj.weight_quantizer.bias"][0] += 1
    elif invalid == "dtype":
        tensors["proj.weight_quantizer.bias"] = tensors["proj.weight_quantizer.bias"].to(torch.bfloat16)
    elif invalid == "shape":
        tensors["proj.weight_quantizer.bias"] = tensors["proj.weight_quantizer.bias"].reshape(1, 2)
    elif invalid == "missing_canonical":
        tensors.pop("proj.bias")
    else:
        tensors["unknown.bias"] = tensors["proj.bias"].clone()
        tensors["unknown.weight_quantizer.bias"] = tensors.pop("proj.weight_quantizer.bias")
    desc = {name: "W8A8_MXFP8" if name in {"proj.weight", "proj.weight_scale"} else "FLOAT" for name in tensors}
    checkpoint = OfflineComponentCheckpoint(_checkpoint(tmp_path / "vae", tensors, desc))
    with pytest.raises(ValueError, match="bias alias|does not match original architecture"):
        load_offline_vae_weights(model, checkpoint)


def test_vae_overlay_maps_legacy_weight_norm_names(tmp_path):
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(32, 2)
    model.conv = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(2, 2, 3))
    tensors = {
        "proj.weight": torch.ones(2, 32, dtype=torch.float8_e4m3fn),
        "proj.weight_scale": torch.full((2, 1), 127, dtype=torch.uint8),
        "proj.bias": torch.zeros(2),
        "conv.weight_g": torch.full((2, 1, 1), 2.0),
        "conv.weight_v": torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
        "conv.bias": torch.zeros(2),
    }
    description = {name: "W8A8_MXFP8" if name.startswith("proj.weight") else "FLOAT" for name in tensors}
    checkpoint = OfflineComponentCheckpoint(_checkpoint(tmp_path / "vae", tensors, description))
    load_offline_vae_weights(model, checkpoint)
    torch.testing.assert_close(model.conv.parametrizations.weight.original0, tensors["conv.weight_g"])
    torch.testing.assert_close(model.conv.parametrizations.weight.original1, tensors["conv.weight_v"])
    reference = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(2, 2, 3))
    reference.load_state_dict(
        {name.removeprefix("conv."): value for name, value in tensors.items() if name.startswith("conv.")}
    )
    torch.testing.assert_close(model.conv.weight, reference.weight)


@pytest.mark.parametrize("scale", [255])
def test_nonfinite_e8m0_is_rejected(scale):
    with pytest.raises(ValueError, match="finite E8M0"):
        reconstruct_mx_weight(torch.ones(1, 32), torch.full((1, 1), scale, dtype=torch.uint8))


def test_mxfp8_partial_last_group_matches_delivered_vae(tmp_path):
    tensors = {
        "proj.weight": torch.ones(2, 24, dtype=torch.float8_e4m3fn),
        "proj.weight_scale": torch.tensor([[126], [128]], dtype=torch.uint8),
        "proj.bias": torch.ones(2),
    }
    # ModelSlim also tags floating biases with the projection's scheme.
    path = _checkpoint(tmp_path / "vae", tensors, {name: "W8A8_MXFP8" for name in tensors})
    weights = dict(OfflineComponentCheckpoint(path).iter_weights(dequantize=True))
    torch.testing.assert_close(weights["proj.weight"], torch.tensor([[0.5], [2.0]]).expand(2, 24))
    torch.testing.assert_close(weights["proj.bias"], tensors["proj.bias"])


@pytest.mark.parametrize("invalid", ["missing_weight", "unknown", "duplicate"])
def test_strict_transformer_rejects_incomplete_or_incompatible_weights(invalid):
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel

    model = object.__new__(MiniMaxH3DiTModel)
    torch.nn.Module.__init__(model)
    model._strict_offline_loading = True
    model.proj = torch.nn.Module()
    model.proj.weight = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)
    model.proj.bias = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    weights = [("proj.weight", torch.ones(2, 2)), ("proj.bias", torch.ones(2))]
    if invalid == "missing_weight":
        weights.pop()
        error = "incomplete"
    elif invalid == "unknown":
        weights.append(("unknown.weight", torch.ones(2, 2)))
        error = "no model parameter"
    else:
        weights.append(weights[0])
        error = "duplicate"
    with pytest.raises(ValueError, match=error):
        model.load_weights(weights)
