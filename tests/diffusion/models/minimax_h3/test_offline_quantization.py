# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for ModelSlim offline INT8 and delivered MX checkpoints."""

import json

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.models.minimax_h3.offline_quantization import (
    OfflineComponentCheckpoint,
    load_offline_vae_weights,
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


def _int8(path, *, shard_scales=False):
    tensors = {
        "proj.weight": torch.tensor([[1, -2], [3, 4]], dtype=torch.int8),
        "proj.weight_scale": torch.tensor([[0.5], [0.25]]),
        "proj.weight_offset": torch.zeros(2, 1),
        "proj.bias": torch.tensor([1.0, 2.0]),
    }
    desc: dict[str, object] = {name: "W8A8_DYNAMIC" if name != "proj.bias" else "FLOAT" for name in tensors}
    desc.update(version="1.0.0", metadata={}, optional={})
    return _checkpoint(path, tensors, desc, shard_scales=shard_scales)


def test_int8_stream_keeps_serialized_values_and_drops_only_zero_offset(tmp_path):
    checkpoint = OfflineComponentCheckpoint(_int8(tmp_path / "int8", shard_scales=True))
    weights = dict(checkpoint.iter_weights(dequantize=False))
    assert weights["proj.weight"].dtype == torch.int8
    assert weights["proj.weight"].tolist() == [[1, -2], [3, 4]]
    torch.testing.assert_close(weights["proj.weight_scale"], torch.tensor([[0.5], [0.25]]))
    assert set(weights) == {"proj.weight", "proj.weight_scale", "proj.bias"}


def test_int8_reference_reconstruction(tmp_path):
    checkpoint = OfflineComponentCheckpoint(_int8(tmp_path / "int8"))
    weights = dict(checkpoint.iter_weights(dequantize=True))
    torch.testing.assert_close(weights["proj.weight"], torch.tensor([[0.5, -1], [0.75, 1.0]]))


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


def test_native_auto_selects_mx_for_a5(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.minimax_h3.offline_quantization.resolve_npu_quant_family", lambda *_: "mx"
    )
    tensors = {
        "proj.weight": torch.ones(1, 32, dtype=torch.float8_e4m3fn),
        "proj.weight_scale": torch.full((1, 1), 127, dtype=torch.uint8),
    }
    path = _checkpoint(tmp_path / "mx", tensors, {name: "W8A8_MXFP8" for name in tensors})
    native, reconstruct = resolve_offline_components({"path": str(path)})
    assert native["transformer"].execution_mode == "native_mx" and reconstruct
    checkpoints, dequantize = resolve_offline_components({"path": str(path), "execution_mode": "dequantize"})
    assert dequantize
    assert checkpoints["transformer"].schemes == {"W8A8_MXFP8"}


def test_missing_dual_scale_is_rejected_before_loading(tmp_path):
    tensors = {
        "proj.weight": torch.full((1, 16), 0x22, dtype=torch.uint8),
        "proj.weight_scale": torch.full((1, 1), 127, dtype=torch.uint8),
    }
    path = _checkpoint(tmp_path / "mx4", tensors, {name: "W4A4_MXFP4_DUALSCALE" for name in tensors})
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


def test_native_mx_rejects_int8_delivery(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.minimax_h3.offline_quantization.resolve_npu_quant_family", lambda *_: "mx"
    )
    with pytest.raises(ValueError, match="requires MXFP8/MXFP4"):
        resolve_offline_components({"path": str(_int8(tmp_path / "int8")), "execution_mode": "native_mx"})


def test_nonzero_int8_offset_is_not_silently_discarded(tmp_path):
    path = _int8(tmp_path / "int8")
    from safetensors.torch import load_file

    # Release the read-only mapping before replacing its file on Windows.
    tensors = {name: tensor.clone() for name, tensor in load_file(str(path / "weights.safetensors")).items()}
    tensors["proj.weight_offset"][0] = 1
    save_file(tensors, str(path / "weights.safetensors"))
    checkpoint = OfflineComponentCheckpoint(path)
    with pytest.raises(ValueError, match="asymmetric"):
        dict(checkpoint.iter_weights(dequantize=False))


@pytest.mark.parametrize("scheme", ["W8A8", "W4A8", "W8A8_UNKNOWN"])
def test_unsupported_scheme_is_rejected(tmp_path, scheme):
    path = _checkpoint(tmp_path / "bad", {"proj.weight": torch.ones(2, 2)}, {"proj.weight": scheme})
    with pytest.raises(ValueError, match="unsupported offline scheme"):
        OfflineComponentCheckpoint(path)


def test_index_cannot_escape_checkpoint_directory(tmp_path):
    path = _int8(tmp_path / "int8")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"proj.weight": "../other.safetensors"}})
    )
    with pytest.raises(ValueError, match="invalid or missing"):
        OfflineComponentCheckpoint(path)


def test_vae_overlay_is_strict_and_uses_reconstructed_weights(tmp_path):
    checkpoint = OfflineComponentCheckpoint(_int8(tmp_path / "int8"))
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(2, 2)
    load_offline_vae_weights(model, checkpoint)
    torch.testing.assert_close(model.proj.weight, torch.tensor([[0.5, -1], [0.75, 1.0]]))
    model.register_buffer("required", torch.ones(1))
    with pytest.raises(ValueError, match="incomplete"):
        load_offline_vae_weights(model, checkpoint)


@pytest.mark.parametrize("shard_scales", [False, True])
def test_vae_overlay_accepts_only_exact_duplicate_quantizer_bias(tmp_path, shard_scales):
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(2, 2)
    tensors = {
        "proj.weight": torch.tensor([[1, -2], [3, 4]], dtype=torch.int8),
        "proj.weight_scale": torch.tensor([0.5, 0.25]),
        "proj.bias": torch.tensor([1.0, 2.0]),
        "proj.weight_quantizer.bias": torch.tensor([1.0, 2.0]),
    }
    desc = {name: "W8A8_DYNAMIC" if name in {"proj.weight", "proj.weight_scale"} else "FLOAT" for name in tensors}
    checkpoint = OfflineComponentCheckpoint(_checkpoint(tmp_path / "vae", tensors, desc, shard_scales=shard_scales))
    load_offline_vae_weights(model, checkpoint)
    torch.testing.assert_close(model.proj.bias, tensors["proj.bias"], rtol=0, atol=0)
    torch.testing.assert_close(model.proj.weight, torch.tensor([[0.5, -1.0], [0.75, 1.0]]))


@pytest.mark.parametrize("invalid", ["value", "dtype", "shape", "missing_canonical", "unknown_target"])
def test_vae_overlay_rejects_conflicting_or_orphan_quantizer_bias(tmp_path, invalid):
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(2, 2)
    tensors = {
        "proj.weight": torch.ones(2, 2, dtype=torch.int8),
        "proj.weight_scale": torch.ones(2),
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
    desc = {name: "W8A8_DYNAMIC" if name in {"proj.weight", "proj.weight_scale"} else "FLOAT" for name in tensors}
    checkpoint = OfflineComponentCheckpoint(_checkpoint(tmp_path / "vae", tensors, desc))
    with pytest.raises(ValueError, match="bias alias|does not match original architecture"):
        load_offline_vae_weights(model, checkpoint)


def test_vae_overlay_maps_legacy_weight_norm_names(tmp_path):
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(2, 2)
    model.conv = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(2, 2, 3))
    tensors = {
        "proj.weight": torch.ones(2, 2, dtype=torch.int8),
        "proj.weight_scale": torch.ones(2),
        "proj.bias": torch.zeros(2),
        "conv.weight_g": torch.full((2, 1, 1), 2.0),
        "conv.weight_v": torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
        "conv.bias": torch.zeros(2),
    }
    description = {name: "W8A8_DYNAMIC" if name.startswith("proj.weight") else "FLOAT" for name in tensors}
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


@pytest.mark.parametrize("dequantize", [False, True])
def test_nonpositive_int8_scale_is_rejected(tmp_path, dequantize):
    tensors = {"proj.weight": torch.ones(2, 4, dtype=torch.int8), "proj.weight_scale": torch.tensor([0.0, 1.0])}
    path = _checkpoint(tmp_path / "bad", tensors, {name: "W8A8_DYNAMIC" for name in tensors})
    with pytest.raises(ValueError, match="positive"):
        dict(OfflineComponentCheckpoint(path).iter_weights(dequantize=dequantize))


@pytest.mark.parametrize("invalid", ["missing_scale", "wrong_dtype", "unknown", "duplicate"])
def test_strict_transformer_rejects_incomplete_or_incompatible_weights(invalid):
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel

    model = object.__new__(MiniMaxH3DiTModel)
    torch.nn.Module.__init__(model)
    model._strict_offline_loading = True
    model.proj = torch.nn.Module()
    model.proj.weight = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.int8), requires_grad=False)
    model.proj.weight_scale = torch.nn.Parameter(torch.ones(2, 1), requires_grad=False)
    weights = [("proj.weight", torch.ones(2, 2, dtype=torch.int8)), ("proj.weight_scale", torch.ones(2, 1))]
    if invalid == "missing_scale":
        weights.pop()
        error = "incomplete"
    elif invalid == "wrong_dtype":
        weights[0] = ("proj.weight", torch.ones(2, 2))
        error = "requires int8"
    elif invalid == "unknown":
        weights.append(("unknown.weight", torch.ones(2, 2)))
        error = "no model parameter"
    else:
        weights.append(weights[0])
        error = "duplicate"
    with pytest.raises(ValueError, match=error):
        model.load_weights(weights)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("diffusers", [False, True])
@pytest.mark.parametrize("native_bits", [None, 4, 8])
def test_serialized_int8_qkv_and_mlp_scales_follow_real_tp_loaders(monkeypatch, rank, diffusers, native_bits):
    from types import SimpleNamespace

    import vllm.distributed.parallel_state as parallel_state
    import vllm.model_executor.layers.linear as linear

    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTArchConfig,
        MiniMaxH3DiTModel,
    )
    from vllm_omni.quantization import int8_config

    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: SimpleNamespace(rank_in_group=rank, world_size=2))
    monkeypatch.setattr(int8_config.current_omni_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(int8_config.current_omni_platform, "is_npu", lambda: True)
    config = int8_config.DiffusionInt8Config(is_checkpoint_int8_serialized=True)
    if native_bits:
        from vllm_omni.diffusion.models.minimax_h3.native_int import H3NativeIntConfig

        config = H3NativeIntConfig({"blocks.0.attn.qkv_proj": native_bits, "blocks.0.mlp.fc1": native_bits})
    model = object.__new__(MiniMaxH3DiTModel)
    torch.nn.Module.__init__(model)
    model.arch = MiniMaxH3DiTArchConfig(num_attention_heads=2, attention_head_dim=2, rope_inv_freq_len=1)
    model._diffusers_weights = diffusers
    model._strict_offline_loading = True
    model._serialized_weight_map = {}
    model._rope_theta = 10000.0
    model.blocks = torch.nn.ModuleList([torch.nn.Module()])
    model.blocks[0].attn = torch.nn.Module()
    model.blocks[0].attn.qkv_proj = linear.QKVParallelLinear(
        4, 2, 2, bias=False, params_dtype=torch.bfloat16, quant_config=config, prefix="blocks.0.attn.qkv_proj"
    )
    model.blocks[0].mlp = torch.nn.Module()
    model.blocks[0].mlp.fc1 = linear.MergedColumnParallelLinear(
        4, [8, 8], bias=False, params_dtype=torch.bfloat16, quant_config=config, prefix="blocks.0.mlp.fc1"
    )
    model.rope = torch.nn.Module()
    model.rope.register_buffer("inv_freq", torch.zeros(1))
    qkv = torch.arange(48, dtype=torch.int8).reshape(12, 4)
    if native_bits == 4:
        qkv = qkv % 16 - 8
    qkv_scale = torch.arange(1, 13, dtype=torch.float32).reshape(12, 1)
    fc1 = torch.arange(64, dtype=torch.int8).reshape(16, 4)
    if native_bits == 4:
        fc1 = fc1 % 16 - 8
    fc1_scale = torch.arange(1, 17, dtype=torch.float32)  # Also accept [N].
    if diffusers:
        weights = []
        for i, slot in enumerate(("q", "k", "v")):
            rows = [i * 2, i * 2 + 1, i * 2 + 6, i * 2 + 7]
            prefix = f"transformer_blocks.0.attn.to_{slot}"
            weights.extend([(prefix + ".weight", qkv[rows]), (prefix + ".weight_scale", qkv_scale[rows])])
        prefix = "transformer_blocks.0.ff.net.0.proj"
        weights.extend(
            [
                (prefix + ".weight", torch.cat(fc1.chunk(2)[::-1])),
                (prefix + ".weight_scale", torch.cat(fc1_scale.chunk(2)[::-1])),
            ]
        )
    else:
        weights = [
            ("blocks.0.attn.qkv_proj.weight", qkv),
            ("blocks.0.attn.qkv_proj.weight_scale", qkv_scale),
            ("blocks.0.mlp.fc1.weight", fc1),
            ("blocks.0.mlp.fc1.weight_scale", fc1_scale),
            ("rope.inv_freq", torch.ones(1)),
        ]
    model.load_weights(weights)
    qkv_layer = model.blocks[0].attn.qkv_proj
    fc1_layer = model.blocks[0].mlp.fc1
    rows = list(range(rank * 6, rank * 6 + 6))
    torch.testing.assert_close(qkv_layer.weight, qkv[rows])
    torch.testing.assert_close(qkv_layer.weight_scale, qkv_scale[rows])
    rows = list(range(rank * 4, rank * 4 + 4)) + list(range(8 + rank * 4, 12 + rank * 4))
    torch.testing.assert_close(fc1_layer.weight, fc1[rows])
    torch.testing.assert_close(fc1_layer.weight_scale, fc1_scale[rows, None])

    if native_bits:
        from vllm_omni.diffusion.models.minimax_h3.integer_utils import unpack_int4

        for layer in (qkv_layer, fc1_layer):
            original_weight, original_scale = layer.weight.clone(), layer.weight_scale.clone()
            layer.quant_method.process_weights_after_loading(layer)
            actual = unpack_int4(layer.weight) if native_bits == 4 else layer.weight
            torch.testing.assert_close(actual[: original_weight.shape[0], :4], original_weight, rtol=0, atol=0)
            torch.testing.assert_close(
                layer.weight_scale[: original_weight.shape[0]], original_scale.flatten(), rtol=0, atol=0
            )
            before = layer.weight.clone()
            layer.quant_method.process_weights_after_loading(layer)
            layer.bfloat16()
            torch.testing.assert_close(layer.weight, before, rtol=0, atol=0)
            assert layer.weight_scale.dtype == torch.float32
            torch.testing.assert_close(
                layer.weight_scale[: original_weight.shape[0]], original_scale.flatten(), rtol=0, atol=0
            )
