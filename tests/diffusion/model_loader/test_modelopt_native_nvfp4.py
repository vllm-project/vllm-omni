# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L1 tests for the ModelOpt-native NVFP4 (weight-only, blockwise) adapter.

The checkpoint format under test (`nvfp4_blockwise_mixed_v1`): per
target module three tensors — ``<m>.weight_packed`` (uint8 packed E2M1),
``<m>.weight_block_scale`` (float8_e4m3fn, ``[rows, ceil(cols/16)]``) and
``<m>.weight_global_scale`` (float32, ``[1]``) — described by a
``transformer/nvfp4_blockwise_mixed_v1.json`` sidecar. Non-target tensors stay
BF16. Unlike the FP8-native adapter, this adapter NEVER dequantizes a target:
it renames on-disk names to vLLM's W4A16 param names and passes bytes through so
the target modules remain FP4-resident.
"""

import json
import math
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.model_loader.checkpoint_adapters import (
    get_checkpoint_adapter,
)
from vllm_omni.diffusion.model_loader.checkpoint_adapters import (
    modelopt_native_nvfp4 as mn4,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

BLOCK = 16
# Two real target names (one UND mlp, one GEN mlp_moe_gen) + shapes small but
# 16-divisible so packed/scale grids are exact.
_TARGETS = {
    "layers.0.mlp.gate_proj.weight": (32, 64),          # out=32, in=64
    "layers.0.mlp_moe_gen.down_proj.weight": (64, 32),  # out=64, in=32
}


# --- fixtures ---------------------------------------------------------------

def _packed_width(cols: int) -> int:
    return (cols + 1) // 2


def _scale_grid(shape):
    rows, cols = shape
    return [rows, math.ceil(cols / BLOCK)]


def _authoritative_sidecar():
    """Sidecar dict mirroring the real deliverable's structure."""
    tensors = {}
    for name, (out, inn) in _TARGETS.items():
        tensors[name] = {
            "weight_shape": [out, inn],
            "packed_shape": [out, _packed_width(inn)],
            "block_scale_shape": _scale_grid((out, inn)),
        }
    return {
        "recipe": "nvfp4_blockwise_mixed_v1",
        "block_size": BLOCK,
        "expected_quantized_count": len(_TARGETS),
        "target_patterns": [r"^layers\.\d+\.mlp\.", r"^layers\.\d+\.mlp_moe_gen\."],
        "forbidden_patterns": [
            "embed_tokens", "lm_head", "self_attn", "norm", "time_embedder",
            "proj_in", "proj_out", "audio_", "action_", "blocks.",
        ],
        "scale_encoding": {
            "block_along": "last_dim",
            "block_scale_dtype": "float8_e4m3fn",
            "block_size": BLOCK,
            "global_scale_dtype": "float32",
            "packed_dtype": "uint8",
        },
        "tensors": tensors,
    }


def _awq_sidecar():
    """An old AWQ-style sidecar that must be rejected (wrong recipe)."""
    s = _authoritative_sidecar()
    s["recipe"] = "nvfp4_awq"
    return s


def _target_triplet(out, inn, *, scale_val=1.0, nan=False):
    packed = torch.randint(0, 256, (out, _packed_width(inn)), dtype=torch.uint8)
    grid = _scale_grid((out, inn))
    block = torch.full(tuple(grid), scale_val, dtype=torch.float8_e4m3fn)
    if nan:
        block[0, 0] = torch.tensor(float("nan"), dtype=torch.float32).to(torch.float8_e4m3fn)
    glob = torch.full((1,), 0.5, dtype=torch.float32)
    return packed, block, glob


def _tensors(nan=False):
    out_t = {}
    for name, (out, inn) in _TARGETS.items():
        base = name[: -len(".weight")]
        packed, block, glob = _target_triplet(out, inn, nan=nan and base.endswith("gate_proj"))
        out_t[base + ".weight_packed"] = packed
        out_t[base + ".weight_block_scale"] = block
        out_t[base + ".weight_global_scale"] = glob
    # one non-target BF16 passthrough
    out_t["layers.0.self_attn.to_q.weight"] = torch.ones(8, 8, dtype=torch.bfloat16)
    return out_t


def _stream(prefix="transformer.", nan=False):
    return [(f"{prefix}{n}", t) for n, t in _tensors(nan=nan).items()]


def _write_model_dir(tmp_path, sidecar, tensors=None):
    from safetensors.torch import save_file

    root = tmp_path / "model"
    (root / "transformer").mkdir(parents=True)
    if sidecar is not None:
        (root / "transformer" / "nvfp4_blockwise_mixed_v1.json").write_text(json.dumps(sidecar))
    if tensors is None:
        tensors = _tensors()
    save_file(tensors, str(root / "transformer" / "model.safetensors"))
    return str(root)


def _source(model_or_path="unused"):
    return SimpleNamespace(model_or_path=model_or_path, subfolder=None, prefix="transformer.")


# --- parse_nvfp4_spec --------------------------------------------------------

def test_parse_spec_accepts_authoritative_sidecar():
    spec = mn4.parse_nvfp4_spec(_authoritative_sidecar())
    assert spec.recipe == "nvfp4_blockwise_mixed_v1"
    assert spec.block_size == 16
    assert spec.expected_count == 2
    assert "layers.0.mlp.gate_proj.weight" in spec.tensors


def test_parse_spec_rejects_wrong_recipe():
    with pytest.raises(mn4.CheckpointIntegrityError) as exc:
        mn4.parse_nvfp4_spec(_awq_sidecar())
    assert "nvfp4_blockwise_mixed_v1" in str(exc.value)


def test_parse_spec_rejects_bad_scale_encoding():
    bad = _authoritative_sidecar()
    bad["scale_encoding"]["block_scale_dtype"] = "bfloat16"
    with pytest.raises(mn4.CheckpointIntegrityError):
        mn4.parse_nvfp4_spec(bad)


# --- pure name/shape calculations -------------------------------------------

def test_remap_name_roundtrip():
    assert mn4.remap_name("x.mlp.gate_proj.weight_packed") == "x.mlp.gate_proj.weight"
    assert mn4.remap_name("x.mlp.gate_proj.weight_block_scale") == "x.mlp.gate_proj.weight_scale"
    assert mn4.remap_name("x.mlp.gate_proj.weight_global_scale") == "x.mlp.gate_proj.weight_scale_2"
    assert mn4.remap_name("x.self_attn.to_q.weight") == "x.self_attn.to_q.weight"  # untouched


def test_matches_target_and_forbidden():
    spec = mn4.parse_nvfp4_spec(_authoritative_sidecar())
    assert mn4.matches_any("layers.0.mlp.gate_proj", spec.target_patterns)
    assert mn4.matches_any("layers.3.mlp_moe_gen.up_proj", spec.target_patterns)
    assert not mn4.matches_any("layers.0.self_attn.to_q", spec.target_patterns)
    assert mn4.matches_any("layers.0.self_attn.to_q", spec.forbidden_patterns)


def test_expected_packed_width_and_grid():
    assert mn4.expected_packed_width(64) == 32
    assert mn4.expected_packed_width(63) == 32  # ceil
    assert mn4.expected_scale_grid((32, 64), 16) == (32, 4)


# --- adapter detect ---------------------------------------------------------

def test_detect_engages_on_sidecar(tmp_path):
    model_dir = _write_model_dir(tmp_path, _authoritative_sidecar())
    adapter = mn4.ModelOptNativeNvfp4CheckpointAdapter.detect(_source(model_dir))
    assert isinstance(adapter, mn4.ModelOptNativeNvfp4CheckpointAdapter)


def test_detect_returns_none_without_sidecar(tmp_path):
    model_dir = _write_model_dir(tmp_path, None)
    adapter = mn4.ModelOptNativeNvfp4CheckpointAdapter.detect(_source(model_dir))
    assert adapter is None


def test_detect_raises_on_malformed_sidecar(tmp_path):
    model_dir = _write_model_dir(tmp_path, _awq_sidecar())
    with pytest.raises(mn4.CheckpointIntegrityError):
        mn4.ModelOptNativeNvfp4CheckpointAdapter.detect(_source(model_dir))


def test_registry_selects_nvfp4_native_before_generic(tmp_path):
    model_dir = _write_model_dir(tmp_path, _authoritative_sidecar())
    model = torch.nn.Linear(2, 2)
    adapter = get_checkpoint_adapter(model, _source(model_dir), quant_config=None, use_safetensors=True)
    assert isinstance(adapter, mn4.ModelOptNativeNvfp4CheckpointAdapter)


# --- adapt: rename + passthrough, no dequant --------------------------------

def _run_adapt(stream, sidecar=None):
    spec = mn4.parse_nvfp4_spec(sidecar or _authoritative_sidecar())
    adapter = mn4.ModelOptNativeNvfp4CheckpointAdapter(spec=spec, source_prefix="transformer.")
    return list(adapter.adapt(stream))


def test_adapt_renames_and_passes_through_unchanged():
    out = dict(_run_adapt(_stream()))
    # packed weight renamed to .weight, still uint8, byte-identical
    key = "transformer.layers.0.mlp.gate_proj.weight"
    assert key in out
    assert out[key].dtype == torch.uint8
    assert "transformer.layers.0.mlp.gate_proj.weight_scale" in out
    assert "transformer.layers.0.mlp.gate_proj.weight_scale_2" in out
    # no *.weight_packed leaks through
    assert not any(k.endswith(".weight_packed") for k in out)


def test_adapt_never_emits_floating_target_weight():
    out = _run_adapt(_stream())
    for name, tensor in out:
        if name.endswith(".weight") and ".mlp" in name:
            assert tensor.dtype == torch.uint8, f"{name} must stay packed uint8, got {tensor.dtype}"


def test_adapt_passes_non_target_bf16_through():
    out = dict(_run_adapt(_stream()))
    key = "transformer.layers.0.self_attn.to_q.weight"
    assert out[key].dtype == torch.bfloat16


def test_adapt_aborts_on_nan_scale():
    with pytest.raises(mn4.CheckpointIntegrityError) as exc:
        _run_adapt(_stream(nan=True))
    assert "nan" in str(exc.value).lower() or "inf" in str(exc.value).lower()


def test_adapt_rejects_forbidden_quantized(tmp_path):
    # Inject a packed tensor on a forbidden module.
    bad = _tensors()
    bad["layers.0.self_attn.to_q.weight_packed"] = torch.zeros(8, 4, dtype=torch.uint8)
    bad["layers.0.self_attn.to_q.weight_block_scale"] = torch.zeros(8, 1, dtype=torch.float8_e4m3fn)
    bad["layers.0.self_attn.to_q.weight_global_scale"] = torch.zeros(1, dtype=torch.float32)
    stream = [(f"transformer.{n}", t) for n, t in bad.items()]
    with pytest.raises(mn4.CheckpointIntegrityError):
        _run_adapt(stream)


def test_adapt_rejects_count_mismatch():
    # Drop one target's triplet -> observed count < declared.
    partial = {k: v for k, v in _tensors().items() if "mlp_moe_gen" not in k}
    stream = [(f"transformer.{n}", t) for n, t in partial.items()]
    with pytest.raises(mn4.CheckpointIntegrityError):
        _run_adapt(stream)


def test_adapt_rejects_wrong_packed_width():
    bad = _tensors()
    # gate_proj in=64 -> packed width should be 32; give 16.
    bad["layers.0.mlp.gate_proj.weight_packed"] = torch.zeros(32, 16, dtype=torch.uint8)
    stream = [(f"transformer.{n}", t) for n, t in bad.items()]
    with pytest.raises(mn4.CheckpointIntegrityError):
        _run_adapt(stream)


# --- CLI probe --------------------------------------------------------------

def test_cli_probe_passes_on_good_dir(tmp_path, capsys):
    model_dir = _write_model_dir(tmp_path, _authoritative_sidecar())
    rc = mn4.main([model_dir])
    assert rc == 0


def test_cli_probe_fails_on_awq_dir(tmp_path):
    model_dir = _write_model_dir(tmp_path, _awq_sidecar())
    rc = mn4.main([model_dir])
    assert rc == 1


def test_cli_probe_usage_error():
    assert mn4.main([]) == 2


# --- additional integrity scenarios (spec-mandated) --------------------------

def test_adapt_rejects_wrong_block_scale_grid():
    bad = _tensors()
    # gate_proj in=64 -> grid should be (32, 4); give (32, 3).
    bad["layers.0.mlp.gate_proj.weight_block_scale"] = torch.zeros(32, 3, dtype=torch.float8_e4m3fn)
    stream = [(f"transformer.{n}", t) for n, t in bad.items()]
    with pytest.raises(mn4.CheckpointIntegrityError) as exc:
        _run_adapt(stream)
    assert "grid" in str(exc.value).lower()


def test_adapt_rejects_missing_block_scale_companion():
    bad = _tensors()
    del bad["layers.0.mlp.gate_proj.weight_block_scale"]  # keep packed + global, drop block scale
    stream = [(f"transformer.{n}", t) for n, t in bad.items()]
    with pytest.raises(mn4.CheckpointIntegrityError) as exc:
        _run_adapt(stream)
    assert "weight_block_scale" in str(exc.value)


def test_adapt_rejects_missing_global_scale_companion():
    bad = _tensors()
    del bad["layers.0.mlp.gate_proj.weight_global_scale"]
    stream = [(f"transformer.{n}", t) for n, t in bad.items()]
    with pytest.raises(mn4.CheckpointIntegrityError) as exc:
        _run_adapt(stream)
    assert "weight_global_scale" in str(exc.value)


def test_adapt_rejects_packed_outside_target_patterns():
    # A packed module matching NEITHER target NOR forbidden patterns, absent
    # from the manifest -> must be rejected (unexpected quantized module).
    bad = _tensors()
    bad["layers.0.router.gate.weight_packed"] = torch.zeros(16, 8, dtype=torch.uint8)
    bad["layers.0.router.gate.weight_block_scale"] = torch.zeros(16, 1, dtype=torch.float8_e4m3fn)
    bad["layers.0.router.gate.weight_global_scale"] = torch.zeros(1, dtype=torch.float32)
    stream = [(f"transformer.{n}", t) for n, t in bad.items()]
    with pytest.raises(mn4.CheckpointIntegrityError) as exc:
        _run_adapt(stream)
    assert "outside target patterns" in str(exc.value) or "not declared" in str(exc.value)


def test_adapt_aborts_on_inf_global_scale():
    bad = _tensors()
    bad["layers.0.mlp.gate_proj.weight_global_scale"] = torch.tensor([float("inf")], dtype=torch.float32)
    stream = [(f"transformer.{n}", t) for n, t in bad.items()]
    with pytest.raises(mn4.CheckpointIntegrityError) as exc:
        _run_adapt(stream)
    assert "inf" in str(exc.value).lower() or "nan" in str(exc.value).lower()
