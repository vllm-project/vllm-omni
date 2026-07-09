# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L1 tests for the ModelOpt-native (state-dict export) FP8-blockwise adapter.

The checkpoint format under test: FP8 (e4m3) weight codes at diffusers names
plus ``<module>.weight_quantizer._scale`` (2D block grid) and ``._amax`` (4D)
tensors, described by a ``quantization_config.json`` sidecar at the model root
(recipe ``fp8_blockwise_mixed``).
"""

import json
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.model_loader.checkpoint_adapters import (
    get_checkpoint_adapter,
)
from vllm_omni.diffusion.model_loader.checkpoint_adapters import (
    modelopt_native as mn,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


# --- fixtures ---------------------------------------------------------------


def _authoritative_sidecar(n_quantized=2, n_scale=4):
    """Sidecar dict mirroring the real deliverable's quantization_config.json."""
    return {
        "recipe": "fp8_blockwise_mixed",
        "weight_only": True,
        "quant_lmhead": True,
        "mixed_precision": {
            "quantized": ["mlp.*", "mlp_moe_gen.*", "lm_head"],
            "bf16_kept": ["self_attn.*"],
            "n_quantized": n_quantized,
        },
        "scale_layout": {
            "granularity": "blockwise-128x128",
            "block_sizes": {"rows": 128, "cols": 128},
            "n_quantized_weight": n_quantized,
            "n_scale": n_scale,
        },
    }


def _stale_sidecar():
    """The stale per-tensor export's sidecar (must be rejected loudly)."""
    return {
        "recipe": "fp8",
        "weight_only": True,
        "quant_lmhead": True,
        "exclusions": ["embed_tokens", "*norm*"],
        "scale_layout": {
            "weight_scale_suffixes": ["_amax", "_scale"],
            "n_scale": 1010,
            "granularity": "per-tensor",
        },
    }


def _fp8(t):
    return t.to(torch.float8_e4m3fn)


def _quantized_pair(_name, out_blocks=2, in_blocks=2, block=128, scale_val=0.5):
    """(weight, scale2d, amax4d) synthetic tensors; *_name* documents the call site."""
    shape = (out_blocks * block, in_blocks * block)
    weight = _fp8(torch.full(shape, 2.0))
    scale = torch.full((out_blocks, in_blocks), scale_val, dtype=torch.bfloat16)
    amax = (scale * 448.0).reshape(out_blocks, 1, in_blocks, 1)
    return weight, scale, amax


def _stream(*, prefix="transformer.", order="scale_first"):
    """Synthetic checkpoint stream: 2 quantized modules + 1 bf16 passthrough."""
    w1, s1, a1 = _quantized_pair("layers.0.mlp.gate_proj.weight")
    w2, s2, a2 = _quantized_pair("layers.0.mlp_moe_gen.down_proj.weight", scale_val=0.25)
    passthrough = torch.ones(4, 4, dtype=torch.bfloat16)
    items = {
        f"{prefix}layers.0.mlp.gate_proj.weight": w1,
        f"{prefix}layers.0.mlp.gate_proj.weight_quantizer._scale": s1,
        f"{prefix}layers.0.mlp.gate_proj.weight_quantizer._amax": a1,
        f"{prefix}layers.0.mlp_moe_gen.down_proj.weight": w2,
        f"{prefix}layers.0.mlp_moe_gen.down_proj.weight_quantizer._scale": s2,
        f"{prefix}layers.0.mlp_moe_gen.down_proj.weight_quantizer._amax": a2,
        f"{prefix}layers.0.self_attn.to_q.weight": passthrough,
    }
    names = list(items)
    if order == "weight_first":
        names = sorted(names, key=lambda n: ("_quantizer." in n, n))
    else:
        names = sorted(names, key=lambda n: ("_quantizer." not in n, n))
    return [(n, items[n]) for n in names]


def _write_model_dir(tmp_path, sidecar, tensors=None):
    """Model dir with sidecar + transformer/diffusion_pytorch_model.safetensors."""
    from safetensors.torch import save_file

    root = tmp_path / "model"
    (root / "transformer").mkdir(parents=True)
    if sidecar is not None:
        (root / "quantization_config.json").write_text(json.dumps(sidecar))
    if tensors is None:
        tensors = {name: t for name, t in _stream(prefix="")}
    save_file(tensors, str(root / "transformer" / "diffusion_pytorch_model.safetensors"))
    return str(root)


def _source(model_or_path="unused"):
    return SimpleNamespace(model_or_path=model_or_path, subfolder=None, prefix="transformer.")


class _TinyModel(nn.Module):
    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False, dtype=dtype)


# --- parse_quant_spec --------------------------------------------------------


def test_parse_spec_accepts_authoritative_sidecar():
    spec = mn.parse_quant_spec(_authoritative_sidecar())
    assert spec.n_quantized == 2
    assert spec.n_scale == 4
    assert spec.block_rows == 128 and spec.block_cols == 128
    assert "mlp.*" in spec.quantized_patterns
    assert "self_attn.*" in spec.kept_patterns


def test_parse_spec_rejects_stale_sidecar_naming_mismatches():
    with pytest.raises(mn.CheckpointIntegrityError) as exc:
        mn.parse_quant_spec(_stale_sidecar())
    msg = str(exc.value)
    assert "fp8_blockwise_mixed" in msg  # expected recipe named
    assert "per-tensor" in msg or "granularity" in msg


def test_parse_spec_rejects_inconsistent_scale_count():
    bad = _authoritative_sidecar(n_quantized=2, n_scale=3)  # must be 2*n
    with pytest.raises(mn.CheckpointIntegrityError):
        mn.parse_quant_spec(bad)


# --- name/dtype calculations -------------------------------------------------


def test_scale_name_roundtrip():
    w = "layers.0.mlp.gate_proj.weight"
    s = mn.scale_name_for(w)
    assert s == "layers.0.mlp.gate_proj.weight_quantizer._scale"
    assert mn.weight_name_for(s) == w
    assert mn.weight_name_for("not_a_scale") is None


def test_classify_tensor_kinds():
    assert mn.classify_name("x.weight_quantizer._scale") is mn.TensorKind.QUANTIZER_SCALE
    assert mn.classify_name("x.weight_quantizer._amax") is mn.TensorKind.QUANTIZER_AMAX
    assert mn.classify_name("x.weight") is mn.TensorKind.OTHER
    assert mn.is_fp8_dtype(torch.float8_e4m3fn)
    assert not mn.is_fp8_dtype(torch.bfloat16)
    assert mn.is_fp8_dtype_str("F8_E4M3")
    assert not mn.is_fp8_dtype_str("BF16")


def test_pattern_matching_with_and_without_prefix():
    spec = mn.parse_quant_spec(_authoritative_sidecar())
    assert mn.matches_any("layers.0.mlp.gate_proj", spec.quantized_patterns)
    assert mn.matches_any("layers.3.mlp_moe_gen.up_proj", spec.quantized_patterns)
    assert mn.matches_any("lm_head", spec.quantized_patterns)
    assert not mn.matches_any("layers.0.self_attn.to_q", spec.quantized_patterns)
    assert mn.matches_any("layers.0.self_attn.to_q", spec.kept_patterns)


# --- scale expansion & dequant ------------------------------------------------


def test_expand_block_scale_broadcasts_per_block():
    scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    full = mn.expand_block_scale(scale, (4, 4), block=(2, 2))
    assert full.shape == (4, 4)
    assert full[0, 0] == 1.0 and full[0, 3] == 2.0
    assert full[3, 0] == 3.0 and full[3, 3] == 4.0


def test_expand_block_scale_rejects_grid_mismatch():
    scale = torch.ones(2, 2)
    with pytest.raises(mn.CheckpointIntegrityError):
        mn.expand_block_scale(scale, (256, 384), block=(128, 128))  # grid 2x3


def test_dequantize_weight_multiplies_codes_by_block_scale():
    weight, scale, _ = _quantized_pair("m.weight", scale_val=0.5)
    out = mn.dequantize_weight(weight, scale, torch.bfloat16, block=(128, 128))
    assert out.dtype == torch.bfloat16
    assert out.shape == weight.shape
    assert torch.allclose(out.float(), torch.full_like(out.float(), 1.0))


def test_dequantize_applies_scale_at_full_precision():
    # Pins the exact dequant value for a non-trivial (bf16-inexact) scale:
    # 36 (exact in e4m3) x bf16(0.1) = 3.609375. A regression that mishandled
    # the scale (e.g. used _amax instead of _scale, or skipped the fp32 upcast
    # on a platform where bf16 mul rounds per-op) would miss this value.
    # NOTE: on CPU the fp32 vs bf16 intermediate cannot be distinguished by
    # output (e4m3 is exactly representable in bf16 and torch CPU bf16-mul
    # accumulates in fp32); the .to(torch.float32) cast is for GPU parity, so
    # this test pins the numeric contract rather than the intermediate dtype.
    weight = _fp8(torch.full((1, 1), 36.0))
    scale = torch.full((1, 1), 0.1, dtype=torch.bfloat16)
    out = mn.dequantize_weight(weight, scale, torch.bfloat16, block=(1, 1))
    fp32_ref = (weight.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, fp32_ref)
    assert abs(out.item() - 3.609375) < 1e-6


def test_dequantize_reconstructs_random_weight_within_tolerance():
    # Real e4m3 blockwise round-trip (spec: "reconstructs the base weight",
    # median rel err < 5%). Uses a NON-128-divisible, asymmetric shape so a
    # wrong block boundary or transposed grid fails — locks Correctness-F1
    # (declared block must be honored, not inferred from shapes).
    torch.manual_seed(0)
    rows, cols, block = 200, 384, 128  # 200 not divisible by 128
    ob, ib = (rows + block - 1) // block, (cols + block - 1) // block  # 2, 3
    ref = torch.randn(rows, cols, dtype=torch.float32) * 0.05
    # per-block amax -> scale = amax/448; quantize codes = round-to-e4m3(w/scale)
    scale = torch.zeros(ob, ib, dtype=torch.float32)
    codes = torch.zeros(rows, cols, dtype=torch.float32)
    for i in range(ob):
        for j in range(ib):
            r0, r1 = i * block, min((i + 1) * block, rows)
            c0, c1 = j * block, min((j + 1) * block, cols)
            blk = ref[r0:r1, c0:c1]
            s = blk.abs().max().item() / 448.0 or 1e-8
            scale[i, j] = s
            codes[r0:r1, c0:c1] = (blk / s).to(torch.float8_e4m3fn).to(torch.float32)
    w8 = codes.to(torch.float8_e4m3fn)
    out = mn.dequantize_weight(w8, scale.to(torch.bfloat16), torch.bfloat16, block=(block, block))
    assert out.shape == (rows, cols)
    rel = (out.float() - ref).abs() / (ref.abs() + 1e-6)
    assert rel.median().item() < 0.05, rel.median().item()
    # A wrong (inferred-from-shape) block would misplace the row-128 boundary:
    wrong = mn.dequantize_weight(
        w8, scale.to(torch.bfloat16), torch.bfloat16, block=(math.ceil(rows / ob), math.ceil(cols / ib))
    )
    assert not torch.equal(out, wrong)


# --- unified verification (shared by adapter and CLI probe) -------------------


def _infos_from(items, prefix="transformer."):
    return [mn.TensorInfo(name[len(prefix) :], mn.is_fp8_dtype(t.dtype), tuple(t.shape)) for name, t in items]


def test_verify_observations_happy_path_is_clean():
    spec = mn.parse_quant_spec(_authoritative_sidecar())
    assert mn.verify_observations(_infos_from(_stream()), spec) == []


def test_verify_observations_flags_quantized_excluded_module():
    spec = mn.parse_quant_spec(_authoritative_sidecar())
    infos = _infos_from(_stream())
    infos.append(mn.TensorInfo("layers.0.self_attn.to_q.weight", True, (256, 256)))
    violations = mn.verify_observations(infos, spec)
    assert any("self_attn" in v for v in violations)


def test_verify_observations_flags_count_drift_and_missing_scale():
    spec = mn.parse_quant_spec(_authoritative_sidecar(n_quantized=3, n_scale=6))
    violations = mn.verify_observations(_infos_from(_stream()), spec)
    assert any("3" in v and "2" in v for v in violations)  # declared vs observed

    spec2 = mn.parse_quant_spec(_authoritative_sidecar())
    infos = [i for i in _infos_from(_stream()) if not i.name.endswith("._scale")]
    violations = mn.verify_observations(infos, spec2)
    assert any("_scale" in v or "scale" in v for v in violations)


def test_verify_observations_flags_scale_grid_mismatch():
    spec = mn.parse_quant_spec(_authoritative_sidecar())
    infos = [
        mn.TensorInfo(i.name, i.is_fp8, (3, 3)) if i.name.endswith("._scale") else i for i in _infos_from(_stream())
    ]
    violations = mn.verify_observations(infos, spec)
    assert any("grid" in v or "shape" in v for v in violations)


# --- adapter shell -------------------------------------------------------------


def _detect(tmp_path, sidecar, tensors=None, dtype=torch.bfloat16):
    root = _write_model_dir(tmp_path, sidecar, tensors)
    return mn.ModelOptNativeFp8CheckpointAdapter.detect(_source(root), target_dtype=dtype)


@pytest.mark.parametrize("order", ["scale_first", "weight_first"])
def test_adapt_dequantizes_and_consumes_quantizer_tensors(tmp_path, order):
    adapter = _detect(tmp_path, _authoritative_sidecar())
    adapted = dict(adapter.adapt(iter(_stream(order=order))))
    assert set(adapted) == {
        "transformer.layers.0.mlp.gate_proj.weight",
        "transformer.layers.0.mlp_moe_gen.down_proj.weight",
        "transformer.layers.0.self_attn.to_q.weight",
    }
    assert all(t.dtype == torch.bfloat16 for t in adapted.values())
    gate = adapted["transformer.layers.0.mlp.gate_proj.weight"]
    assert torch.allclose(gate.float(), torch.full_like(gate.float(), 1.0))
    moe = adapted["transformer.layers.0.mlp_moe_gen.down_proj.weight"]
    assert torch.allclose(moe.float(), torch.full_like(moe.float(), 0.5))


def test_adapt_aggregates_violations_into_one_error(tmp_path):
    adapter = _detect(tmp_path, _authoritative_sidecar())
    bad = [(n, t) for n, t in _stream() if not n.endswith("._scale")]  # both scales missing
    with pytest.raises(mn.CheckpointIntegrityError) as exc:
        list(adapter.adapt(iter(bad)))
    msg = str(exc.value)
    assert "gate_proj" in msg and "down_proj" in msg  # aggregated, not first-hit


def test_adapt_bounds_pending_buffer(tmp_path):
    n = mn.MAX_PENDING_FP8 + 1
    sidecar = _authoritative_sidecar(n_quantized=n, n_scale=2 * n)
    weights = []
    for i in range(n):
        w, _, _ = _quantized_pair(f"layers.{i}.mlp.up_proj.weight", 1, 1)
        weights.append((f"transformer.layers.{i}.mlp.up_proj.weight", w))
    adapter = _detect(tmp_path, sidecar)
    with pytest.raises(mn.CheckpointIntegrityError):
        list(adapter.adapt(iter(weights)))


def test_validate_source_sidecar_is_a_loader_preflight(tmp_path):
    # Called before weight-file discovery: no-op for plain and valid dirs,
    # loud for a mislabeled (stale) sidecar (FA ordering: the integrity
    # report must be the primary serve-path error, not a generic file error).
    plain = _write_model_dir(tmp_path / "plain", None)
    mn.ModelOptNativeFp8CheckpointAdapter.validate_source_sidecar(_source(plain))
    good = _write_model_dir(tmp_path / "good", _authoritative_sidecar())
    mn.ModelOptNativeFp8CheckpointAdapter.validate_source_sidecar(_source(good))
    stale = _write_model_dir(tmp_path / "stale", _stale_sidecar())
    with pytest.raises(mn.CheckpointIntegrityError):
        mn.ModelOptNativeFp8CheckpointAdapter.validate_source_sidecar(_source(stale))


def test_detect_paths(tmp_path):
    assert _detect(tmp_path / "plain", None) is None  # no sidecar -> not quantized
    with pytest.raises(mn.CheckpointIntegrityError):
        _detect(tmp_path / "stale", _stale_sidecar())  # stale -> loud
    adapter = _detect(tmp_path / "good", _authoritative_sidecar())
    assert adapter is not None


def test_get_checkpoint_adapter_engages_without_quant_config(tmp_path, monkeypatch):
    # Default route stays dequant-on-load; the resident W8A16 path is an explicit opt-in
    # covered in test_modelopt_native_fp8_w8a16.
    monkeypatch.delenv("VLLM_OMNI_FP8_BLOCKWISE_W8A16", raising=False)
    root = _write_model_dir(tmp_path, _authoritative_sidecar())
    model = _TinyModel()
    adapter = get_checkpoint_adapter(model=model, source=_source(root), quant_config=None, use_safetensors=True)
    assert isinstance(adapter, mn.ModelOptNativeFp8CheckpointAdapter)

    plain = _write_model_dir(tmp_path / "plain", None)
    assert get_checkpoint_adapter(model=model, source=_source(plain), quant_config=None, use_safetensors=True) is None


# --- fp8 guard -----------------------------------------------------------------


def test_assert_not_fp8_guard():
    mn.assert_not_fp8("x.weight", torch.bfloat16)  # no raise
    with pytest.raises(mn.CheckpointIntegrityError) as exc:
        mn.assert_not_fp8("x.weight", torch.float8_e4m3fn)
    assert "x.weight" in str(exc.value)


# --- CLI probe (header-only) ----------------------------------------------------


def test_cli_probe_exit_codes(tmp_path, capsys):
    good = _write_model_dir(tmp_path / "good", _authoritative_sidecar())
    assert mn.main([good]) == 0
    out = capsys.readouterr().out
    assert "OK" in out and "2" in out  # observed counts reported

    stale = _write_model_dir(tmp_path / "stale", _stale_sidecar())
    assert mn.main([stale]) != 0

    nosidecar = _write_model_dir(tmp_path / "none", None)
    assert mn.main([nosidecar]) != 0  # probe on unquantized dir: explicit failure

    truncated = _write_model_dir(
        tmp_path / "trunc",
        _authoritative_sidecar(),
        tensors={n: t for n, t in _stream(prefix="") if "gate_proj" not in n},
    )
    assert mn.main([truncated]) != 0
