# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L1 tests for the FP8-blockwise W8A16 resident checkpoint adapter.

The on-disk format mirrors the dequant adapter's FP8 ``<m>.weight`` plus bf16
2D ``._scale`` grid and ``._amax`` twin. The adapter keeps MLP targets
FP8-resident and dequantizes ``lm_head``.
"""

import json
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.model_loader.checkpoint_adapters import (
    ModelOptNativeFp8CheckpointAdapter,
    ModelOptNativeFp8W8A16CheckpointAdapter,
    get_checkpoint_adapter,
)
from vllm_omni.diffusion.model_loader.checkpoint_adapters import (
    modelopt_native_fp8_w8a16 as mw8,
)
from vllm_omni.diffusion.model_loader.checkpoint_adapters.modelopt_native import (
    CheckpointIntegrityError,
    dequantize_weight,
    is_fp8_dtype,
    parse_quant_spec,
)
from vllm_omni.quantization.fp8_blockwise_w8a16 import build_fp8_blockwise_w8a16_config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

BLOCK = 128
# out, in (both multiples of 128 so the block grid is exact)
_TARGETS = {
    "layers.0.mlp.gate_proj": (256, 128),
    "layers.0.mlp_moe_gen.down_proj": (128, 256),
}
_LM_HEAD = ("lm_head", (256, 128))


def _grid(out, inn):
    return ((out + BLOCK - 1) // BLOCK, (inn + BLOCK - 1) // BLOCK)


def _sidecar(n_quantized=3, n_scale=None):
    if n_scale is None:
        n_scale = 2 * n_quantized
    return {
        "recipe": "fp8_blockwise_mixed",
        "weight_only": True,
        "mixed_precision": {
            "quantized": ["mlp.*", "mlp_moe_gen.*", "lm_head"],
            "bf16_kept": ["self_attn.*"],
            "n_quantized": n_quantized,
        },
        "scale_layout": {
            "granularity": "blockwise-128x128",
            "block_sizes": {"rows": BLOCK, "cols": BLOCK},
            "n_scale": n_scale,
        },
    }


def _fp8(out, inn):
    return (torch.randn(out, inn) * 0.1).to(torch.float8_e4m3fn)


def _scale(out, inn, nan=False):
    g = _grid(out, inn)
    s = (torch.rand(*g) + 0.5).to(torch.bfloat16)
    if nan:
        s = s.to(torch.float32)
        s[0, 0] = float("nan")
        s = s.to(torch.bfloat16)
    return s


def _amax(out, inn):
    r, c = _grid(out, inn)
    return torch.rand(r, 1, c, 1).to(torch.bfloat16)


def _stream(prefix="transformer.", nan_target_scale=False, extra=None):
    items = []
    for name, (out, inn) in _TARGETS.items():
        items.append((f"{prefix}{name}.weight", _fp8(out, inn)))
        nan = nan_target_scale and name.endswith("gate_proj")
        items.append((f"{prefix}{name}.weight_quantizer._scale", _scale(out, inn, nan=nan)))
        items.append((f"{prefix}{name}.weight_quantizer._amax", _amax(out, inn)))
    name, (out, inn) = _LM_HEAD
    items.append((f"{prefix}{name}.weight", _fp8(out, inn)))
    items.append((f"{prefix}{name}.weight_quantizer._scale", _scale(out, inn)))
    items.append((f"{prefix}{name}.weight_quantizer._amax", _amax(out, inn)))
    # one non-target BF16 passthrough
    items.append((f"{prefix}layers.0.self_attn.to_q.weight", torch.ones(128, 128, dtype=torch.bfloat16)))
    if extra:
        items.extend(extra)
    return items


def _adapter(sidecar=None):
    spec = parse_quant_spec(sidecar or _sidecar())
    return mw8.ModelOptNativeFp8W8A16CheckpointAdapter(
        spec=spec, source_prefix="transformer.", target_dtype=torch.bfloat16
    )


# --- resident vs dequant routing --------------------------------------------


def test_adapt_targets_resident_lm_head_dequant_and_amax_dropped():
    out = dict(_adapter().adapt(_stream()))

    # MLP targets stay FP8-resident, scale renamed to the method's param name
    gp = "transformer.layers.0.mlp.gate_proj"
    assert out[f"{gp}.weight"].dtype == torch.float8_e4m3fn
    assert out[f"{gp}.weight"].element_size() == 1
    assert out[f"{gp}.weight_scale"].dtype == torch.bfloat16
    dp = "transformer.layers.0.mlp_moe_gen.down_proj"
    assert out[f"{dp}.weight"].dtype == torch.float8_e4m3fn
    assert f"{dp}.weight_scale" in out

    # lm_head dequantized to BF16 (not resident); its scale is consumed, not emitted
    assert out["transformer.lm_head.weight"].dtype == torch.bfloat16
    assert "transformer.lm_head.weight_scale" not in out

    # BF16 non-target passes through unchanged
    assert out["transformer.layers.0.self_attn.to_q.weight"].dtype == torch.bfloat16

    # no raw quantizer tensors leak into the output stream
    assert not any(".weight_quantizer." in k for k in out)


def test_adapt_count_mismatch_aborts():
    # sidecar declares 4 quantized but the stream carries 3 fp8 weights
    with pytest.raises(CheckpointIntegrityError, match="count mismatch"):
        dict(_adapter(_sidecar(n_quantized=4, n_scale=8)).adapt(_stream()))


def test_adapt_forbidden_module_quantized_aborts():
    # a bf16_kept module (self_attn) present as an FP8 weight -> integrity abort
    extra = [
        ("transformer.layers.0.self_attn.to_k.weight", _fp8(128, 128)),
        ("transformer.layers.0.self_attn.to_k.weight_quantizer._scale", _scale(128, 128)),
        ("transformer.layers.0.self_attn.to_k.weight_quantizer._amax", _amax(128, 128)),
    ]
    with pytest.raises(CheckpointIntegrityError):
        dict(_adapter(_sidecar(n_quantized=4, n_scale=8)).adapt(_stream(extra=extra)))


def test_adapt_nan_scale_aborts():
    with pytest.raises(CheckpointIntegrityError, match="NaN/Inf"):
        dict(_adapter().adapt(_stream(nan_target_scale=True)))


# --- detection + probe -------------------------------------------------------


def _write_model_dir(tmp_path, sidecar=None):
    from safetensors.torch import save_file

    root = tmp_path / "model"
    (root / "transformer").mkdir(parents=True)
    (root / "quantization_config.json").write_text(json.dumps(sidecar or _sidecar()))
    tensors = {k[len("transformer.") :]: v for k, v in _stream()}
    save_file(tensors, str(root / "transformer" / "model.safetensors"))
    return str(root)


def _source(model_or_path):
    return SimpleNamespace(model_or_path=model_or_path, subfolder="transformer", prefix="transformer.")


def _remote_source(repo_id, resolved_model_or_path):
    return SimpleNamespace(
        model_or_path=repo_id,
        resolved_model_or_path=resolved_model_or_path,
        subfolder="transformer",
        prefix="transformer.",
    )


def test_detect_engages_on_fp8_sidecar(tmp_path):
    root = _write_model_dir(tmp_path)
    adapter = mw8.ModelOptNativeFp8W8A16CheckpointAdapter.detect(_source(root))
    assert adapter is not None


def test_detect_uses_resolved_model_root_for_remote_sources(tmp_path):
    root = _write_model_dir(tmp_path)
    adapter = mw8.ModelOptNativeFp8W8A16CheckpointAdapter.detect(_remote_source("owner/repo", root))
    assert adapter is not None


def test_detect_none_without_sidecar(tmp_path):
    root = tmp_path / "empty"
    (root / "transformer").mkdir(parents=True)
    assert mw8.ModelOptNativeFp8W8A16CheckpointAdapter.detect(_source(str(root))) is None


def test_probe_main_exit0(tmp_path, capsys):
    root = _write_model_dir(tmp_path)
    rc = mw8.main([root])
    captured = capsys.readouterr()
    assert rc == 0
    assert "2 MLP targets resident" in captured.out
    assert "1 non-target(s) dequant" in captured.out


def test_probe_main_usage():
    assert mw8.main([]) == 2


# --- recipe-gated dispatch (default: dequant, W8A16 via explicit opt-in) --------


def test_dispatch_default_selects_dequant(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_FP8_BLOCKWISE_W8A16", raising=False)
    root = _write_model_dir(tmp_path)
    adapter = get_checkpoint_adapter(torch.nn.Module(), _source(root), quant_config=None, use_safetensors=True)
    assert isinstance(adapter, ModelOptNativeFp8CheckpointAdapter)
    assert not isinstance(adapter, ModelOptNativeFp8W8A16CheckpointAdapter)


def test_dispatch_opt_in_selects_w8a16(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FP8_BLOCKWISE_W8A16", "1")
    root = _write_model_dir(tmp_path)
    cfg = build_fp8_blockwise_w8a16_config()
    adapter = get_checkpoint_adapter(torch.nn.Module(), _source(root), quant_config=cfg, use_safetensors=True)
    assert isinstance(adapter, ModelOptNativeFp8W8A16CheckpointAdapter)


def test_dispatch_opt_in_rejects_missing_w8a16_config(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FP8_BLOCKWISE_W8A16", "1")
    root = _write_model_dir(tmp_path)
    with pytest.raises(CheckpointIntegrityError, match="W8A16"):
        get_checkpoint_adapter(torch.nn.Module(), _source(root), quant_config=None, use_safetensors=True)


def test_dispatch_default_inert_without_fp8_sidecar(tmp_path, monkeypatch):
    # No FP8 sidecar (mirrors NVFP4-dist, which has no root quantization_config.json):
    # the recipe predicate is False, so W8A16 does not engage and NVFP4 is unaffected.
    monkeypatch.setenv("VLLM_OMNI_FP8_BLOCKWISE_W8A16", "1")
    root = tmp_path / "no_sidecar"
    (root / "transformer").mkdir(parents=True)
    adapter = get_checkpoint_adapter(torch.nn.Module(), _source(str(root)), quant_config=None, use_safetensors=True)
    # No sidecar at all -> no adapter engages (W8A16 declines, dequant/NVFP4 find no
    # sidecar); assert the exact None (stronger than "not W8A16", which None also satisfies).
    assert adapter is None


def test_adapter_rejects_declared_forbidden_family():
    # A sidecar whose manifest DECLARES a forbidden family (self_attn.*) as quantized ->
    # fail-fast at adapter construction, before any weight is routed resident.
    bad = _sidecar()
    bad["mixed_precision"]["quantized"] = ["mlp.*", "self_attn.*"]
    with pytest.raises(CheckpointIntegrityError, match="target family"):
        _adapter(bad)


# --- adapter numerical correctness + edge cases (sharded-review hardening) ---


def test_adapt_lm_head_dequant_numerically_correct():
    w = _fp8(256, 128)
    s = _scale(256, 128)
    stream = [
        ("transformer.lm_head.weight", w),
        ("transformer.lm_head.weight_quantizer._scale", s),
        ("transformer.lm_head.weight_quantizer._amax", _amax(256, 128)),
    ]
    out = dict(_adapter(_sidecar(n_quantized=1, n_scale=2)).adapt(stream))
    got = out["transformer.lm_head.weight"]
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got, dequantize_weight(w, s, torch.bfloat16, (128, 128)))


def test_adapt_scale_before_weight_flushes_pending():
    # non-target (lm_head) scale arrives BEFORE its weight -> dequant still correct
    w = _fp8(256, 128)
    s = _scale(256, 128)
    stream = [
        ("transformer.lm_head.weight_quantizer._scale", s),
        ("transformer.lm_head.weight", w),
        ("transformer.lm_head.weight_quantizer._amax", _amax(256, 128)),
    ]
    out = dict(_adapter(_sidecar(n_quantized=1, n_scale=2)).adapt(stream))
    torch.testing.assert_close(out["transformer.lm_head.weight"], dequantize_weight(w, s, torch.bfloat16, (128, 128)))


def test_adapt_max_pending_overflow_aborts():
    # >8 non-target fp8 weights buffered without scales -> fail-fast (no GB buffering)
    stream = [(f"transformer.extra{i}.weight", _fp8(128, 128)) for i in range(9)]
    with pytest.raises(CheckpointIntegrityError, match="pending"):
        dict(_adapter(_sidecar(n_quantized=9, n_scale=18)).adapt(stream))


def test_is_fp8_dtype_classifies_storage_dtypes():
    # Float8 resident weights are FP8; BF16 dequant output and uint8 packed
    # NVFP4 storage are not.
    assert is_fp8_dtype(torch.float8_e4m3fn) is True
    assert is_fp8_dtype(torch.bfloat16) is False
    assert is_fp8_dtype(torch.uint8) is False
