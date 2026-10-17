# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the AITER MHA v4 diffusion attention backend."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.backends import aiter_quant
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.registry import (
    DiffusionAttentionBackendEnum,
)
from vllm_omni.diffusion.attention.backends.utils import aiter_mha_v4
from vllm_omni.diffusion.data import AiterQuantSpec, AttentionSpec

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_GFX942_FORMATS = ("fp8", "i8fp8")
_GFX950_FORMATS = (
    "bf16",
    "f6f4",
    "f8f6",
    "fp8",
    "i8fp8",
    "mxfp4",
    "mxfp6",
    "mxfp8",
)


class _TestPlatform:
    def __init__(self, arch: str):
        self.arch = arch

    def get_gfx_arch(self) -> str:
        return self.arch


def _impl(monkeypatch, *, arch: str = "gfx950", **overrides):
    monkeypatch.setattr(
        aiter_quant,
        "current_omni_platform",
        _TestPlatform(arch),
    )
    monkeypatch.setattr(aiter_quant, "check_aiter_mha_v4_available", lambda: None)
    kwargs = {
        "num_heads": 8,
        "head_size": 128,
        "softmax_scale": 128**-0.5,
        "causal": False,
        "num_kv_heads": 8,
        "qkv_layout": "BSHD",
        "backend_kwargs": {"format": "fp8"},
    }
    kwargs.update(overrides)
    return aiter_quant.AiterQuantImpl(**kwargs)


def test_config_and_backend_default_to_fp8(monkeypatch):
    spec = AttentionSpec(backend="AITER_QUANT_ATTN")

    assert spec.aiter_quant is not None
    assert spec.aiter_quant.format == "fp8"
    assert spec.backend_kwargs() == {"format": "fp8"}
    assert _impl(monkeypatch, backend_kwargs=None).format == "fp8"


@pytest.mark.parametrize("format_name", _GFX950_FORMATS)
def test_config_normalizes_and_serializes_valid_formats(format_name):
    spec = AttentionSpec(
        backend="AITER_QUANT_ATTN",
        aiter_quant={"format": format_name.upper()},
    )

    assert spec.aiter_quant is not None
    assert spec.aiter_quant.format == format_name
    assert spec.backend_kwargs() == {"format": format_name}


@pytest.mark.parametrize("format_name", ["mxfp0", "unknown"])
def test_config_rejects_unsupported_formats(format_name):
    with pytest.raises(ValueError, match="aiter_quant.format"):
        AttentionSpec(
            backend="AITER_QUANT_ATTN",
            aiter_quant={"format": format_name},
        )


def test_config_rejects_aiter_options_for_other_backends():
    with pytest.raises(ValueError, match="only supported by the AITER_QUANT_ATTN"):
        AttentionSpec(
            backend="TORCH_SDPA",
            aiter_quant={"format": "fp8"},
        )


def test_registry_resolves_aiter_quant_backend():
    backend = DiffusionAttentionBackendEnum.AITER_QUANT_ATTN

    assert backend.get_class() is aiter_quant.AiterQuantBackend


@pytest.mark.parametrize(
    ("arch", "format_name"),
    [
        *(("gfx942", format_name) for format_name in _GFX942_FORMATS),
        *(("gfx950", format_name) for format_name in _GFX950_FORMATS),
    ],
)
def test_architecture_accepts_supported_formats(monkeypatch, arch, format_name):
    impl = _impl(
        monkeypatch,
        arch=arch,
        backend_kwargs={"format": format_name},
    )

    assert impl.format == format_name


@pytest.mark.parametrize(
    "format_name",
    sorted(set(_GFX950_FORMATS) - set(_GFX942_FORMATS)),
)
def test_gfx942_rejects_gfx950_only_formats(monkeypatch, format_name):
    with pytest.raises(RuntimeError, match="not available on gfx942"):
        _impl(
            monkeypatch,
            arch="gfx942",
            backend_kwargs={"format": format_name},
        )


def test_rejects_unsupported_architecture(monkeypatch):
    with pytest.raises(RuntimeError, match="requires a gfx942 or gfx950"):
        _impl(monkeypatch, arch="gfx1100")


@pytest.mark.parametrize("gqa_ratio", [1, 2, 4, 8, 16])
def test_accepts_supported_gqa_ratios(monkeypatch, gqa_ratio):
    impl = _impl(
        monkeypatch,
        num_heads=2 * gqa_ratio,
        num_kv_heads=2,
    )

    assert impl.format == "fp8"


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"causal": True}, NotImplementedError, "causal attention"),
        ({"head_size": 64}, NotImplementedError, "head_dim=128"),
        ({"qkv_layout": "BHSD"}, ValueError, "expects.*BSHD"),
        (
            {"num_heads": 8, "num_kv_heads": 3},
            ValueError,
            "divisible by KV heads",
        ),
        (
            {"num_heads": 32, "num_kv_heads": 1},
            ValueError,
            "supports GQA ratios",
        ),
    ],
)
def test_rejects_unsupported_attention_contracts(
    monkeypatch,
    overrides,
    error,
    match,
):
    with pytest.raises(error, match=match):
        _impl(monkeypatch, **overrides)


@pytest.mark.parametrize("qkv_layout", [None, "BSND", "BSHD", "bsnd"])
def test_accepts_bshd_compatible_layouts(monkeypatch, qkv_layout):
    impl = _impl(monkeypatch, qkv_layout=qkv_layout)

    assert impl.qkv_layout == qkv_layout


def test_accepts_bfloat16_model_dtype(monkeypatch):
    monkeypatch.setattr(
        aiter_quant,
        "get_current_diffusion_config_or_none",
        lambda: SimpleNamespace(dtype=torch.bfloat16),
    )

    assert _impl(monkeypatch).format == "fp8"


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rejects_non_bfloat16_model_dtype(monkeypatch, dtype):
    monkeypatch.setattr(
        aiter_quant,
        "get_current_diffusion_config_or_none",
        lambda: SimpleNamespace(dtype=dtype),
    )

    with pytest.raises(TypeError, match="requires bfloat16"):
        _impl(monkeypatch)


def test_format_lists_stay_in_sync():
    # Adding a format means touching the arch table, the config spec, and the recipe table.
    assert set(_GFX942_FORMATS) == aiter_quant._FORMATS_BY_ARCH["gfx942"]
    assert set(_GFX950_FORMATS) == aiter_quant._FORMATS_BY_ARCH["gfx950"]
    assert set(aiter_mha_v4._FORWARD_FNS) == set(aiter_quant._ALL_FORMATS)
    assert set(AiterQuantSpec._VALID_FORMATS) == set(aiter_quant._ALL_FORMATS)


def test_missing_aiter_reports_actionable_error(monkeypatch):
    monkeypatch.setattr(
        aiter_quant,
        "current_omni_platform",
        _TestPlatform("gfx950"),
    )

    def unavailable():
        raise RuntimeError("AITER_QUANT_ATTN requires an AITER build containing aiter.ops.mha_v4.")

    monkeypatch.setattr(aiter_quant, "check_aiter_mha_v4_available", unavailable)

    with pytest.raises(RuntimeError, match="requires an AITER build"):
        aiter_quant.AiterQuantImpl(
            num_heads=8,
            head_size=128,
            softmax_scale=128**-0.5,
            num_kv_heads=8,
            qkv_layout="BSHD",
            backend_kwargs={"format": "fp8"},
        )


def test_validate_available_reports_failed_aiter_import(monkeypatch):
    import_error = ImportError("No module named 'aiter'")
    monkeypatch.setattr(aiter_mha_v4, "_MHA_V4_IMPORT_ERROR", import_error)

    with pytest.raises(RuntimeError, match="requires an AITER build providing aiter.ops.mha_v4") as exc_info:
        aiter_quant.AiterQuantBackend.validate_available()

    assert exc_info.value.__cause__ is import_error


def test_validate_available_rejects_mha_v4_without_required_kwargs(monkeypatch):
    def old_mha_v4(query, key, value, q_format, k_format, v_format, softmax_scale=None):
        raise AssertionError("must not be called")

    monkeypatch.setattr(aiter_mha_v4, "_MHA_V4_IMPORT_ERROR", None)
    monkeypatch.setattr(aiter_mha_v4, "_aiter_mha_v4", old_mha_v4)

    with pytest.raises(RuntimeError, match="does not accept") as exc_info:
        aiter_quant.AiterQuantBackend.validate_available()

    for kwarg in ("q_scale_mode", "k_scale_mode", "v_scale_mode"):
        assert kwarg in str(exc_info.value)
    assert "softmax_scale" not in str(exc_info.value)


def test_validate_available_accepts_complete_mha_v4(monkeypatch):
    def new_mha_v4(
        query,
        key,
        value,
        q_format,
        k_format,
        v_format,
        softmax_scale=None,
        q_scale_mode=None,
        k_scale_mode=None,
        v_scale_mode=None,
    ):
        raise AssertionError("must not be called")

    monkeypatch.setattr(aiter_mha_v4, "_MHA_V4_IMPORT_ERROR", None)
    monkeypatch.setattr(aiter_mha_v4, "_aiter_mha_v4", new_mha_v4)

    aiter_quant.AiterQuantBackend.validate_available()


def test_rejects_attention_mask(monkeypatch):
    impl = _impl(monkeypatch)
    tensor = torch.zeros((1, 2, 8, 128), dtype=torch.bfloat16)
    metadata = AttentionMetadata(attn_mask=torch.ones((1, 2), dtype=torch.bool))

    with pytest.raises(NotImplementedError, match="attention masks"):
        impl.forward_hip(tensor, tensor, tensor, metadata)
