# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the offline W4A8 exporter's key remap and QKV pre-fusion.

The exporter (``examples/quantization/export_quark_svdquant_w4a8.py``) runs Quark
+ a GPU at export time, but its state-dict shaping is pure tensor logic and must
be correct without either. These tests exercise that logic directly.
"""

import importlib.util
import os

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_EXPORTER = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "examples",
    "quantization",
    "export_quark_svdquant_w4a8.py",
)


def _load_exporter():
    spec = importlib.util.spec_from_file_location("export_quark_svdquant_w4a8", os.path.abspath(_EXPORTER))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remap_keys():
    mod = _load_exporter()
    assert mod._remap_key("b.0.attn2.to_q.layer.weight") == "b.0.attn2.to_q.weight"
    assert mod._remap_key("b.0.attn2.to_q.layer.bias") == "b.0.attn2.to_q.bias"
    assert mod._remap_key("b.0.attn2.to_q.correction.l1.weight") == "b.0.attn2.to_q.proj_down"
    assert mod._remap_key("b.0.attn2.to_q.correction.l2.weight") == "b.0.attn2.to_q.proj_up"
    # A plain param (not an ErrorCorrectedModule) is untouched.
    assert mod._remap_key("b.0.norm.weight") == "b.0.norm.weight"


def test_qkv_fuse_roundtrip():
    """The fused residual + block-diagonal correction must equal the three
    separate per-head linear outputs concatenated."""
    mod = _load_exporter()
    torch.manual_seed(0)
    k, rank = 64, 4
    sizes = {"q": 48, "k": 48, "v": 16}
    x = torch.randn(5, k)
    prefix = "blocks.0.attn1."

    sd, refs = {}, {}
    for name, n in sizes.items():
        weight = torch.randn(n, k)
        l2 = torch.randn(n, rank)
        l1 = torch.randn(rank, k)
        residual = weight - l2 @ l1
        sd[f"{prefix}to_{name}.weight"] = residual
        sd[f"{prefix}to_{name}.proj_down"] = l1
        sd[f"{prefix}to_{name}.proj_up"] = l2
        refs[name] = x @ weight.T

    mod._fuse_qkv(sd)

    n_total = sum(sizes.values())
    assert {k for k in sd if "attn1" in k} == {
        f"{prefix}to_qkv.weight",
        f"{prefix}to_qkv.proj_down",
        f"{prefix}to_qkv.proj_up",
    }
    fused_w = sd[f"{prefix}to_qkv.weight"]
    proj_down = sd[f"{prefix}to_qkv.proj_down"]
    proj_up = sd[f"{prefix}to_qkv.proj_up"]
    assert fused_w.shape == (n_total, k)
    assert proj_down.shape == (3 * rank, k)
    assert proj_up.shape == (n_total, 3 * rank)

    out = x @ fused_w.T + (x @ proj_down.T) @ proj_up.T
    ref = torch.cat([refs["q"], refs["k"], refs["v"]], dim=1)
    assert torch.allclose(out, ref, atol=1e-4)

    # proj_up is block-diagonal: the q rows (0:48) only touch the q rank block (0:rank).
    assert torch.count_nonzero(proj_up[: sizes["q"], rank:]) == 0


def test_fold_unsupported_factors_folds_small_layer():
    """A layer the vLLM-Omni SVD gate rejects has its correction folded back into
    the weight so no orphan factor keys reach the checkpoint."""
    mod = _load_exporter()
    torch.manual_seed(0)
    k, rank, n = 64, 4, 48  # n=48 < 256 -> not SVD-tileable
    weight = torch.randn(n, k)
    l2 = torch.randn(n, rank)
    l1 = torch.randn(rank, k)
    sd = {"x.weight": weight - l2 @ l1, "x.proj_down": l1, "x.proj_up": l2}

    mod._fold_unsupported_factors(sd)

    assert "x.proj_down" not in sd and "x.proj_up" not in sd
    assert torch.allclose(sd["x.weight"], weight, atol=1e-4)


def test_fold_unsupported_factors_keeps_aligned_layer():
    mod = _load_exporter()
    k, rank, n = 256, 4, 256  # both 256-aligned -> kept
    sd = {
        "y.weight": torch.randn(n, k),
        "y.proj_down": torch.randn(rank, k),
        "y.proj_up": torch.randn(n, rank),
    }
    mod._fold_unsupported_factors(sd)
    assert "y.proj_down" in sd and "y.proj_up" in sd
