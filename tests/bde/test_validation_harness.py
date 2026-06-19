# SPDX-License-Identifier: Apache-2.0
"""Tests for the BDE DreamZero validation harness (metrics / artifacts / config).

Exercises everything except the weighted model run (synthetic videos/tensors).
"""

import numpy as np
import pytest
import torch

from benchmarks.bde import artifacts, metrics
from benchmarks.bde.dreamzero_config import bde_config_for_dreamzero
from benchmarks.bde.parity_check import compare


def _vid(t=6, h=32, w=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (t, h, w, 3)).astype(np.uint8)


# --- metrics ----------------------------------------------------------------


def test_frame_metrics_identical():
    v = _vid()
    m = metrics.frame_metrics(v, v)
    assert m["max_abs_diff"] == 0.0
    assert m["ssim_min"] == pytest.approx(1.0)
    assert m["psnr_min"] == float("inf")


def test_frame_metrics_detects_difference():
    v = _vid()
    w = v.copy()
    w[:, :16] = 0
    m = metrics.frame_metrics(v, w)
    assert m["max_abs_diff"] > 0
    assert m["ssim_min"] < 1.0
    assert m["psnr_min"] < float("inf")


def test_tensor_parity():
    a = torch.zeros(2, 3, 4)
    assert metrics.tensor_parity(a, a.clone())["allclose"]
    b = a.clone()
    b[0, 0, 0] = 1.0
    r = metrics.tensor_parity(a, b)
    assert not r["allclose"] and r["max_abs"] == 1.0


def test_check_gates():
    assert metrics.check_gates({"psnr_min": 50.0, "ssim_min": 0.999})["passed"]
    assert not metrics.check_gates({"psnr_min": 30.0, "ssim_min": 0.999})["passed"]
    assert not metrics.check_gates({"psnr_min": 50.0, "ssim_min": 0.5})["passed"]


# --- config derivation ------------------------------------------------------


def test_config_derivation_matches_window():
    # Frame-granular: one pool block = one frame, so chunk_size == frame_seqlen and
    # window_chunks == local_attn_size (matches BDEModelRunner._preallocate_kv_cache,
    # which overwrites chunk_size to frame_seqlen but preserves window_chunks).
    cfg = bde_config_for_dreamzero(num_frame_per_block=3, frame_seqlen=220, local_attn_size=21)
    assert cfg.chunk_size == 220
    assert cfg.window_chunks == 21
    assert cfg.sliding_window == 21 * 220  # == max_attention_size
    assert cfg.enable and not cfg.reset_at_boundary


def test_config_window_is_frame_granular():
    # Regression: frame granularity makes the window exact for any
    # num_frame_per_block (no divisibility constraint), and window_chunks tracks
    # local_attn_size rather than local_attn_size // num_frame_per_block — the old
    # causal-block math the runtime would silently shrink to a too-small window.
    cfg = bde_config_for_dreamzero(num_frame_per_block=4, frame_seqlen=10, local_attn_size=21)
    assert cfg.chunk_size == 10
    assert cfg.window_chunks == 21
    assert cfg.sliding_window == 21 * 10


def test_config_rejects_full_attention():
    with pytest.raises(ValueError):
        bde_config_for_dreamzero(num_frame_per_block=3, frame_seqlen=10, local_attn_size=-1)


# --- artifacts --------------------------------------------------------------


def test_save_load_mp4_roundtrip(tmp_path):
    v = _vid()
    path = artifacts.save_mp4(tmp_path / "a.mp4", v, fps=5)
    back = artifacts.load_mp4(path)
    assert back.shape == v.shape  # mp4v is lossy; shape must match


def test_side_by_side_and_diff_shapes():
    a, b = _vid(seed=1), _vid(seed=2)
    sbs = artifacts.side_by_side(a, b)
    assert sbs.shape[0] == a.shape[0]
    assert sbs.shape[2] >= a.shape[2] * 2  # two panes + gap
    d = artifacts.diff_heatmap(a, b)
    assert d.shape[:3] == a.shape[:3] and d.shape[3] == 3


def test_write_run_json_handles_inf(tmp_path):
    import json

    p = artifacts.write_run_json(tmp_path / "run.json", metrics={"psnr_min": float("inf")})
    data = json.loads(p.read_text())
    assert data["metrics"]["psnr_min"] == "Infinity"


# --- end-to-end compare pipeline (no model) ---------------------------------


def test_compare_pipeline_produces_artifacts(tmp_path):
    v = _vid()
    off = artifacts.save_mp4(tmp_path / "off.mp4", v, fps=5)
    on = artifacts.save_mp4(tmp_path / "on.mp4", v, fps=5)
    res = compare(off, on, tmp_path / "cmp", label="t")
    out = tmp_path / "cmp"
    assert (out / "run.json").exists()
    assert (out / "t_sidebyside.mp4").exists()
    assert (out / "t_diff.mp4").exists()
    assert (out / "t_metrics.png").exists()
    # Same source through the same codec -> near-perfect parity, gates pass.
    assert res["metrics"]["ssim_min"] > 0.9
    assert res["gates"]["passed"]


# --- profiling (memory plateau, no model) -----------------------------------


def test_kv_memory_plateau(tmp_path):
    from benchmarks.bde.dreamzero_config import bde_config_for_dreamzero
    from benchmarks.bde.profile import kv_memory_plateau, plot_memory

    cfg = bde_config_for_dreamzero(num_frame_per_block=1, frame_seqlen=16, local_attn_size=2)
    prof = kv_memory_plateau(cfg, num_chunks=10, num_layers=2, num_kv_heads=4, head_size=64, block_size=16)
    # Usage plateaus once the window fills; resident bounded by window + 1.
    assert prof["plateaued"] is True
    assert max(prof["resident_blocks"]) <= cfg.window_chunks + 1
    assert prof["peak_used"] == prof["used_blocks"][cfg.window_chunks]
    out = plot_memory(tmp_path / "mem.png", prof)
    assert out.exists()
