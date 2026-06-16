# SPDX-License-Identifier: Apache-2.0
"""Accuracy metrics for BDE DreamZero validation (KV-on vs KV-off)."""

from __future__ import annotations

import numpy as np
import torch
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


def _to_nchw(frames) -> torch.Tensor:
    """``(T, H, W, C)`` uint8/float -> ``(T, C, H, W)`` float in ``[0, 1]``."""
    t = torch.as_tensor(np.asarray(frames))
    if t.ndim != 4:
        raise ValueError(f"expected (T, H, W, C) frames, got shape {tuple(t.shape)}")
    t = t.float()
    if float(t.max()) > 1.5:  # uint8 range
        t = t / 255.0
    return t.permute(0, 3, 1, 2).contiguous()


_LPIPS = "unloaded"


def _lpips_model():
    """Lazily build an LPIPS metric; ``None`` if its weights can't load (offline)."""
    global _LPIPS
    if _LPIPS == "unloaded":
        try:
            from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

            _LPIPS = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).eval()
        except Exception:  # weights unavailable / no network
            _LPIPS = None
    return _LPIPS


def frame_metrics(ref, test) -> dict:
    """Per-frame PSNR / SSIM (+ LPIPS if available) between two videos.

    ``ref`` is the reference (KV-off / model-local); ``test`` is KV-on.
    """
    r, t = _to_nchw(ref), _to_nchw(test)
    if r.shape != t.shape:
        raise ValueError(f"shape mismatch ref={tuple(r.shape)} test={tuple(t.shape)}")
    psnr, ssim = [], []
    for i in range(r.shape[0]):
        ri, ti = r[i : i + 1], t[i : i + 1]
        psnr.append(float(peak_signal_noise_ratio(ti, ri, data_range=1.0)))
        ssim.append(float(structural_similarity_index_measure(ti, ri, data_range=1.0)))
    out = {
        "num_frames": int(r.shape[0]),
        "psnr": psnr,
        "ssim": ssim,
        "psnr_min": min(psnr),
        "psnr_mean": sum(psnr) / len(psnr),
        "ssim_min": min(ssim),
        "ssim_mean": sum(ssim) / len(ssim),
        "max_abs_diff": float((t - r).abs().max()),
        "lpips": None,
    }
    model = _lpips_model()
    if model is not None:
        with torch.no_grad():
            lp = [float(model(t[i : i + 1].clamp(0, 1), r[i : i + 1].clamp(0, 1))) for i in range(r.shape[0])]
        out["lpips"] = lp
        out["lpips_max"] = max(lp)
        out["lpips_mean"] = sum(lp) / len(lp)
    return out


def tensor_parity(ref, test, *, atol: float = 1e-4, rtol: float = 1e-4) -> dict:
    """Strict numerical parity for attention-input / latent tensors (Tier A.1)."""
    r = torch.as_tensor(ref).float()
    t = torch.as_tensor(test).float()
    if r.shape != t.shape:
        raise ValueError(f"shape mismatch ref={tuple(r.shape)} test={tuple(t.shape)}")
    diff = (t - r).abs()
    return {
        "shape": list(r.shape),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "allclose": bool(torch.allclose(t, r, atol=atol, rtol=rtol)),
        "atol": atol,
        "rtol": rtol,
    }


def check_gates(
    metrics: dict, *, psnr_min: float = 40.0, ssim_min: float = 0.99, lpips_max: float = 0.01
) -> dict:
    """Apply the §5 acceptance gates to a ``frame_metrics`` result."""

    def gate(value, threshold, ok):
        return {"pass": ok, "value": value, "threshold": threshold}

    gates = {
        "psnr": gate(metrics["psnr_min"], psnr_min, metrics["psnr_min"] >= psnr_min),
        "ssim": gate(metrics["ssim_min"], ssim_min, metrics["ssim_min"] >= ssim_min),
    }
    if metrics.get("lpips") is not None:
        gates["lpips"] = gate(metrics["lpips_max"], lpips_max, metrics["lpips_max"] <= lpips_max)
    return {"passed": all(g["pass"] for g in gates.values()), "gates": gates}
