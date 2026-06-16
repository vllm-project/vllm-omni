# SPDX-License-Identifier: Apache-2.0
"""Save BDE DreamZero validation artifacts: videos, comparisons, run.json."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np


def _u8(frames) -> np.ndarray:
    a = np.asarray(frames)
    if a.dtype != np.uint8:
        a = (a * 255.0 if float(a.max()) <= 1.5 else a).clip(0, 255).astype(np.uint8)
    return a


def save_mp4(path, frames, fps: int = 5) -> Path:
    """Write ``(T, H, W, C)`` RGB frames to mp4 (matches the export script writer)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = _u8(frames)
    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return path


def load_mp4(path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames, axis=0)


def side_by_side(*videos, gap: int = 4) -> np.ndarray:
    """Stitch videos horizontally (``real | off | on``) into one clip."""
    vids = [_u8(v) for v in videos]
    n_frames = min(v.shape[0] for v in vids)
    height = max(v.shape[1] for v in vids)
    cols = []
    for v in vids:
        v = v[:n_frames]
        if v.shape[1] < height:
            pad = np.zeros((n_frames, height - v.shape[1], v.shape[2], 3), np.uint8)
            v = np.concatenate([v, pad], axis=1)
        cols.append(v)
        cols.append(np.zeros((n_frames, height, gap, 3), np.uint8))
    return np.concatenate(cols[:-1], axis=2)


def diff_heatmap(ref, test) -> np.ndarray:
    """Colorized per-pixel abs-diff video ``|test - ref|`` (RGB)."""
    r = _u8(ref).astype(np.int16)
    t = _u8(test).astype(np.int16)
    n = min(r.shape[0], t.shape[0])
    d = np.abs(t[:n] - r[:n]).max(axis=-1).astype(np.uint8)  # (T, H, W)
    out = np.stack([cv2.applyColorMap(d[i], cv2.COLORMAP_JET) for i in range(n)])  # BGR
    return out[..., ::-1].copy()  # -> RGB


def metrics_plot(path, metrics: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    psnr = [min(p, 100.0) for p in metrics["psnr"]]  # cap inf for plotting
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax[0].plot(psnr)
    ax[0].axhline(40.0, ls="--", c="r")
    ax[0].set_ylabel("PSNR (dB, capped 100)")
    ax[1].plot(metrics["ssim"])
    ax[1].axhline(0.99, ls="--", c="r")
    ax[1].set_ylabel("SSIM")
    ax[1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def _safe(value):
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    return value


def write_run_json(path, **fields) -> Path:
    """Persist reproducibility metadata + metrics + gates (inf/nan-safe)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        return _safe(x)

    path.write_text(json.dumps(clean(fields), indent=2))
    return path
