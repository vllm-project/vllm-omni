# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import argparse
import json
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

p = argparse.ArgumentParser()
p.add_argument("baseline", type=Path)
p.add_argument("cached", type=Path)
a = p.parse_args()
b = json.loads((a.baseline / "results.json").read_text())
c = json.loads((a.cached / "results.json").read_text())
assert [(r["name"], r["prompt"], r["seed"]) for r in b["records"]] == [
    (r["name"], r["prompt"], r["seed"]) for r in c["records"]
]
rows = []
for base, cache in zip(b["records"], c["records"]):
    x = np.load(a.baseline / (base["name"] + ".npy"))
    y = np.load(a.cached / (cache["name"] + ".npy"))
    assert x.shape == y.shape
    ssim = [
        structural_similarity(
            u, v, channel_axis=-1, data_range=1, gaussian_weights=True, sigma=1.5, use_sample_covariance=False
        )
        for u, v in zip(x, y)
    ]
    dx = np.diff(x, axis=0)
    dy = np.diff(y, axis=0)
    rows.append(
        {
            "name": base["name"],
            "ssim": float(np.mean(ssim)),
            "temporal_error_ratio": float(np.abs(dy - dx).mean() / (np.abs(dx).mean() + 1e-8)),
        }
    )
ratio = sum(r["seconds"] for r in c["records"]) / sum(r["seconds"] for r in b["records"])
summary = {
    "mean_ssim": float(np.mean([r["ssim"] for r in rows])),
    "min_video_ssim": min(r["ssim"] for r in rows),
    "mean_temporal_error_ratio": float(np.mean([r["temporal_error_ratio"] for r in rows])),
    "latency_ratio": ratio,
    "videos": rows,
}
summary["passed"] = (
    summary["mean_ssim"] >= 0.95
    and summary["min_video_ssim"] >= 0.9
    and summary["mean_temporal_error_ratio"] <= 0.1
    and ratio <= 0.9
)
(a.cached / "comparison.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != "videos"}))
