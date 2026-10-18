#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare dumped latents between two dump dirs (from --dump-outputs-dir).

Files are named <tag>-NNN.pt; both sides must carry the same prompts/sizes/order,
so files pair up by name. Reports per-pair max-abs diff, max relative diff and
PSNR, plus an overall summary.

Usage:
  python mixfusion_ab_latent_diff.py <dirA> <dirB>
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def load_dir(d: Path) -> dict[str, torch.Tensor]:
    return {p.name: torch.load(p, map_location="cpu", weights_only=True) for p in sorted(d.glob("*.pt"))}


def compare(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    diff = (a.float() - b.float()).abs()
    max_abs = diff.max().item()
    denom = b.float().abs().clamp_min(1e-6)
    max_rel = (diff / denom).max().item()
    mse = diff.pow(2).mean().item()
    psnr = 10 * torch.log10(torch.tensor(4.0) / mse).item() if mse > 0 else float("inf")
    return max_abs, max_rel, psnr


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <dirA> <dirB>")
    da, db = Path(sys.argv[1]), Path(sys.argv[2])
    ta, tb = load_dir(da), load_dir(db)
    if set(ta) != set(tb):
        sys.exit(f"file sets differ:\n  {da}: {sorted(ta)}\n  {db}: {sorted(tb)}")
    worst = (0.0, 0.0, float("inf"))
    for name in sorted(ta):
        a, b = ta[name], tb[name]
        if a.shape != b.shape:
            print(f"{name}: SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        mx, mr, ps = compare(a, b)
        worst = (max(worst[0], mx), max(worst[1], mr), min(worst[2], ps))
        print(f"{name}: shape={tuple(a.shape)} max_abs={mx:.3e} max_rel={mr:.3e} psnr={ps:.2f}dB")
    print(f"\nsummary: max_abs={worst[0]:.3e} max_rel={worst[1]:.3e} worst_psnr={worst[2]:.2f}dB")


if __name__ == "__main__":
    main()
