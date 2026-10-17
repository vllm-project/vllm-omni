# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Audit real-hook decisions and paired final latents after a campaign."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--cfg", type=int, required=True)
    args = parser.parse_args()
    groups = defaultdict(list)
    for path in (args.run / "trace/cache").glob("rank-*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["kind"] == "decision" and row["request_name"] not in ("warmup", "engine-warmup"):
                groups[(row["pp_rank"], row["cfg_rank"], row["context"], row["request_name"])].append(row)
    assert groups, "Missing hook decisions"
    assert {key[0] for key in groups} == set(range(args.pp))
    assert {key[1] for key in groups} == set(range(args.cfg))
    counts = defaultdict(lambda: {"full": 0, "hits": 0, "requests": 0})
    for (pp, cfg, branch, request), rows in groups.items():
        assert rows[0]["step"] == 0, (pp, cfg, branch, request, "request counter was not reset")
        assert rows[0]["compute"], (pp, cfg, branch, request, "first call reused a previous request")
        assert branch in ("teacache_positive", "teacache_negative")
        if args.cfg == 2:
            assert branch == ("teacache_positive" if cfg == 0 else "teacache_negative")
        key = f"pp{pp}/cfg{cfg}/{branch}"
        counts[key]["requests"] += 1
        counts[key]["full"] += sum(row["compute"] for row in rows)
        counts[key]["hits"] += sum(not row["compute"] for row in rows)
    latents = []
    for cached in sorted((args.run / "trace/cache").glob("*.pt")):
        if cached.stem in ("warmup", "engine-warmup"):
            continue
        x = torch.load(args.run / "trace/none" / cached.name, weights_only=True).float()
        y = torch.load(cached, weights_only=True).float()
        assert x.shape == y.shape and torch.isfinite(x).all() and torch.isfinite(y).all()
        error = y - x
        latents.append(
            {
                "name": cached.stem,
                "max_abs": error.abs().max().item(),
                "rmse": error.square().mean().sqrt().item(),
                "relative_l2": (error.norm() / x.norm().clamp_min(1e-8)).item(),
            }
        )
    assert latents, "Missing paired latent captures"
    summary = {
        "counts": dict(counts),
        "latents": latents,
        "all_branches_skipped_blocks": all(row["hits"] > 0 for row in counts.values()),
    }
    (args.run / "audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"counts": dict(counts), "latent_pairs": len(latents)}))


if __name__ == "__main__":
    main()
