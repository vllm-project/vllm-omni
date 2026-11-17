# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Audit real-hook decisions and paired final latents after a campaign."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

WARMUP_REQUESTS = ("warmup", "engine-warmup")


def expected_groups(pp: int, cfg: int) -> set[tuple[int, int, str]]:
    """Every (PP rank, CFG rank, branch) that must make TeaCache decisions."""
    if cfg == 2:
        branches = {0: ("teacache_positive",), 1: ("teacache_negative",)}
    else:
        branches = {0: ("teacache_positive", "teacache_negative")}
    return {(p, c, branch) for p in range(pp) for c in range(cfg) for branch in branches[c]}


def audit_run(run: Path, pp: int, cfg: int) -> dict:
    """Check that every stage, CFG rank, branch and request was observed.

    The request manifest and step count come from the cached run's
    ``results.json``; missing groups, steps or latent pairs fail the audit.
    """
    results = json.loads((run / "cache/results.json").read_text())
    requests = [record["name"] for record in results["records"]]
    assert requests, "Empty request manifest"
    assert len(set(requests)) == len(requests), "Duplicate request names"
    steps = 6 if results["args"]["split"] == "smoke" else 50

    groups = defaultdict(list)
    for path in (run / "trace/cache").glob("rank-*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["kind"] == "decision" and row["request_name"] not in WARMUP_REQUESTS:
                groups[(row["pp_rank"], row["cfg_rank"], row["context"], row["request_name"])].append(row)
    assert groups, "Missing hook decisions"

    expected = expected_groups(pp, cfg)
    observed = {key[:3] for key in groups}
    assert observed <= expected, f"Unexpected stage/rank/branch groups: {sorted(observed - expected)}"
    missing = sorted((*group, request) for group in expected for request in requests if (*group, request) not in groups)
    assert not missing, f"Missing decisions for {len(missing)} stage/rank/branch/request groups, e.g. {missing[:4]}"
    unexpected = sorted({key[3] for key in groups} - set(requests))
    assert not unexpected, f"Decisions for requests outside the manifest: {unexpected}"

    counts = {f"pp{p}/cfg{c}/{branch}": {"full": 0, "hits": 0, "requests": 0} for p, c, branch in sorted(expected)}
    for (p, c, branch, request), rows in groups.items():
        key = (p, c, branch, request)
        assert [row["step"] for row in rows] == list(range(steps)), (key, "incomplete or reordered step sequence")
        assert rows[0]["compute"], (key, "first call reused a previous request")
        count = counts[f"pp{p}/cfg{c}/{branch}"]
        count["requests"] += 1
        count["full"] += sum(row["compute"] for row in rows)
        count["hits"] += sum(not row["compute"] for row in rows)

    latents = []
    names = {
        mode: {p.stem for p in (run / "trace" / mode).glob("*.pt")} - set(WARMUP_REQUESTS) for mode in ("none", "cache")
    }
    for mode, found in names.items():
        assert found == set(requests), (
            f"{mode} latents do not match the request manifest: {sorted(found ^ set(requests))}"
        )
    for name in requests:
        x = torch.load(run / "trace/none" / f"{name}.pt", weights_only=True).float()
        y = torch.load(run / "trace/cache" / f"{name}.pt", weights_only=True).float()
        assert x.shape == y.shape and torch.isfinite(x).all() and torch.isfinite(y).all()
        error = y - x
        latents.append(
            {
                "name": name,
                "max_abs": error.abs().max().item(),
                "rmse": error.square().mean().sqrt().item(),
                "relative_l2": (error.norm() / x.norm().clamp_min(1e-8)).item(),
            }
        )
    return {
        "counts": counts,
        "latents": latents,
        "all_branches_skipped_blocks": all(row["hits"] > 0 for row in counts.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--cfg", type=int, required=True)
    args = parser.parse_args()
    summary = audit_run(args.run, args.pp, args.cfg)
    (args.run / "audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"counts": summary["counts"], "latent_pairs": len(summary["latents"])}))


if __name__ == "__main__":
    main()
