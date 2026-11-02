# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Freeze a calibration-only choice, retaining failures as diagnostic candidates."""

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("campaigns", type=Path, nargs="+")
    args = parser.parse_args()
    candidates = []
    for campaign in args.campaigns:
        for path in sorted(campaign.rglob("comparison.json")):
            metrics = json.loads(path.read_text())
            run = json.loads((path.parent / "results.json").read_text())["args"]
            assert run["split"] == "calibration", "Held-out results must never select a profile"
            assert run["pp"] == args.pp
            hits = defaultdict(int)
            for log in (path.parent.parent / "trace/cache").glob("rank-*.jsonl"):
                for line in log.read_text().splitlines():
                    row = json.loads(line)
                    if row["kind"] == "decision" and row["request_name"] not in ("warmup", "engine-warmup"):
                        key = (row["pp_rank"], row["cfg_rank"], row["context"])
                        hits[key] += int(not row["compute"])
            if not hits or {key[0] for key in hits} != set(range(args.pp)) or not all(hits.values()):
                continue
            score = max(
                0.95 / max(metrics["mean_ssim"], 1e-8),
                0.90 / max(metrics["min_video_ssim"], 1e-8),
                metrics["mean_temporal_error_ratio"] / 0.10,
                metrics["latency_ratio"] / 0.90,
            )
            candidates.append(
                {
                    "threshold": run["threshold"],
                    "warmup_steps": run.get("cache_warmup_steps", 0),
                    "source": str(path),
                    "metrics": metrics,
                    "gate_score": score,
                }
            )
    assert candidates, "No candidate actually skipped blocks in every stage/branch"
    qualified = [row for row in candidates if row["metrics"]["passed"]]
    winner = (
        min(qualified, key=lambda row: row["metrics"]["latency_ratio"])
        if qualified
        else min(candidates, key=lambda row: (row["gate_score"], row["metrics"]["latency_ratio"]))
    )
    selection = {**winner, "calibration_all_gates_passed": bool(qualified), "diagnostic_only": not bool(qualified)}
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ["model.json", "prompts.json", f"coefficients-pp{args.pp}.json", f"coefficients-pp{args.pp}.fit.json"]:
        shutil.copyfile(args.inputs / name, args.output / name)
    (args.output / "model").symlink_to((args.inputs / "model").resolve(), target_is_directory=True)
    (args.output / f"selection-pp{args.pp}.json").write_text(json.dumps(selection, indent=2))
    manifest = {
        "validation_source_sha": subprocess.check_output(
            ["git", "-C", str(args.source_checkout), "rev-parse", "HEAD"], text=True
        ).strip(),
        "candidate_count": len(candidates),
        "input_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob("*.json")},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for path in args.output.glob("*.json"):
        path.chmod(0o444)
    print(
        json.dumps(
            {
                "threshold": selection["threshold"],
                "warmup_steps": selection["warmup_steps"],
                "diagnostic_only": selection["diagnostic_only"],
                "source": selection["source"],
            }
        )
    )


if __name__ == "__main__":
    main()
