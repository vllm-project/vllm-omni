# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Refine thresholds on calibration prompts only; preserve every measured candidate."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--thresholds", type=float, nargs="+", required=True)
    args = parser.parse_args()
    scripts = Path(__file__).parent
    inputs = Path(os.environ["WAN_VALIDATION_ROOT"])
    candidates = []
    for threshold in args.thresholds:
        output = args.root / f"tune-{threshold}"
        output.mkdir(parents=True, exist_ok=True)
        env = dict(
            os.environ,
            WAN_TRACE_MODE="cache",
            WAN_TRACE_DIR=str(output / "trace"),
            WAN_CONTROL=str(output / "control.json"),
            WAN_COEFFICIENTS=str(inputs / f"coefficients-pp{args.pp}.json"),
        )
        command = [
            sys.executable,
            str(scripts / "generate.py"),
            "--pp",
            str(args.pp),
            "--cfg",
            "1",
            "--mode",
            "cache",
            "--split",
            "calibration",
            "--paired",
            "--limit",
            "6",
            "--cache-warmup-steps",
            str(args.warmup_steps),
            "--threshold",
            str(threshold),
            "--out",
            str(output),
        ]
        with (output / "run.log").open("w") as log:
            log.write(json.dumps({"command": command}) + "\n")
            log.flush()
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
        subprocess.run(
            [sys.executable, str(scripts / "compare.py"), str(output / "none"), str(output / "cache")], check=True
        )
        subprocess.run(
            [sys.executable, str(scripts / "audit.py"), str(output), "--pp", str(args.pp), "--cfg", "1"], check=True
        )
        metric = json.loads((output / "cache/comparison.json").read_text())
        audit = json.loads((output / "audit.json").read_text())
        candidates.append(
            {
                "threshold": threshold,
                "warmup_steps": args.warmup_steps,
                "metrics": metric,
                "actual_hits": audit["all_branches_skipped_blocks"],
            }
        )
        (args.root / "candidates.json").write_text(json.dumps(candidates, indent=2))
    eligible = [
        row
        for row in candidates
        if row["actual_hits"]
        and row["metrics"]["mean_ssim"] >= 0.95
        and row["metrics"]["min_video_ssim"] >= 0.90
        and row["metrics"]["mean_temporal_error_ratio"] <= 0.10
    ]
    if not eligible:
        raise RuntimeError("No refined threshold met quality gates with actual block skips")
    winner = min(eligible, key=lambda row: row["metrics"]["latency_ratio"])
    selection = {
        "threshold": winner["threshold"],
        "warmup_steps": args.warmup_steps,
        "calibration_latency_ratio": winner["metrics"]["latency_ratio"],
        "source": str(args.root / "candidates.json"),
    }
    (inputs / f"selection-pp{args.pp}.json").write_text(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
