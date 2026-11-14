# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Sequential calibration and held-out validation. Never fit to held-out data."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--pp", type=int, required=True)
p.add_argument("--cfg", type=int, default=1)
p.add_argument("--phase", choices=["calibrate", "validate"], required=True)
p.add_argument("--root", type=Path, required=True)
a = p.parse_args()
scripts = Path(__file__).parent
root = Path(os.environ.get("WAN_VALIDATION_ROOT", scripts))
a.root.mkdir(parents=True, exist_ok=True)


def run(name, mode, split, extra=(), coeff=None):
    out = a.root / name
    if (out / "run.log").exists() or (out / "provenance.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing experiment: {out}")
    env = dict(os.environ, WAN_TRACE_MODE=mode, WAN_TRACE_DIR=str(out / "trace"))
    if coeff:
        env["WAN_COEFFICIENTS"] = str(coeff)
    env["WAN_CONTROL"] = str(out / "control.json")
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(scripts / "generate.py"),
        "--pp",
        str(a.pp),
        "--cfg",
        str(a.cfg),
        "--mode",
        mode,
        "--split",
        split,
        "--out",
        str(out),
        *map(str, extra),
    ]
    with (out / "run.log").open("w") as f:
        f.write(json.dumps({"command": cmd, "start": time.time()}) + "\n")
        f.flush()
        subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=6 * 3600)
    return out


def compare(out):
    subprocess.run([sys.executable, str(scripts / "compare.py"), str(out / "none"), str(out / "cache")], check=True)
    return json.loads((out / "cache/comparison.json").read_text())


def quality_passed(metric):
    return (
        metric["mean_ssim"] >= 0.95 and metric["min_video_ssim"] >= 0.9 and metric["mean_temporal_error_ratio"] <= 0.1
    )


def measure_candidate(name, split, threshold, warmup=0):
    extra = ("--paired", "--threshold", threshold, "--cache-warmup-steps", warmup)
    detailed = run(name, "cache", split, extra, coeff)
    metrics = compare(detailed)
    subprocess.run(
        [sys.executable, str(scripts / "audit.py"), str(detailed), "--pp", str(a.pp), "--cfg", str(a.cfg)], check=True
    )
    audit = json.loads((detailed / "audit.json").read_text())
    metrics["cache_use_passed"] = audit["all_branches_skipped_blocks"]
    if not quality_passed(metrics) or not metrics["cache_use_passed"]:
        metrics["passed"] = False
        metrics["timing_source"] = "diagnostic trace-inclusive run; formal timing not performed"
        (detailed / "cache/comparison.json").write_text(json.dumps(metrics, indent=2))
        return metrics, False
    minimal = run(name + "-timing", "cache", split, (*extra, "--trace-detail", "minimal"), coeff)
    timing = compare(minimal)
    assert (detailed / "production-profile.json").read_bytes() == (minimal / "production-profile.json").read_bytes()
    left, right = (json.loads((path / "provenance.json").read_text()) for path in (detailed, minimal))
    assert left["sha"] == right["sha"] and left["source_hashes"] == right["source_hashes"]
    # Keep trace-inclusive timing as diagnostic data; acceptance uses the separate,
    # symmetric minimal-observation run of this exact production profile.
    metrics["diagnostic_latency_ratio"] = metrics["latency_ratio"]
    metrics["latency_ratio"] = timing["latency_ratio"]
    metrics["timing_source"] = str(minimal / "cache/comparison.json")
    metrics["timing_quality_passed"] = quality_passed(timing)
    metrics["passed"] = quality_passed(metrics) and quality_passed(timing) and timing["latency_ratio"] <= 0.9
    (detailed / "cache/comparison.json").write_text(json.dumps(metrics, indent=2))
    return metrics, metrics["passed"]


coeff = root / f"coefficients-pp{a.pp}.json"
if a.phase == "calibrate":
    out = run("collect", "collect", "calibration")
    subprocess.run([sys.executable, str(scripts / "fit.py"), str(out / "trace" / "collect"), str(coeff)], check=True)
    candidates = []
    # Include the historical PP=2 warmup strategy, but select using only the
    # complete calibration set and the unchanged gates.
    for warmup in (0, 30):
        for threshold in (0.025, 0.05, 0.07, 0.1, 0.2):
            metric, passed = measure_candidate(f"warmup-{warmup}/tune-{threshold}", "calibration", threshold, warmup)
            if passed:
                candidates.append((metric["latency_ratio"], threshold, warmup))
    if not candidates:
        raise RuntimeError(
            "No calibration candidate met all quality/cache-use/performance gates; do not run held-out acceptance"
        )
    ratio, threshold, warmup = min(candidates)
    (root / f"selection-pp{a.pp}.json").write_text(
        json.dumps(
            {
                "threshold": threshold,
                "warmup_steps": warmup,
                "calibration_latency_ratio": ratio,
                "calibration_all_gates_passed": True,
                "diagnostic_only": False,
            },
            indent=2,
        )
    )
else:
    selection = json.loads((root / f"selection-pp{a.pp}.json").read_text())
    if not selection.get("calibration_all_gates_passed") or selection.get("diagnostic_only", True):
        raise RuntimeError("Independent acceptance requires an all-gates-passing frozen calibration candidate")
    metric, passed = measure_candidate(
        "held-out", "validation", selection["threshold"], selection.get("warmup_steps", 0)
    )
    subprocess.run(
        [
            sys.executable,
            str(scripts / "quality_junit.py"),
            str(a.root / "held-out/cache/comparison.json"),
            str(a.root / "quality.xml"),
        ],
        check=True,
    )
    if not passed:
        raise RuntimeError("Independent acceptance failed; retain the failed gates and keep the PR experimental")
