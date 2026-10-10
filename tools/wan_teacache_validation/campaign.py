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


coeff = root / f"coefficients-pp{a.pp}.json"
if a.phase == "calibrate":
    out = run("collect", "collect", "calibration")
    subprocess.run([sys.executable, str(scripts / "fit.py"), str(out / "trace" / "collect"), str(coeff)], check=True)
    candidates = []
    for threshold in (0.025, 0.05, 0.1, 0.2):
        out = run(
            f"tune-{threshold}", "cache", "calibration", ("--paired", "--limit", "6", "--threshold", threshold), coeff
        )
        subprocess.run([sys.executable, str(scripts / "compare.py"), str(out / "none"), str(out / "cache")], check=True)
        metric = json.loads((out / "cache/comparison.json").read_text())
        if (
            metric["mean_ssim"] >= 0.95
            and metric["min_video_ssim"] >= 0.9
            and metric["mean_temporal_error_ratio"] <= 0.1
        ):
            candidates.append((metric["latency_ratio"], threshold))
    if not candidates:
        raise RuntimeError("No calibration candidate met quality criteria; do not run held-out acceptance")
    ratio, threshold = min(candidates)
    (root / f"selection-pp{a.pp}.json").write_text(
        json.dumps({"threshold": threshold, "calibration_latency_ratio": ratio}, indent=2)
    )
else:
    selection = json.loads((root / f"selection-pp{a.pp}.json").read_text())
    threshold = selection["threshold"]
    out = run(
        "held-out",
        "cache",
        "validation",
        ("--paired", "--threshold", threshold, "--cache-warmup-steps", selection.get("warmup_steps", 0)),
        coeff,
    )
    subprocess.run([sys.executable, str(scripts / "compare.py"), str(out / "none"), str(out / "cache")], check=True)
