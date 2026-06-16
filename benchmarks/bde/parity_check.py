# SPDX-License-Identifier: Apache-2.0
"""Tier-A numerical parity check for weighted DreamZero: KV-off vs KV-on.

The model run is delegated to the existing offline export script; this module
orchestrates the two runs, loads the saved videos, computes metrics, saves
artifacts (side-by-side, diff heatmap, metrics plot, run.json), and gates on
PSNR/SSIM/LPIPS.

    python -m benchmarks.bde.parity_check \\
        --deploy-off vllm_omni/deploy/dreamzero.yaml \\
        --deploy-on  vllm_omni/deploy/dreamzero_bde.yaml \\
        --input-video <dir> --session-id parity0 --out artifacts/dreamzero/parity

``compare()`` is the model-free core (load -> metrics -> artifacts -> gates) and
is unit-tested; ``run_export()`` shells out to the weighted model.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from benchmarks.bde import artifacts, metrics

EXPORT_SCRIPT = "examples/offline_inference/dreamzero/export_prediction_video.py"


def compare(
    off_video,
    on_video,
    out_dir,
    *,
    fps: int = 5,
    label: str = "parity",
    real_video=None,
    run_meta: dict | None = None,
) -> dict:
    """Load KV-off / KV-on videos, compute metrics, save artifacts, gate."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    off = artifacts.load_mp4(off_video)
    on = artifacts.load_mp4(on_video)
    frame_metrics = metrics.frame_metrics(off, on)
    gates = metrics.check_gates(frame_metrics)

    panes = [v for v in (artifacts.load_mp4(real_video) if real_video else None, off, on) if v is not None]
    artifacts.save_mp4(out / f"{label}_sidebyside.mp4", artifacts.side_by_side(*panes), fps=fps)
    artifacts.save_mp4(out / f"{label}_diff.mp4", artifacts.diff_heatmap(off, on), fps=fps)
    artifacts.metrics_plot(out / f"{label}_metrics.png", frame_metrics)
    artifacts.write_run_json(
        out / "run.json",
        label=label,
        metrics=frame_metrics,
        gates=gates,
        meta=run_meta or {},
    )
    return {"metrics": frame_metrics, "gates": gates, "out_dir": str(out)}


def run_export(model: str, deploy_config, session_id: str, out_stem, *, extra=()) -> Path:
    """Run the offline DreamZero export (saves the generated video). Needs weights."""
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, EXPORT_SCRIPT,
        "--model", model,
        "--deploy-config", str(deploy_config),
        "--session-id", session_id,
        "--output-stem", str(out_stem),
        "--save-gif", "--save-input-video", "--save-actions",
        *extra,
    ]
    subprocess.run(cmd, check=True)
    return out_stem.with_suffix(".mp4")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DreamZero BDE KV parity check (KV-off vs KV-on)")
    ap.add_argument("--model", default="GEAR-Dreams/DreamZero-DROID")
    ap.add_argument("--deploy-off", required=True, help="deploy config with BDE KV disabled (model-local)")
    ap.add_argument("--deploy-on", required=True, help="deploy config with BDE KV enabled")
    ap.add_argument("--session-id", default="parity0")
    ap.add_argument("--out", default="artifacts/dreamzero/parity")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--export-arg", action="append", default=[], help="extra arg forwarded to export")
    args = ap.parse_args(argv)

    base = Path(args.out)
    sid = args.session_id
    off = run_export(args.model, args.deploy_off, sid, base / "off" / sid / "pred", extra=args.export_arg)
    on = run_export(args.model, args.deploy_on, sid, base / "on" / sid / "pred", extra=args.export_arg)

    result = compare(
        off,
        on,
        base / "compare",
        fps=args.fps,
        label=sid,
        run_meta={
            "model": args.model,
            "deploy_off": args.deploy_off,
            "deploy_on": args.deploy_on,
            "session_id": sid,
        },
    )
    m, g = result["metrics"], result["gates"]
    status = "PASS" if g["passed"] else "FAIL"
    print(
        f"[parity] {status}  PSNR_min={m['psnr_min']:.2f}  "
        f"SSIM_min={m['ssim_min']:.4f}  max_abs_diff={m['max_abs_diff']:.4f}"
    )
    print(f"[parity] artifacts: {result['out_dir']}")
    return 0 if g["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
