# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Freeze a calibration-only choice, retaining failures as diagnostic candidates."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audit import audit_run  # noqa: E402

CALIBRATION_SEEDS = (17, 29)


def calibration_manifest(inputs: Path) -> set[tuple[str, str, int]]:
    """The complete (name, prompt, seed) set generate.py runs for calibration."""
    prompts = json.loads((inputs / "prompts.json").read_text())["calibration"]
    return {(f"{i:02d}-{seed}", prompt, seed) for i, prompt in enumerate(prompts) for seed in CALIBRATION_SEEDS}


def acceptance_receipt(run_dir: Path, metrics: dict, run: dict, pp: int, cfg: int, manifest: set) -> dict | None:
    """Return the evidence that lets a candidate qualify, or None if any is missing.

    Requires the complete calibration manifest at the requested topology and a
    matched minimal-trace timing run of the same source and production profile.
    """
    records = json.loads((run_dir / "cache/results.json").read_text())["records"]
    if run.get("limit") or run["cfg"] != cfg or run["pp"] != pp:
        return None
    if {(r["name"], r["prompt"], r["seed"]) for r in records} != manifest or len(records) != len(manifest):
        return None
    timing_source = metrics.get("timing_source")
    if not timing_source or not Path(timing_source).is_file():
        return None
    timing_dir = Path(timing_source).parent.parent
    timing = json.loads(Path(timing_source).read_text())
    timing_run = json.loads((timing_dir / "cache/results.json").read_text())
    detailed_provenance = json.loads((run_dir / "provenance.json").read_text())
    timing_provenance = json.loads((timing_dir / "provenance.json").read_text())
    timing_records = {(r["name"], r["prompt"], r["seed"]) for r in timing_run["records"]}
    if (
        timing_provenance.get("trace_detail") != "minimal"
        or timing_provenance["sha"] != detailed_provenance["sha"]
        or timing_provenance["source_hashes"] != detailed_provenance["source_hashes"]
        or (timing_dir / "production-profile.json").read_bytes() != (run_dir / "production-profile.json").read_bytes()
        or timing_run["args"]["pp"] != pp
        or timing_run["args"]["cfg"] != cfg
        or timing_records != manifest
        or timing["latency_ratio"] != metrics["latency_ratio"]
        or not timing["passed"]
    ):
        return None
    return {
        "source_sha": detailed_provenance["sha"],
        "calibration_requests": len(manifest),
        "timing_source": str(timing_source),
        "profile_sha256": hashlib.sha256((run_dir / "production-profile.json").read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--cfg", type=int, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("campaigns", type=Path, nargs="+")
    args = parser.parse_args()
    candidates = []
    manifest = calibration_manifest(args.inputs)
    for campaign in args.campaigns:
        for path in sorted(campaign.rglob("comparison.json")):
            if path.parent.name != "cache" or path.parent.parent.name.endswith("-timing"):
                continue
            run_dir = path.parent.parent
            metrics = json.loads(path.read_text())
            run = json.loads((path.parent / "results.json").read_text())["args"]
            assert run["split"] == "calibration", "Held-out results must never select a profile"
            assert run["pp"] == args.pp
            try:
                audit = audit_run(run_dir, args.pp, run["cfg"])
            except AssertionError:
                continue
            if not audit["all_branches_skipped_blocks"]:
                continue
            score = max(
                0.95 / max(metrics["mean_ssim"], 1e-8),
                0.90 / max(metrics["min_video_ssim"], 1e-8),
                metrics["mean_temporal_error_ratio"] / 0.10,
                metrics["latency_ratio"] / 0.90,
            )
            receipt = acceptance_receipt(run_dir, metrics, run, args.pp, args.cfg, manifest)
            candidates.append(
                {
                    "threshold": run["threshold"],
                    "warmup_steps": run.get("cache_warmup_steps", 0),
                    "source": str(path),
                    "metrics": metrics,
                    "gate_score": score,
                    "qualified": bool(metrics.get("passed")) and receipt is not None,
                    "receipt": receipt,
                }
            )
    assert candidates, "No candidate actually skipped blocks in every stage/branch"
    qualified = [row for row in candidates if row["qualified"]]
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
