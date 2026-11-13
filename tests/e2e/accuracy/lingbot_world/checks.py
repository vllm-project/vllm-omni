# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run packaged benchmark scoring, apply profile score floors and publish the accuracy record.

WBench and VBench scores are compared with the profile's ``score_floors``.
Numerical precision has no stored reference tensors, so tensor error, PSNR and
SSIM are not computed; that suite checks a complete, finite fixed-input replay.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from tests.e2e.accuracy.lingbot_world.runner import ROOT
from vllm_omni.benchmarks.realtime_video.eval import read_json, write_json


def run_accuracy(settings, data, suite):
    from tests.e2e.accuracy import helpers
    from vllm_omni.benchmarks.realtime_video.serving import build_serve_args
    from vllm_omni.benchmarks.serve import main as bench_serve

    root, manifest = data
    report = {
        "passed": False,
        "status": "failed",
        "suite": suite,
        "measurements": {},
        "errors": [],
    }
    try:
        arguments = [
            "--model",
            read_json(ROOT / "tests/e2e/accuracy/lingbot_world/assets.lock.json")["model_id"],
            "--realtime-video-evaluate",
            suite,
            "--realtime-video-candidate",
            str(root / "generation.json"),
            "--realtime-video-eval-output-dir",
            str(settings["output"] / suite),
            "--save-result",
            "--result-dir",
            str(settings["output"] / "results"),
            "--result-filename",
            f"bench-{suite}.json",
        ]
        if suite != "precision":
            from tests.e2e.accuracy.lingbot_world.prepare import evaluator_cache_key, prepare_evaluators

            evaluator_root = Path(
                os.environ.get(
                    "LINGBOT_EVALUATOR_ROOT",
                    str(Path(os.environ.get("HF_HOME", "/tmp")) / f"lingbot-evaluators-{evaluator_cache_key()}"),
                )
            ).resolve()
            python = (
                Path(os.environ["LINGBOT_EVALUATOR_PYTHON"])
                if "LINGBOT_EVALUATOR_PYTHON" in os.environ
                else prepare_evaluators(evaluator_root)
            )
            arguments.extend(
                [
                    "--realtime-video-evaluator-root",
                    str(evaluator_root),
                    "--realtime-video-evaluator-python",
                    str(python),
                    "--realtime-video-evaluator-requirements",
                    str(ROOT / "requirements/lingbot-quality.txt"),
                ]
            )
        report = bench_serve(build_serve_args(arguments))["realtime_video_evaluation"]
    except Exception as error:
        # Keep this suite's failure visible without consuming another test's
        # verdict or hiding its usable candidate data.
        report["errors"].append(str(error))
        print(f"LingBot {suite} measurement failed: {error}", flush=True)
    measurements = report["measurements"]
    if suite == "precision":
        metrics = {
            f"{name.replace('/', '_')}_{statistic}": row[statistic]
            for name, row in measurements.get("replay", {}).get("tensors", {}).items()
            for statistic in ("minimum", "maximum", "mean", "std")
        }
    else:
        metrics = {
            f"{case}_{dimension}": value
            for case, dimensions in measurements.get("scores", {}).items()
            for dimension, value in dimensions.items()
        }
    floors = {} if suite == "precision" else settings["score_floors"][suite]
    # bench_serve returns a fresh report, so the status is set after measurement.
    report["comparison_status"] = "finite_only" if suite == "precision" else "score_floor"
    if floors:
        report["floor_violations"] = [
            f"{name}: {metrics.get(name)!r} < {floor}"
            for name, floor in floors.items()
            if not (isinstance(metrics.get(name), (int, float)) and math.isfinite(metrics[name]))
            or metrics[name] < floor
        ]
        if report["floor_violations"]:
            report["passed"] = False
            report["errors"].extend(f"Score below floor: {row}" for row in report["floor_violations"])
    write_json(settings["output"] / f"{suite}.json", report)
    helpers.write_accuracy_result(
        model_id=read_json(ROOT / "tests/e2e/accuracy/lingbot_world/assets.lock.json")["model_id"],
        dataset=f"lingbot_world_{suite}",
        metrics=metrics,
        baseline={settings["hardware"]: floors} if floors else {},
        result_dir=settings["output"] / "results",
        extra={
            "Hardware": settings["hardware"],
            "generation_platform": settings["platform"],
            "generation_hardware": manifest["hardware"],
            "evaluator_hardware": helpers._accuracy_hardware_label(),
            "candidate_revision": manifest["candidate_revision"],
            "model_revision": settings["model_revision"],
            "workload": manifest["contract"]["profile"],
            "comparison_status": report["comparison_status"],
            "measurement_status": report["status"],
            "measurement_report": report,
        },
    )
    if floors:
        violations = report["floor_violations"]
        print(
            f"LingBot {suite}: {report['status']}; {len(violations)} of {len(floors)} score floors failed", flush=True
        )
    else:
        print(f"LingBot {suite}: {report['status']}; finite replay only, no stored reference tensors", flush=True)
    return report
