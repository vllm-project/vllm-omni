# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent judgments over actual current-run reference/candidate artifacts."""

from __future__ import annotations

import importlib.metadata
from itertools import combinations
from typing import Any

from benchmarks.lingbot_world.precision import compare_tensor_samples
from benchmarks.lingbot_world.regression import (
    RegressionError,
    check_video_precision,
    read_json,
    write_json,
)
from benchmarks.lingbot_world.video_quality import compare_video_pair
from tests.e2e.accuracy.lingbot_world.runner import ROOT, checked_artifact
from vllm_omni.benchmarks.realtime_video.workload import num_frames_for_chunks


def record_accuracy_result(settings, data, suite, report):
    """Keep generation identity separate from the machine running the scorer."""
    from tests.e2e.accuracy import helpers

    _, manifest = data
    policy = read_json(ROOT / "tests/e2e/accuracy/lingbot_world/quality.json")
    return helpers.write_accuracy_result(
        model_id=read_json(ROOT / "tests/e2e/accuracy/lingbot_world/assets.lock.json")["model_id"],
        dataset=f"lingbot_world_{suite}",
        metrics=report,
        baseline={settings["hardware"]: policy["pixel_budgets"][settings["platform"]]}
        if suite == "precision"
        else {settings["hardware"]: policy["candidate_tolerances"][settings["platform"]][suite]},
        result_dir=settings["output"] / "results",
        extra={
            "Hardware": settings["hardware"],
            "generation_platform": settings["platform"],
            "generation_hardware": manifest["contract"]["hardware"],
            "evaluator_hardware": helpers._accuracy_hardware_label(),
            "candidate_revision": manifest["candidate_revision"],
            "reference_revision": manifest["reference_revision"],
            "model_revision": settings["model_revision"],
            "workload": manifest["contract"]["profile"],
        },
    )


def numeric_report(settings, data):
    root, manifest = data
    report: dict[str, Any] = {
        "passed": False,
        "tensors": [],
        "pixels": {},
        "errors": [],
        "decoder_packages": {name: importlib.metadata.version(name) for name in ("numpy", "opencv-python")},
    }
    policy = read_json(ROOT / "tests/e2e/accuracy/lingbot_world/quality.json")
    try:
        for role in ("reference", "candidate"):
            for repeat in (1, 2):
                try:
                    report["tensors"].append(
                        compare_tensor_samples(
                            checked_artifact(root, manifest, f"{role}/0/precision.npz"),
                            checked_artifact(root, manifest, f"{role}/{repeat}/precision.npz"),
                            atol=0,
                            rtol=0,
                        )
                    )
                except Exception as error:
                    report["errors"].append({"tensor": f"{role}/{repeat}", "error": str(error)})
        for repeat in range(3):
            try:
                report["tensors"].append(
                    compare_tensor_samples(
                        checked_artifact(root, manifest, "reference/0/precision.npz"),
                        checked_artifact(root, manifest, f"candidate/{repeat}/precision.npz"),
                        atol=settings["precision_atol"],
                        rtol=settings["precision_rtol"],
                    )
                )
            except Exception as error:
                report["errors"].append({"tensor": f"candidate/{repeat}", "error": str(error)})
        for case in settings["cases"]:
            try:

                def video(role, repeat):
                    return checked_artifact(root, manifest, f"{role}/{repeat}/{case['id']}/rollout.mp4")

                def compare(first, second):
                    row = compare_video_pair(
                        first, second, window_seconds=policy["pixel_window_seconds"], ssim_stride=24
                    )
                    expected = (
                        num_frames_for_chunks(settings["num_chunks"]),
                        settings["width"],
                        settings["height"],
                        settings["mux_fps"],
                    )
                    for key in ("reference", "candidate"):
                        metadata = row[key]
                        if tuple(metadata[x] for x in ("frame_count", "width", "height", "fps")) != expected:
                            raise RegressionError("Decoded video does not match the workload")
                    return row

                reference = [
                    compare(video("reference", a), video("reference", b)) for a, b in combinations(range(3), 2)
                ]
                candidates = [compare(video("reference", 0), video("candidate", r)) for r in range(3)]
                report["pixels"][case["id"]] = {
                    "reference": reference,
                    "candidate": candidates,
                    "checks": [
                        check_video_precision(
                            reference,
                            row,
                            **policy["pixel_budgets"][settings["platform"]],
                            min_psnr_db=policy["min_psnr_db"],
                            reject_uniform_frames=policy["reject_uniform_frames"],
                        )
                        for row in candidates
                    ],
                }
            except Exception as error:
                report["errors"].append({"case": case["id"], "error": str(error)})
        report["passed"] = (
            not report["errors"]
            and len(report["tensors"]) == 7
            and all(row["status"] == "passed" for row in report["tensors"])
            and len(report["pixels"]) == len(settings["cases"])
            and all(c["passed"] for row in report["pixels"].values() for c in row["checks"])
        )
    finally:
        write_json(settings["output"] / "precision.json", report)
    return report
