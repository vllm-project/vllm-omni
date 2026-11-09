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
    calibrate,
    check_performance,
    check_video_precision,
    read_json,
    write_json,
)
from benchmarks.lingbot_world.video_quality import compare_video_pair
from tests.dfx.lingbot_world.runner import ROOT, _measured_metrics, checked_artifact


def performance_report(settings, data):
    root, manifest = data
    contract = manifest["contract"]
    report: dict[str, Any] = {"passed": False, "cases": {}, "errors": []}
    request_ids: set[str] = set()
    try:
        for case in settings["cases"]:
            metrics: dict[str, list[dict[str, Any]]] = {"reference": [], "candidate": []}
            try:
                for role in metrics:
                    for repeat in range(3):
                        raw = read_json(
                            checked_artifact(root, manifest, f"{role}/{repeat}/{case['id']}/benchmark.json")
                        )
                        expected_server = (
                            manifest["reference_revision"] if role == "reference" else manifest["candidate_revision"]
                        )
                        if (
                            raw.get("server_revision") != expected_server
                            or raw.get("git_revision") != manifest["candidate_revision"]
                        ):
                            raise RegressionError("Wrong benchmark client/server source")
                        measured = _measured_metrics(raw, settings)
                        ids = {session["request_id"] for session in [*raw["sessions"], *raw["warmups"]]}
                        if ids & request_ids:
                            raise RegressionError("Requests reused across independent measurements")
                        request_ids.update(ids)
                        metrics[role].append(measured)
                baseline = calibrate(
                    metrics["reference"],
                    contract=contract,
                    reference_revision=settings["reference_revision"],
                    max_cv=settings["max_reference_cv"],
                    regression_fraction=settings["performance_regression_fraction"],
                    transient_noise_sigma=settings["transient_noise_sigma"],
                )
                report["cases"][case["id"]] = {
                    "baseline": baseline,
                    "checks": [check_performance(row, baseline, contract=contract) for row in metrics["candidate"]],
                }
            except (RegressionError, OSError, KeyError, ValueError) as error:
                report["errors"].append({"case": case["id"], "error": str(error)})
        report["passed"] = (
            not report["errors"]
            and bool(report["cases"])
            and all(check["passed"] for case in report["cases"].values() for check in case["checks"])
        )
    finally:
        write_json(settings["output"] / "performance.json", report)
    return report


def numeric_report(settings, data):
    root, manifest = data
    report: dict[str, Any] = {
        "passed": False,
        "tensors": [],
        "pixels": {},
        "errors": [],
        "decoder_packages": {name: importlib.metadata.version(name) for name in ("numpy", "opencv-python")},
    }
    policy = read_json(ROOT / "tests/dfx/lingbot_world/quality.json")
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
                        (settings["num_chunks"] * 3 - 1) * 4 + 1,
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
