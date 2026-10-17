# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU evaluator for both CUDA- and NPU-generated artifact populations."""

from __future__ import annotations

import os
import subprocess
from itertools import combinations
from pathlib import Path

import pytest

from benchmarks.lingbot_world.encoder_inference import verify_background_coverage
from benchmarks.lingbot_world.regression import (
    RegressionError,
    check_binary_activity,
    check_quality,
    check_video_precision,
    read_json,
    sha256,
    verify_environment,
    verify_quality_population_contract,
    verify_reference_calibration_contract,
    write_json,
)
from benchmarks.lingbot_world.video_quality import compare_video_pair_in_environment, evaluate_video_quality
from tests.dfx.lingbot_world.runner import ROOT, git_revision, hardware_contract
from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_wbench_vbench_and_video_precision(lingbot_settings):
    settings = lingbot_settings
    output = settings["output"]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "quality.json", {"passed": False, "status": "running"})
    generation = settings["generation"]
    if generation is None:
        raise RegressionError("--lingbot-generation must name the downloaded generation artifact directory")
    generation = generation.resolve()
    manifest = read_json(generation / "generation.json")
    if type(manifest.get("passed")) is not bool or manifest.get("contract", {}).get("profile") != read_json(
        settings["profile_path"]
    ):
        raise RegressionError("Generation is incomplete or belongs to a different profile")
    if manifest["candidate_revision"] != git_revision(ROOT):
        raise RegressionError("Generation artifacts belong to a different candidate commit")
    videos = {}
    for index, case in enumerate(settings["cases"]):
        for role in ("reference", "candidate"):
            for repeat in range(3):
                videos[f"case{index}_{role}{repeat}"] = generation / role / str(repeat) / case["id"] / "rollout.mp4"
    required = {str(path.relative_to(generation)) for path in videos.values()}
    if not required.issubset(manifest["artifacts"]):
        raise RegressionError("Unhashed generation videos; complete quality populations are required")
    for relative, digest in manifest["artifacts"].items():
        if sha256(generation / relative) != digest:
            raise RegressionError(f"Generation artifact checksum mismatch: {relative}")
    limits = read_json(ROOT / "tests/dfx/lingbot_world/quality.json")
    candidate_tolerances = limits["candidate_tolerances"][settings["platform"]]
    reference_spreads = limits["max_reference_spreads"][settings["platform"]]
    pixel_budgets = limits["pixel_budgets"][settings["platform"]]
    write_json(output / "quality-policy.json", limits)
    write_json(output / "profile.json", manifest["contract"]["profile"])
    calibration_errors = []
    if settings["platform"] == "cuda":
        calibration_path = ROOT / "tests/dfx/lingbot_world/calibration/cuda_nightly_reference.json"
        calibration = read_json(calibration_path)
        write_json(output / "reference-calibration.json", calibration)
        try:
            verify_reference_calibration_contract(manifest["contract"], calibration)
        except RegressionError as error:
            calibration_errors.append(
                {
                    "record": str(calibration_path.relative_to(ROOT)),
                    "sha256": sha256(calibration_path),
                    "error": str(error),
                }
            )
    lock = read_json(ROOT / "tests/dfx/lingbot_world/evaluators.lock.json")
    evaluator_root = Path(os.environ["LINGBOT_EVALUATOR_ROOT"]).resolve()
    python = Path(os.environ["LINGBOT_EVALUATOR_PYTHON"])
    packages = verify_environment(python, ROOT / "requirements/lingbot-quality.txt")
    population = None
    population_path = None
    population_files = {40: "cuda_nightly_quality_population.json", 600: "cuda_long_quality_population.json"}
    if settings["platform"] == "cuda" and settings["num_chunks"] in population_files:
        population_path = ROOT / "tests/dfx/lingbot_world/calibration" / population_files[settings["num_chunks"]]
        population = read_json(population_path)
        write_json(output / "quality-reference-population.json", population)
        try:
            verify_quality_population_contract(
                population,
                manifest["contract"],
                evaluator_packages=packages,
                evaluator_lock_sha256=sha256(ROOT / "tests/dfx/lingbot_world/evaluators.lock.json"),
                asset_lock_sha256=sha256(ROOT / "tests/dfx/lingbot_world/assets.lock.json"),
            )
        except RegressionError as error:
            calibration_errors.append(
                {
                    "record": str(population_path.relative_to(ROOT)),
                    "sha256": sha256(population_path),
                    "error": str(error),
                }
            )
            population = None
        else:
            reference_spreads = population["max_reference_spreads"]
    for relative, digest in lock["files"].items():
        if sha256(evaluator_root / relative) != digest:
            raise RegressionError(f"Evaluator weight checksum mismatch: {relative}")
    for suite in ("vbench", "wbench"):
        if git_revision(evaluator_root / suite) != lock["repositories"][suite]["revision"]:
            raise RegressionError(f"Wrong {suite} evaluator revision")
        subprocess.run(
            ["git", "-C", str(evaluator_root / suite), "diff", "--exit-code", "HEAD", "--", ".", ":(exclude).claude"],
            check=True,
            timeout=30,
        )
    os.environ["VBENCH_CACHE_DIR"] = str(evaluator_root / "vbench-cache")
    os.environ["WBENCH_WEIGHTS_DIR"] = str(evaluator_root / "wbench/weights")
    # No model downloads or evaluator backend changes during scoring.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    report = {
        "passed": False,
        "generation_passed": manifest["passed"],
        "quality_passed": False,
        "generation": str(generation),
        "generation_manifest_sha256": sha256(generation / "generation.json"),
        "generation_platform": settings["platform"],
        "quality_policy_sha256": sha256(ROOT / "tests/dfx/lingbot_world/quality.json"),
        "suites": {},
        "pixels": {},
        "calibration_errors": calibration_errors,
        "evaluator_packages": packages,
        "evaluator_hardware": hardware_contract(
            {**settings, "platform": "cuda", "cards": 1, "dies": 1, "reserved_cards": 1}
        ),
        "evaluator_lock_sha256": sha256(ROOT / "tests/dfx/lingbot_world/evaluators.lock.json"),
    }
    if population_path is not None:
        report["reference_population_sha256"] = sha256(population_path)
    # A forced job timeout can bypass finally; retain provenance before workers start.
    write_json(output / "quality.json", report)
    try:
        evaluations = {}
        for suite in ("vbench", "wbench"):
            tolerances = candidate_tolerances[suite]
            evaluation = evaluate_video_quality(
                videos,
                output / suite,
                suite=suite,
                repo=evaluator_root / suite,
                python=python,
                dimensions=list(tolerances),
                timeout_s=limits["timeout_s"],
            )
            write_json(output / f"{suite}-raw.json", evaluation)
            if evaluation["status"] != "passed":
                raise RegressionError(f"{suite} evaluation failed: {evaluation['errors']}")
            if suite == "wbench" and evaluation["protocol"].get("dynamic_degree_backend") != "raft":
                raise RegressionError("WBench must use the pinned RAFT backend; fallback scores are not comparable")
            evaluations[suite] = evaluation
        # Retain both native score populations even when one reference
        # calibration fails. Every failed calibration still blocks acceptance.
        for suite, evaluation in evaluations.items():
            if suite == "vbench":
                expected_frames = (settings["num_chunks"] * 3 - 1) * 4 + 1
                for label in videos:
                    stats = evaluation["native_provenance"].get(label, {}).get("background_clip_inference", {})
                    try:
                        verify_background_coverage(stats, expected_frames)
                    except ValueError as error:
                        raise RegressionError(f"{label}: {error}") from error
            binary_dimensions = limits["binary_activity_dimensions"]
            tolerances = {
                name: value for name, value in candidate_tolerances[suite].items() if name not in binary_dimensions
            }
            cases = {}
            for index, case in enumerate(settings["cases"]):
                if calibration_errors:
                    cases[case["id"]] = [
                        {"passed": False, "status": "unjudged", "calibration_errors": calibration_errors}
                    ]
                    continue
                reference_scores = [evaluation["scores"][f"case{index}_reference{repeat}"] for repeat in range(3)]
                reference = [{name: row[name] for name in tolerances} for row in reference_scores]
                checks = []
                quality_reference = reference
                if population is not None:
                    quality_reference = population["cases"][case["id"]]["suites"][suite]
                    # Keep the fresh cohort's spread gate, then reject fresh
                    # references that degrade beyond the reviewed population.
                    for repeat, measured in enumerate(reference):
                        try:
                            check_quality(
                                reference,
                                measured,
                                tolerances=tolerances,
                                max_reference_spreads=reference_spreads[suite],
                            )
                            check = check_quality(
                                quality_reference,
                                measured,
                                tolerances=tolerances,
                                max_reference_spreads=reference_spreads[suite],
                            )
                        except RegressionError as error:
                            check = {"passed": False, "error": str(error)}
                        checks.append({"reference_repeat": repeat, **check})
                for dimension in binary_dimensions:
                    try:
                        check = check_binary_activity(
                            [row[dimension] for row in reference_scores],
                            [evaluation["scores"][f"case{index}_candidate{repeat}"][dimension] for repeat in range(3)],
                            tolerance=candidate_tolerances[suite][dimension],
                        )
                    except RegressionError as error:
                        check = {"passed": False, "error": str(error)}
                    checks.append({"dimension": dimension, **check})
                for repeat in range(3):
                    try:
                        check = check_quality(
                            quality_reference,
                            {name: evaluation["scores"][f"case{index}_candidate{repeat}"][name] for name in tolerances},
                            tolerances=tolerances,
                            max_reference_spreads=reference_spreads[suite],
                        )
                    except RegressionError as error:
                        check = {"passed": False, "error": str(error)}
                    checks.append({"candidate_repeat": repeat, **check})
                cases[case["id"]] = checks
            report["suites"][suite] = cases
        for case in settings["cases"]:

            def directory(role, repeat):
                return generation / role / str(repeat) / case["id"] / "rollout.mp4"

            # Pixel divergence is calibrated against independent same-code
            # repeats. It is distinct from the fixed-input VAE precision gate.
            reference_pairs = list(combinations(range(3), 2))
            reference = [
                compare_video_pair_in_environment(
                    directory("reference", first),
                    directory("reference", second),
                    output / "pixels",
                    python=python,
                    expected_packages=packages,
                    window_seconds=limits["pixel_window_seconds"],
                    ssim_stride=24,
                    timeout_s=limits["timeout_s"],
                )
                for first, second in reference_pairs
            ]
            candidate = [
                compare_video_pair_in_environment(
                    directory("reference", 0),
                    directory("candidate", repeat),
                    output / "pixels",
                    python=python,
                    expected_packages=packages,
                    window_seconds=limits["pixel_window_seconds"],
                    ssim_stride=24,
                    timeout_s=limits["timeout_s"],
                )
                for repeat in range(3)
            ]
            quality_reference = reference if population is None else population["cases"][case["id"]]["pixels"]
            for row in [*reference, *candidate]:
                metadata = row["reference"]
                expected_frames = (settings["num_chunks"] * 3 - 1) * 4 + 1
                if (metadata["frame_count"], metadata["width"], metadata["height"], metadata["fps"]) != (
                    expected_frames,
                    settings["width"],
                    settings["height"],
                    settings["mux_fps"],
                ):
                    raise RegressionError("Decoded video does not match the pinned workload")

            def check_pixel(row):
                if calibration_errors:
                    return {"passed": False, "status": "unjudged", "calibration_errors": calibration_errors}
                return check_video_precision(
                    quality_reference,
                    row,
                    psnr_drop_db=pixel_budgets["psnr_drop_db"],
                    min_psnr_db=limits["min_psnr_db"],
                    ssim_drop=pixel_budgets["ssim_drop"],
                    reject_uniform_frames=limits["reject_uniform_frames"],
                )

            report["pixels"][case["id"]] = {
                "reference_pairs": reference_pairs,
                "reference_repeat": reference,
                "reference_checks": [] if population is None else [check_pixel(row) for row in reference],
                "candidate": candidate,
                "checks": [check_pixel(row) for row in candidate],
            }
        report["quality_passed"] = (
            not calibration_errors
            and all(
                check["passed"] for suite in report["suites"].values() for checks in suite.values() for check in checks
            )
            and all(
                check["passed"]
                for case in report["pixels"].values()
                for check in [*case["reference_checks"], *case["checks"]]
            )
        )
        report["passed"] = report["generation_passed"] and report["quality_passed"]
    finally:
        write_json(output / "quality.json", report)
    assert report["passed"], f"Quality regression; see {output}/quality.json"
