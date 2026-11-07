# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Freeze reference-only profile calibration; never establish acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from itertools import combinations
from pathlib import Path
from uuid import UUID

from packaging.version import Version

from benchmarks.lingbot_world.encoder_inference import verify_background_coverage
from benchmarks.lingbot_world.regression import (
    RegressionError,
    calibrate,
    check_binary_activity,
    check_performance,
    check_quality,
    check_video_precision,
    contract_digest,
    number,
    read_json,
    sha256,
    verify_environment,
    verify_quality_population_contract,
    verify_reference_calibration_contract,
)
from benchmarks.lingbot_world.video_quality import compare_video_pair_in_environment
from tests.dfx.lingbot_world.prepare import profile_for_case_shard
from tests.dfx.lingbot_world.runner import ROOT, _measured_metrics
from tests.dfx.lingbot_world.runtime_sources import verify_source_attestations

PROTOCOL = ROOT / "tests/dfx/lingbot_world/calibration/v033_reference_protocol.json"
POLICY = ROOT / "tests/dfx/lingbot_world/quality.json"


def _immutable_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _verified_artifacts(root: Path, manifest: dict) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RegressionError("Missing hashed raw artifacts")
    for relative, digest in artifacts.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise RegressionError("Unsafe artifact path")
        try:
            (root / path).resolve().relative_to(root.resolve())
        except ValueError as error:
            raise RegressionError("Artifact escapes its cohort directory") from error
        if sha256(root / path) != digest:
            raise RegressionError(f"Raw artifact checksum mismatch: {relative}")


def _semantic_contract(contract: dict) -> dict:
    result = json.loads(json.dumps(contract))
    # Allocation UUIDs/PCI identifiers describe reservation, not model behavior.
    # Preserve those in the immutable source manifests; match device properties.
    result["hardware"].pop("allocation", None)
    return result


def _collection_identity(manifest: dict, report: dict) -> str:
    keys = ("collection_id", "collection_backend", "buildkite_job_id")
    if any(key not in document for document in (manifest, report) for key in keys):
        raise RegressionError("Reference collection requires explicit collection provenance")
    identity = manifest["collection_id"]
    if not isinstance(identity, str):
        raise RegressionError("Reference collection requires a canonical UUID")
    try:
        canonical = str(UUID(identity))
    except ValueError as error:
        raise RegressionError("Reference collection requires a canonical UUID") from error
    if identity != canonical:
        raise RegressionError("Reference collection requires a canonical UUID")
    backend, job = manifest["collection_backend"], manifest["buildkite_job_id"]
    if backend == "buildkite" and isinstance(job, str) and job:
        job_identity = job
    elif backend == "local" and job is None:
        job_identity = f"local:{identity}"
    else:
        raise RegressionError("Reference collection backend and job provenance are inconsistent")
    if any(report[key] != manifest[key] for key in keys):
        raise RegressionError("Reference quality collection provenance differs from its generation")
    return job_identity


def _measurement_producers(generation: Path, manifest: dict, report: dict) -> dict[int, dict]:
    jobs = manifest.get("measurement_jobs")
    if jobs is None:
        if report.get("measurement_jobs") is not None or any(
            path.startswith("reference/provenance/measurement-") for path in manifest["artifacts"]
        ):
            raise RegressionError("Reference measurement producer provenance is missing or invented")
        return {repeat: manifest for repeat in range(3)}
    if (
        not isinstance(jobs, list)
        or len(jobs) != 3
        or report.get("measurement_jobs") != jobs
        or any(not isinstance(job, dict) for job in jobs)
        or [job.get("repeat") for job in jobs] != list(range(3))
        or any(type(job.get("repeat")) is not int for job in jobs)
    ):
        raise RegressionError("Aggregated references require three ordered measurement producers")
    identities: set[str] = set()
    collection_ids: set[str] = set()
    producers: dict[int, dict] = {}
    for repeat, job in enumerate(jobs):
        identity = _collection_identity(job, job)
        if identity in identities or job["collection_id"] in collection_ids:
            raise RegressionError("Aggregated reference producers must be independent")
        identities.add(identity)
        collection_ids.add(job["collection_id"])
        relative = f"reference/provenance/measurement-{repeat}.json"
        if (
            relative not in manifest["artifacts"]
            or job.get("manifest_path") != relative
            or manifest["artifacts"][relative] != job.get("manifest_sha256")
        ):
            raise RegressionError("Aggregated reference producer requires its hashed source manifest")
        source = read_json(generation / relative)
        _collection_identity(source, job)
        if (
            source.get("passed") is not False
            or source.get("reference_only") is not True
            or source.get("measurement_only") is not True
            or source.get("complete") is not True
            or type(source.get("repeat")) is not int
            or source["repeat"] != repeat
            or any(
                source.get(key) != manifest[key]
                for key in ("reference_revision", "harness_revision", "reference_protocol_sha256")
            )
            or _semantic_contract(source["contract"]) != _semantic_contract(manifest["contract"])
            or source.get("records") != {"reference": [manifest["records"]["reference"][repeat]]}
        ):
            raise RegressionError("Aggregated reference source manifest differs from its cohort")
        expected_payload = {
            path: digest for path, digest in manifest["artifacts"].items() if path.startswith(f"reference/{repeat}/")
        }
        if not expected_payload or source.get("artifacts") != expected_payload:
            raise RegressionError("Aggregated reference source artifacts do not match the retained cohort")
        producers[repeat] = job
    return producers


def _load_cohort(generation: Path, quality: Path, protocol_sha: str, policy: dict) -> dict:
    manifest = read_json(generation / "generation.json")
    report = read_json(quality / "quality.json")
    for document in (manifest, report):
        if (
            document.get("reference_only") is not True
            or document.get("measurement_only") is not True
            or document.get("complete") is not True
            or document.get("passed") is not False
            or document.get("reference_protocol_sha256") != protocol_sha
            or "candidate_revision" in document
        ):
            raise RegressionError("Incomplete, mixed or unfrozen reference-only input")
    if (generation / "candidate").exists() or set(manifest.get("records", {})) != {"reference"}:
        raise RegressionError("Candidate input is forbidden during calibration")
    job = _collection_identity(manifest, report)
    _verified_artifacts(generation, manifest)
    _verified_artifacts(quality, report)
    if any(not path.startswith("reference/") for path in manifest["artifacts"]):
        raise RegressionError("Generation contains non-reference artifacts")
    if (
        report.get("generation_manifest_sha256") != sha256(generation / "generation.json")
        or report.get("contract") != manifest["contract"]
        or report.get("reference_revision") != manifest["reference_revision"]
        or report.get("harness_revision") != manifest["harness_revision"]
        or report.get("quality_policy_sha256") != sha256(POLICY)
        or report.get("evaluator_lock_sha256") != sha256(ROOT / "tests/dfx/lingbot_world/evaluators.lock.json")
        or report.get("quality_passed") is not False
    ):
        raise RegressionError("Quality input does not match its generation and evaluator policy")
    contract = _semantic_contract(manifest["contract"])
    profile = contract["profile"]
    if profile.get("platform") not in ("cuda", "npu") or profile.get("num_chunks") not in (40, 160, 600):
        raise RegressionError("Calibration requires an exact full CUDA/NPU profile")
    level = {40: "nightly", 160: "weekly", 600: "long"}[profile["num_chunks"]]
    configured = ROOT / f"tests/dfx/lingbot_world/configs/{profile['platform']}_{level}.json"
    expected_profile = read_json(configured)
    shard = profile.get("case_shard")
    if shard is not None:
        if (
            not isinstance(shard, dict)
            or set(shard) != {"index", "count", "source_sha256"}
            or type(shard["index"]) is not int
            or type(shard["count"]) is not int
        ):
            raise RegressionError("Case shard requires integer coordinates and its immutable source hash")
        expected_profile = profile_for_case_shard(expected_profile, f"{shard['index']}/{shard['count']}")
        expected_profile["case_shard"]["source_sha256"] = sha256(configured)
    if profile != expected_profile or contract["deploy_sha256"] != sha256(ROOT / profile["deploy_config"]):
        raise RegressionError("Reference workload or deployment differs from its frozen source profile")
    if profile.get("performance_regression_fraction") != 0.05:
        raise RegressionError("Candidate performance allowance must remain five percent")
    hardware = contract["hardware"]
    if (
        any(hardware.get(key) != profile[key] for key in ("platform", "cards", "dies"))
        or len(hardware.get("devices", [])) != profile["dies"]
    ):
        raise RegressionError("Hardware topology differs from the pinned profile")
    snapshot = profile["inference_runtime"].get("source_snapshot_sha256")
    if snapshot is not None:
        verify_source_attestations(hardware.get("source_attestations"), expected_sha256=snapshot)
        if contract_digest(hardware["source_attestations"]["packages"]) != profile["inference_runtime"].get(
            "source_packages_sha256"
        ):
            raise RegressionError("Runtime source packages differ from the pinned profile")
    for name, expected in profile["inference_runtime"]["packages"].items():
        actual = hardware["packages"].get(name)
        if not isinstance(actual, str):
            raise RegressionError(f"Missing inference package: {name}")
        parsed = Version(actual)
        matches = (
            actual == expected
            if profile["platform"] == "npu"
            else parsed.base_version == expected and not parsed.is_prerelease and not parsed.is_devrelease
        )
        if not matches:
            raise RegressionError(f"Calibration inference runtime differs from the profile: {name}")
    if manifest["reference_revision"] != profile["reference_revision"]:
        raise RegressionError("Reference source does not match the measured profile")
    cases = profile["cases"]
    if len(manifest["records"]["reference"]) != 3:
        raise RegressionError("Each cohort needs three independent starts")
    request_ids: set[str] = set()
    for repeat in range(3):
        if f"reference/{repeat}/precision.npz" not in manifest["artifacts"]:
            raise RegressionError("Reference precision artifact is unhashed")
        if set(manifest["records"]["reference"][repeat]) != {case["id"] for case in cases}:
            raise RegressionError("Incomplete reference case population")
        for case in cases:
            prefix = f"reference/{repeat}/{case['id']}"
            if any(f"{prefix}/{name}" not in manifest["artifacts"] for name in ("benchmark.json", "rollout.mp4")):
                raise RegressionError("Missing hashed reference benchmark or video")
            raw = read_json(generation / prefix / "benchmark.json")
            if (
                raw.get("git_revision") != manifest["harness_revision"]
                or raw.get("server_revision") != profile["reference_revision"]
                or _measured_metrics(raw, profile) != manifest["records"]["reference"][repeat][case["id"]]
            ):
                raise RegressionError("Raw reference metrics or source revision changed")
            ids = {session["request_id"] for session in raw["sessions"]}
            if ids & request_ids:
                raise RegressionError("Duplicated reference session across independent starts")
            request_ids.update(ids)
    producers = _measurement_producers(generation, manifest, report)
    for suite, tolerances in policy["candidate_tolerances"][profile["platform"]].items():
        relative = f"{suite}-raw.json"
        if relative not in report["artifacts"]:
            raise RegressionError("Native scores have no raw artifact hash")
        raw = read_json(quality / relative)
        labels = {f"case{index}_reference{repeat}" for index in range(len(cases)) for repeat in range(3)}
        if raw.get("status") != "passed" or set(raw.get("scores", {})) != labels:
            raise RegressionError("Native score population is incomplete or contains candidate input")
        if suite == "vbench":
            for label in labels:
                try:
                    verify_background_coverage(
                        raw.get("native_provenance", {}).get(label, {}).get("background_clip_inference", {}),
                        (profile["num_chunks"] * 3 - 1) * 4 + 1,
                    )
                except ValueError as error:
                    raise RegressionError(str(error)) from error
        if suite == "wbench" and raw.get("protocol", {}).get("dynamic_degree_backend") != "raft":
            raise RegressionError("WBench must use the frozen RAFT backend")
        if set(report["suites"][suite]) != {case["id"] for case in cases}:
            raise RegressionError("Quality report case coverage is incomplete")
        for index, case in enumerate(cases):
            rows = [raw["scores"][f"case{index}_reference{repeat}"] for repeat in range(3)]
            if report["suites"][suite][case["id"]]["reference_scores"] != rows:
                raise RegressionError("Native score summary differs from retained raw scores")
            if report["suites"][suite][case["id"]].get("checks"):
                raise RegressionError("Reference collection contains candidate quality checks")
            for row in rows:
                if set(row) != set(tolerances):
                    raise RegressionError("Native dimension population changed")
                for value in row.values():
                    if number(value, "native score") > 1:
                        raise RegressionError("Native score is outside [0, 1]")
    return {
        "generation": generation,
        "quality": quality,
        "manifest": manifest,
        "report": report,
        "contract": contract,
        "job": job,
        "requests": request_ids,
        "producers": producers,
    }


def _round_up(value: float, step: float) -> float:
    return math.ceil(value / step) * step


def _performance(cohorts: list[dict], contract: dict) -> dict:
    profile = contract["profile"]
    cases = [case["id"] for case in profile["cases"]]
    groups = {case: [[row[case] for row in c["manifest"]["records"]["reference"]] for c in cohorts] for case in cases}
    maximum_cv = max(
        statistics.pstdev([r["p95_ms"] for r in rows]) / statistics.mean(r["p95_ms"] for r in rows)
        for pair in groups.values()
        for rows in pair
    )
    p95_cv = max(profile["max_reference_cv"], _round_up(maximum_cv * 1.25, 0.005))
    floors = {"ttfc_ms": 0.0, "stall_ms": 0.0}
    deltas = []
    for case, pair in groups.items():
        for first, second in ((0, 1), (1, 0)):
            training, heldout = pair[first], pair[second]
            baseline = calibrate(
                training,
                contract=contract,
                reference_revision=profile["reference_revision"],
                max_cv=profile["max_reference_cv"],
                regression_fraction=profile["performance_regression_fraction"],
                transient_noise_sigma=profile["transient_noise_sigma"],
                p95_max_cv=p95_cv,
            )
            for metric in floors:
                worst = max(row[metric] for row in training)
                for row in heldout:
                    delta = max(0.0, row[metric] - worst) if row[metric] > baseline["limits"][metric]["value"] else 0.0
                    floors[metric] = max(floors[metric], delta)
                    deltas.append(
                        {
                            "case": case,
                            "training_cohort": first,
                            "heldout_cohort": second,
                            "metric": metric,
                            "required_floor_ms": delta,
                        }
                    )
    selected = {"ttfc_ms": _round_up(floors["ttfc_ms"] * 1.25, 5), "stall_ms": _round_up(floors["stall_ms"] * 1.25, 1)}
    checks: list[dict] = []
    for case, pair in groups.items():
        for first, second in ((0, 1), (1, 0)):
            baseline = calibrate(
                pair[first],
                contract=contract,
                reference_revision=profile["reference_revision"],
                max_cv=profile["max_reference_cv"],
                regression_fraction=profile["performance_regression_fraction"],
                transient_noise_sigma=profile["transient_noise_sigma"],
                startup_stall_jitter_floor_ms=selected["stall_ms"],
                ttfc_jitter_floor_ms=selected["ttfc_ms"],
                p95_max_cv=p95_cv,
            )
            checks.extend(
                {
                    "case": case,
                    "training_cohort": first,
                    "heldout_cohort": second,
                    "check": check_performance(row, baseline, contract=contract),
                }
                for row in pair[second]
            )
    if not all(row["check"]["passed"] for row in checks):
        raise RegressionError("Frozen reference performance null checks fail; investigate without candidate fitting")
    return {
        "schema_version": 1,
        "workload": profile,
        "qualified_profile_sha256": contract_digest(profile),
        "hardware": contract["hardware"],
        "deploy_sha256": contract["deploy_sha256"],
        "engine_environment": contract["engine_environment"],
        "reference_revision": profile["reference_revision"],
        "ttfc_calibration": {
            "selected_floor_ms": selected["ttfc_ms"],
            "maximum_required_floor_ms": floors["ttfc_ms"],
            "reserve_fraction": 0.25,
            "round_up_ms": 5,
        },
        "startup_stall_calibration": {
            "selected_floor_ms": selected["stall_ms"],
            "maximum_required_floor_ms": floors["stall_ms"],
            "reserve_fraction": 0.25,
            "round_up_ms": 1,
        },
        "p95_reference_cv_calibration": {"selected_ceiling": p95_cv, "maximum_training_p95_cv": maximum_cv},
        "performance_reference_null_checks": checks,
        "transient_reference_deltas": deltas,
    }


def build_calibration(
    generations: list[Path], qualities: list[Path], *, protocol: Path, python: Path, output: Path
) -> dict:
    if len(generations) != 2 or len(qualities) != 2:
        raise RegressionError("Freeze exactly two three-start reference cohorts")
    frozen = read_json(protocol)
    if frozen != read_json(PROTOCOL):
        raise RegressionError("Calibration algorithm requires the frozen v033 reference protocol")
    policy = read_json(POLICY)
    cohorts = [_load_cohort(g, q, sha256(protocol), policy) for g, q in zip(generations, qualities)]
    first, second = cohorts
    contract = first["contract"]
    profile = contract["profile"]
    platform = profile["platform"]
    reference_name = Path(profile["reference_calibration"]).name
    population_name = Path(profile["quality_reference_population"]).name
    if first["manifest"]["harness_revision"] != second["manifest"]["harness_revision"]:
        raise RegressionError("Reference training cohorts used different benchmark harness revisions")
    if (
        contract != second["contract"]
        or first["job"] == second["job"]
        or first["manifest"]["collection_id"] == second["manifest"]["collection_id"]
        or first["requests"] & second["requests"]
        or {row["collection_id"] for row in first["producers"].values()}
        & {row["collection_id"] for row in second["producers"].values()}
        or {_collection_identity(row, row) for row in first["producers"].values()}
        & {_collection_identity(row, row) for row in second["producers"].values()}
    ):
        raise RegressionError("Training requires distinct cohorts with equal model/runtime/workload contracts")
    if contract["profile"]["reference_revision"] != frozen["reference_revision"]:
        raise RegressionError("Reference revision differs from the frozen protocol")
    packages = first["report"]["evaluator_packages"]
    if verify_environment(python, ROOT / "requirements/lingbot-quality.txt") != packages:
        raise RegressionError("Pixel evaluator environment differs from the collected cohort")
    for key in ("evaluator_packages", "evaluator_opencv_provider", "evaluator_lock_sha256", "quality_policy_sha256"):
        if first["report"][key] != second["report"][key]:
            raise RegressionError(f"Reference evaluator contracts differ: {key}")
    if any((output / name).exists() for name in (reference_name, population_name, "calibration-build.json")):
        raise RegressionError("Frozen calibration output already exists; never overwrite training records")
    reference = _performance(cohorts, contract)
    common = {
        "candidate_inputs_used": False,
        "candidate_allowances_changed": False,
        "full_acceptance": False,
        "independent_validation_complete": False,
        "reference_only_protocol": frozen,
        "protocol_sha256": sha256(protocol),
        "quality_policy_sha256": sha256(POLICY),
        "evaluator_requirements_sha256": sha256(ROOT / "requirements/lingbot-quality.txt"),
        "inputs": [
            {
                "generation_job_id": c["job"],
                "collection_id": c["manifest"]["collection_id"],
                "collection_backend": c["manifest"]["collection_backend"],
                "buildkite_job_id": c["manifest"]["buildkite_job_id"],
                "measurement_jobs": c["manifest"].get("measurement_jobs"),
                "generation_manifest_sha256": sha256(c["generation"] / "generation.json"),
                "quality_manifest_sha256": sha256(c["quality"] / "quality.json"),
            }
            for c in cohorts
        ],
    }
    reference.update(common)
    pairs = [list(pair) for pair in combinations(range(6), 2)]
    population = {
        **common,
        "schema_version": 4,
        "contract": contract,
        "evaluator_packages_sha256": hashlib.sha256(
            json.dumps(packages, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "evaluator_lock_sha256": sha256(ROOT / "tests/dfx/lingbot_world/evaluators.lock.json"),
        "asset_lock_sha256": sha256(ROOT / "tests/dfx/lingbot_world/assets.lock.json"),
        "reference_pixel_calibration": {
            "protocol": "all_unordered_pairs_of_six_references",
            "protocol_sha256": sha256(protocol),
        },
        "max_reference_spreads": {},
        "cases": {},
    }
    for case in contract["profile"]["cases"]:
        name = case["id"]
        suites, spreads = {}, {}
        for suite, tolerances in policy["candidate_tolerances"][platform].items():
            rows = [row for c in cohorts for row in c["report"]["suites"][suite][name]["reference_scores"]]
            continuous = {
                key: val for key, val in tolerances.items() if key not in policy["binary_activity_dimensions"]
            }
            for dimension in policy["binary_activity_dimensions"]:
                check = check_binary_activity(
                    [r[dimension] for r in rows[:3]], [r[dimension] for r in rows[3:]], tolerance=tolerances[dimension]
                )
                if not check["passed"]:
                    raise RegressionError("Reference binary activity is not repeatable across cohorts")
            suites[suite] = [{key: row[key] for key in continuous} for row in rows]
            spreads[suite] = {
                key: max(
                    policy["max_reference_spreads"][platform][suite][key],
                    _round_up((max(row[key] for row in rows) - min(row[key] for row in rows)) * 1.25, 0.005),
                )
                for key in continuous
            }
            for row in suites[suite]:
                if not check_quality(suites[suite], row, tolerances=continuous, max_reference_spreads=spreads[suite])[
                    "passed"
                ]:
                    raise RegressionError("Reference native calibration does not pass its own null checks")
        videos = [(c, f"reference/{repeat}/{name}/rollout.mp4") for c in cohorts for repeat in range(3)]
        pixels = [
            compare_video_pair_in_environment(
                videos[a][0]["generation"] / videos[a][1],
                videos[b][0]["generation"] / videos[b][1],
                output / "pixels" / name,
                python=python,
                expected_packages=packages,
                window_seconds=policy["pixel_window_seconds"],
                ssim_stride=24,
                timeout_s=policy["timeout_s"],
            )
            for a, b in pairs
        ]
        for row in pixels:
            profile = contract["profile"]
            expected = (
                (profile["num_chunks"] * 3 - 1) * 4 + 1,
                profile["width"],
                profile["height"],
                profile["mux_fps"],
            )
            if (
                any(
                    tuple(row[role][key] for key in ("frame_count", "width", "height", "fps")) != expected
                    for role in ("reference", "candidate")
                )
                or row.get("decoded_frames") != expected[0]
            ):
                raise RegressionError("Reference pixel coverage differs from its measured workload")
            window_frames = round(profile["mux_fps"] * policy["pixel_window_seconds"])
            expected_windows = [
                (index, start, min(window_frames, expected[0] - start))
                for index, start in enumerate(range(0, expected[0], window_frames))
            ]
            if [
                (window["index"], window["start_frame"], window["frame_count"]) for window in row["windows"]
            ] != expected_windows:
                raise RegressionError("Reference pixel windows omit or change the pinned frame population")
            if (
                row.get("pixel_environment", {}).get("opencv_build_sha256")
                != first["report"]["evaluator_opencv_provider"]["opencv_build_sha256"]
            ):
                raise RegressionError("Pixel decoder build differs from the verified cohort provider")
            if not check_video_precision(
                pixels,
                row,
                **policy["pixel_budgets"][platform],
                min_psnr_db=policy["min_psnr_db"],
                reject_uniform_frames=policy["reject_uniform_frames"],
            )["passed"]:
                raise RegressionError("Reference pixel null checks fail")
        population["cases"][name] = {
            "input": case,
            "suites": suites,
            "max_reference_spreads": spreads,
            # Full per-frame raw data remains in hashed pixel worker artifacts.
            "pixels": [{key: value for key, value in row.items() if key != "frame_metrics"} for row in pixels],
            "pixel_pairs": pairs,
            "reference_videos": [
                {
                    "generation_job_id": c["job"],
                    "path": path,
                    "sha256": c["manifest"]["artifacts"][path],
                    "producer_job_id": c["producers"][int(Path(path).parts[1])]["buildkite_job_id"],
                    "producer_collection_id": c["producers"][int(Path(path).parts[1])]["collection_id"],
                    "producer_backend": c["producers"][int(Path(path).parts[1])]["collection_backend"],
                }
                for c, path in videos
            ],
        }
    population["max_reference_spreads"] = {
        suite: {
            dimension: max(case["max_reference_spreads"][suite][dimension] for case in population["cases"].values())
            for dimension in next(iter(population["cases"].values()))["max_reference_spreads"][suite]
        }
        for suite in ("vbench", "wbench")
    }
    verify_reference_calibration_contract(contract, reference)
    verify_quality_population_contract(
        population,
        contract,
        evaluator_packages=packages,
        evaluator_lock_sha256=population["evaluator_lock_sha256"],
        asset_lock_sha256=population["asset_lock_sha256"],
    )
    # Recheck immutable input files after workers have consumed the videos.
    for cohort in cohorts:
        _verified_artifacts(cohort["generation"], cohort["manifest"])
        _verified_artifacts(cohort["quality"], cohort["report"])
    _immutable_json(output / reference_name, reference)
    _immutable_json(output / population_name, population)
    result = {
        "complete": True,
        "full_acceptance": False,
        "independent_validation_complete": False,
        "artifacts": {name: sha256(output / name) for name in (reference_name, population_name)},
        "pixel_raw_artifacts": {
            str(path.relative_to(output)): sha256(path) for path in (output / "pixels").rglob("*") if path.is_file()
        },
    }
    _immutable_json(output / "calibration-build.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, action="append", required=True)
    parser.add_argument("--quality", type=Path, action="append", required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        build_calibration(args.generation, args.quality, protocol=args.protocol, python=args.python, output=args.output)
    except (RegressionError, OSError) as error:
        parser.exit(1, f"Calibration failed: {error}\n")


if __name__ == "__main__":
    main()
