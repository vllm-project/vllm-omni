# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reference-only NPU matrix jobs preserve raw producer evidence and fail closed."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from benchmarks.lingbot_world.regression import RegressionError, read_json, sha256, write_json
from tests.dfx.lingbot_world import runner

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
pytest_plugins = (
    "tests.benchmarks.test_lingbot_world_aggregation",
    "tests.benchmarks.test_lingbot_world_runtime",
)


@pytest.fixture
def reference_matrix(measurement_campaign, monkeypatch):
    settings = measurement_campaign
    for repeat in range(3):
        root = settings["generation"] / f"pair-{repeat}"
        shutil.rmtree(root / "candidate")
        path = root / "measurement.json"
        manifest = read_json(path)
        manifest.pop("candidate_revision")
        manifest.update(
            passed=False,
            reference_only=True,
            harness_revision=runner.git_revision(runner.ROOT),
            reference_protocol_sha256=sha256(
                runner.ROOT / "tests/dfx/lingbot_world/calibration/v033_reference_protocol.json"
            ),
            collection_id=str(uuid.uuid4()),
            collection_backend="buildkite",
            buildkite_job_id=f"reference-producer-{repeat}",
        )
        manifest["contract"]["hardware"]["packages"] = dict(settings["inference_runtime"]["packages"])
        manifest["records"] = {
            "reference": [
                {
                    case["id"]: runner._measured_metrics(
                        read_json(root / f"reference/{repeat}/{case['id']}/benchmark.json"), settings
                    )
                    for case in settings["cases"]
                }
            ]
        }
        manifest["artifacts"] = {
            relative: digest for relative, digest in manifest["artifacts"].items() if relative.startswith("reference/")
        }
        write_json(path, manifest)

    def forbidden(*args, **kwargs):
        pytest.fail("Reference aggregation consulted calibration or candidate acceptance")

    original_read = runner.read_json

    def read(path):
        if Path(path) == Path(settings["reference_calibration"]):
            forbidden()
        return original_read(path)

    monkeypatch.setattr(runner, "read_json", read)
    monkeypatch.setattr(runner, "_finalize_measurements", forbidden)
    return settings


@pytest.mark.parametrize("backend", ["buildkite", "local"])
def test_reference_aggregate_retains_three_independent_producers_without_acceptance(
    reference_matrix, monkeypatch, backend
):
    settings = reference_matrix
    monkeypatch.setenv("BUILDKITE_JOB_ID", "actual-cpu-aggregate-job")
    producers = []
    for repeat in range(3):
        path = settings["generation"] / f"pair-{repeat}/measurement.json"
        manifest = read_json(path)
        if backend == "local":
            manifest.update(collection_backend="local", buildkite_job_id=None)
            write_json(path, manifest)
        producers.append(manifest)
    report = runner.aggregate_reference_measurements(settings)
    assert report["complete"] and report["measurement_only"] and report["reference_only"]
    assert report["passed"] is False and set(report["records"]) == {"reference"}
    assert len(report["records"]["reference"]) == 3
    assert not (settings["output"] / "candidate").exists()
    assert report["buildkite_job_id"] == "actual-cpu-aggregate-job"
    assert report["collection_id"] not in {row["collection_id"] for row in producers}
    assert len(report["measurement_jobs"]) == 3
    for repeat, (job, original) in enumerate(zip(report["measurement_jobs"], producers)):
        assert job["repeat"] == repeat
        assert {key: job[key] for key in ("collection_id", "collection_backend", "buildkite_job_id")} == {
            key: original[key] for key in ("collection_id", "collection_backend", "buildkite_job_id")
        }
        assert read_json(settings["output"] / job["manifest_path"]) == original
        assert sha256(settings["output"] / job["manifest_path"]) == job["manifest_sha256"]
        assert report["artifacts"][job["manifest_path"]] == job["manifest_sha256"]
    runner.validate_reference_collection(settings, settings["output"], report)


@pytest.mark.parametrize(
    "measurement_campaign", ["npu_nightly", *(("npu_weekly", shard) for shard in range(6))], indirect=True
)
def test_reference_aggregate_preserves_nightly_sessions_and_every_weekly_shard(reference_matrix):
    settings = reference_matrix
    report = runner.aggregate_reference_measurements(settings)
    assert set(report["records"]["reference"][0]) == {case["id"] for case in settings["cases"]}
    expected_samples = (settings["num_chunks"] - settings["warmup_chunks"] - 1) * settings["measured_sessions"]
    for row in report["records"]["reference"]:
        assert all(metrics["steady_samples"] == expected_samples for metrics in row.values())
    assert not report["passed"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_producer",
        "missing_start",
        "collection_reuse",
        "job_reuse",
        "repeat",
        "candidate",
        "candidate_revision",
        "unlisted",
        "corrupt",
        "forged_metrics",
        "request_reuse",
        "missing_session",
        "hardware",
        "profile",
        "engine",
        "source",
        "protocol",
        "unsafe",
        "wrong_repeat",
        "accepted",
    ],
)
def test_reference_aggregate_rejects_incomplete_mixed_or_reused_inputs(reference_matrix, mutation):
    settings = reference_matrix
    root = settings["generation"] / "pair-2"
    path = root / "measurement.json"
    manifest = read_json(path)
    first_path = settings["generation"] / "pair-0/measurement.json"
    first = read_json(first_path)
    case = settings["cases"][0]["id"]
    raw_path = root / f"reference/2/{case}/benchmark.json"
    if mutation == "missing_producer":
        path.unlink()
    elif mutation == "missing_start":
        manifest["records"]["reference"] = []
    elif mutation == "collection_reuse":
        manifest["collection_id"] = first["collection_id"]
    elif mutation == "job_reuse":
        manifest["buildkite_job_id"] = first["buildkite_job_id"]
    elif mutation == "repeat":
        manifest["repeat"] = 0
    elif mutation == "candidate":
        (root / "candidate").mkdir()
    elif mutation == "candidate_revision":
        manifest["candidate_revision"] = "c" * 40
    elif mutation == "unlisted":
        del manifest["artifacts"][f"reference/2/{case}/rollout.mp4"]
    elif mutation == "corrupt":
        (root / f"reference/2/{case}/rollout.mp4").write_bytes(b"damaged")
    elif mutation == "forged_metrics":
        manifest["records"]["reference"][0][case]["mean_ms"] = 0
    elif mutation in {"request_reuse", "missing_session", "source"}:
        raw = read_json(raw_path)
        if mutation == "request_reuse":
            other = read_json(settings["generation"] / f"pair-0/reference/0/{case}/benchmark.json")
            raw["sessions"][0]["request_id"] = other["sessions"][0]["request_id"]
        elif mutation == "missing_session":
            raw["sessions"].pop()
        else:
            raw["server_revision"] = "c" * 40
        write_json(raw_path, raw)
        manifest["artifacts"][str(raw_path.relative_to(root))] = sha256(raw_path)
    elif mutation == "hardware":
        manifest["contract"]["hardware"]["devices"][0]["total_bytes"] += 1
    elif mutation == "profile":
        manifest["contract"]["profile"]["width"] += 1
    elif mutation == "engine":
        manifest["contract"]["engine_environment"] = {}
    elif mutation == "protocol":
        manifest["reference_protocol_sha256"] = "0" * 64
    elif mutation == "unsafe":
        manifest["artifacts"]["reference/../../escaped"] = "0" * 64
    elif mutation == "wrong_repeat":
        manifest["artifacts"]["reference/0/precision.npz"] = "0" * 64
    elif mutation == "accepted":
        manifest["passed"] = True
    if mutation != "missing_producer":
        write_json(path, manifest)
    with pytest.raises(RegressionError):
        runner.aggregate_reference_measurements(settings)
    assert read_json(settings["output"] / "generation.json")["passed"] is False


def test_paired_acceptance_rejects_reference_matrix_inputs(reference_matrix):
    with pytest.raises(RegressionError, match="Incomplete, duplicate or wrong-revision"):
        runner.aggregate_measurements(reference_matrix)


@pytest.mark.parametrize("repeat", range(3))
def test_reference_matrix_collection_runs_exactly_one_reference_start(reference_collection, monkeypatch, repeat):
    settings, _, reference, captures, starts = reference_collection
    settings = {**settings, "platform": "npu", "output": settings["output"].parent / f"pair-{repeat}"}
    profile = read_json(settings["profile_path"])
    profile["platform"] = "npu"
    write_json(settings["profile_path"], profile)
    before = len(starts)
    report = runner.generate(settings, measurement_repeat=repeat, reference_only=True)
    assert starts[before:] == captures[before:] == [reference]
    assert len(report["records"]["reference"]) == 1 and set(report["records"]) == {"reference"}
    assert report["repeat"] == repeat and not report["passed"]
    assert read_json(settings["output"] / "measurement.json") == report
    assert read_json(settings["output"] / "generation.json")["status"] == "awaiting_reference_aggregate"
    runner.validate_reference_collection(settings, settings["output"], report, measurement_repeat=repeat)
    with pytest.raises(RegressionError, match="independent-start"):
        runner.validate_reference_collection(settings, settings["output"], report)
