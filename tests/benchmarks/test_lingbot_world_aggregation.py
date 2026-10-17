# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Synthetic artifact tests verify aggregation gates, not model inference."""

import copy
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from benchmarks.lingbot_world.precision import save_tensor_sample
from benchmarks.lingbot_world.regression import RegressionError, sha256, write_json
from tests.benchmarks.test_lingbot_world_regression import document
from tests.dfx.lingbot_world import runner
from tests.dfx.lingbot_world.runner import git_revision as source_revision

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def measurement_campaign(tmp_path, monkeypatch, request):
    revision = "b" * 40
    monkeypatch.setattr(runner, "git_revision", lambda root: revision)
    name = getattr(request, "param", "npu_long")
    shard = None
    if isinstance(name, tuple):
        name, shard = name
    profile = json.loads((runner.ROOT / f"tests/dfx/lingbot_world/configs/{name}.json").read_text())
    if shard is not None:
        from tests.dfx.lingbot_world.prepare import profile_for_case_shard

        profile = profile_for_case_shard(profile, f"{shard}/6")
    profile_path = tmp_path / "profile.json"
    write_json(profile_path, profile)
    source = tmp_path / "measurements"
    contract = {
        "profile": profile,
        "engine_environment": {
            "VLLM_OMNI_AR_DIFFUSION_KV_GATHER": "0",
            "AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH": "0",
        },
        "hardware": {
            "platform": "npu",
            "cards": 4,
            "dies": 4,
            "devices": [{"name": "Ascend910B3", "total_bytes": 64 * 1024**3}] * 4,
            "packages": {"torch": "2.10.0", "vllm": "0.31.0"},
        },
        "deploy_sha256": sha256(runner.ROOT / profile["deploy_config"]),
    }
    cases = [case["id"] for case in profile["cases"]]
    for repeat in range(3):
        root = source / f"pair-{repeat}"
        for role in ("reference", "candidate"):
            arm = root / role / str(repeat)
            for case in cases:
                destination = arm / case
                destination.mkdir(parents=True)
                raw = document(num_chunks=profile["num_chunks"])
                raw["sessions"] = [
                    {
                        **copy.deepcopy(raw["sessions"][0]),
                        "session_index": index,
                        "request_id": f"{role}-{repeat}-{case}-{index}",
                    }
                    for index in range(profile["measured_sessions"])
                ]
                raw["git_revision"] = revision
                raw["server_revision"] = profile["reference_revision"] if role == "reference" else revision
                write_json(destination / "benchmark.json", raw)
                (destination / "rollout.mp4").write_bytes(b"synthetic media; decoder is checked by the quality job")
            save_tensor_sample(arm / "precision.npz", inputs={"latent": np.ones(2)}, outputs={"decoded": np.ones(2)})
            if repeat == 0:
                write_json(arm / "diagnostic.json", {"synthetic": True})
        manifest = {
            "measurement_only": True,
            "complete": True,
            "repeat": repeat,
            "contract": contract,
            "candidate_revision": revision,
            "reference_revision": profile["reference_revision"],
            "buildkite_job_id": f"pair-{repeat}",
            "records": {role: [{case: {"mean_ms": 0} for case in cases}] for role in ("reference", "candidate")},
            "artifacts": {
                str(path.relative_to(root)): sha256(path)
                for role in ("reference", "candidate")
                for path in (root / role).rglob("*")
                if path.is_file()
            },
        }
        write_json(root / "measurement.json", manifest)
    return {**profile, "profile_path": profile_path, "generation": source, "output": tmp_path / "aggregate"}


def test_aggregate_recomputes_all_six_populations_and_gates_p99(measurement_campaign):
    report = runner.aggregate_measurements(measurement_campaign)
    assert report["passed"]
    case = report["cases"]["boardwalk-42"]
    assert case["baseline"]["independent_starts"] == 3
    assert "p99_ms" in case["baseline"]["limits"]
    assert len(case["checks"]) == 3
    assert all(check["measured"]["steady_samples"] == 1186 for check in case["checks"])
    assert all(check["measured"]["mean_ms"] == 500 for check in case["checks"])
    assert len(report["precision"]) == 7 and all(p["status"] == "passed" for p in report["precision"])


@pytest.mark.parametrize("measurement_campaign", [("npu_weekly", shard) for shard in range(6)], indirect=True)
@pytest.mark.parametrize("change", [None, "missing_pair", "other_shard"])
def test_weekly_shard_aggregate_requires_three_matching_pairs(measurement_campaign, change):
    settings = measurement_campaign
    assert len(settings["cases"]) == 3 and settings["measured_sessions"] == 2
    manifest = settings["generation"] / "pair-2/measurement.json"
    if change == "missing_pair":
        manifest.unlink()
    elif change == "other_shard":
        value = json.loads(manifest.read_text())
        value["contract"]["profile"]["case_shard"]["index"] = (settings["case_shard"]["index"] + 1) % 6
        write_json(manifest, value)
    if change is not None:
        with pytest.raises(RegressionError, match="Missing measurement pair|profile or hardware mismatch"):
            runner.aggregate_measurements(settings)
        assert not json.loads((settings["output"] / "generation.json").read_text())["passed"]
        return
    report = runner.aggregate_measurements(settings)
    assert report["passed"] and set(report["cases"]) == {case["id"] for case in settings["cases"]}
    assert len(report["measurement_jobs"]) == 3
    for case in report["cases"].values():
        assert case["baseline"]["independent_starts"] == 3 and len(case["checks"]) == 3
        assert all(check["passed"] and check["measured"]["steady_samples"] == 306 for check in case["checks"])
    assert len(report["precision"]) == 7 and all(row["status"] == "passed" for row in report["precision"])


@pytest.mark.parametrize("measurement_campaign", ["npu_nightly"], indirect=True)
@pytest.mark.parametrize("change", [None, "missing_session", "p95_regression"])
def test_nightly_aggregate_pools_complete_sessions_and_preserves_p95_gate(measurement_campaign, change):
    settings = measurement_campaign
    assert settings["measured_sessions"] == 7 and settings["min_steady_samples"] == 231
    if change is not None:
        root = settings["generation"] / "pair-2"
        path = root / "candidate/2/boardwalk-42/benchmark.json"
        raw = json.loads(path.read_text())
        if change == "missing_session":
            raw["sessions"].pop()
        else:
            for session in raw["sessions"]:
                arrival = 0.0
                for chunk in session["chunks"]:
                    if chunk["index"] in (30, 31):
                        chunk["inter_arrival_ms"] = 900
                    arrival += chunk["inter_arrival_ms"] / 1000
                    chunk["arrival_s"] = arrival
                session["metrics"]["wall_seconds"] = arrival
        write_json(path, raw)
        manifest_path = root / "measurement.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"][str(path.relative_to(root))] = sha256(path)
        write_json(manifest_path, manifest)
    if change == "missing_session":
        with pytest.raises(RegressionError, match="measured sessions"):
            runner.aggregate_measurements(settings)
        return
    report = runner.aggregate_measurements(settings)
    assert report["passed"] is (change is None)
    assert set(report["cases"]) == {case["id"] for case in settings["cases"]}
    for case in report["cases"].values():
        assert case["baseline"]["independent_starts"] == 3
        assert "p95_ms" in case["baseline"]["limits"] and "p99_ms" not in case["baseline"]["limits"]
        assert all(check["measured"]["steady_samples"] == 231 for check in case["checks"])
    if change == "p95_regression":
        check = report["cases"]["boardwalk-42"]["checks"][2]
        assert check["measured"]["mean_ms"] < report["cases"]["boardwalk-42"]["baseline"]["limits"]["mean_ms"]["value"]
        assert any("p95_ms" in violation for violation in check["violations"])
    assert len(report["precision"]) == 7 and all(p["status"] == "passed" for p in report["precision"])


@pytest.mark.parametrize("change", ["missing", "enabled"])
def test_aggregate_rejects_unpinned_engine_switches_even_when_all_pairs_agree(measurement_campaign, change):
    settings = measurement_campaign
    for repeat in range(3):
        path = settings["generation"] / f"pair-{repeat}/measurement.json"
        manifest = json.loads(path.read_text())
        if change == "missing":
            manifest["contract"].pop("engine_environment")
        else:
            manifest["contract"]["engine_environment"]["VLLM_OMNI_AR_DIFFUSION_KV_GATHER"] = "1"
        write_json(path, manifest)
    with pytest.raises(RegressionError, match="engine environment"):
        runner.aggregate_measurements(settings)


@pytest.mark.parametrize(
    "change",
    [
        "missing_pair",
        "incomplete",
        "repeat",
        "revision",
        "server",
        "profile",
        "hardware",
        "dependencies",
        "artifact",
        "hash",
        "path",
        "cases",
        "short",
        "duplicate_sessions",
        "duplicate_start",
        "policy",
    ],
)
def test_incomplete_or_incompatible_pairs_cannot_pass(measurement_campaign, change):
    settings = measurement_campaign
    root = settings["generation"] / "pair-2"
    path = root / "measurement.json"
    manifest = json.loads(path.read_text())
    raw_path = root / "candidate/2/boardwalk-42/benchmark.json"
    if change == "missing_pair":
        path.unlink()
    elif change == "incomplete":
        manifest["complete"] = False
    elif change == "repeat":
        manifest["repeat"] = 1
    elif change == "revision":
        manifest["candidate_revision"] = "c" * 40
    elif change == "profile":
        manifest["contract"]["profile"]["num_chunks"] = 100
    elif change == "hardware":
        manifest["contract"]["hardware"]["devices"][0]["name"] = "different accelerator"
    elif change == "dependencies":
        manifest["contract"]["hardware"]["packages"]["torch"] = "other-version"
    elif change == "artifact":
        manifest["artifacts"].pop("candidate/2/precision.npz")
    elif change == "hash":
        raw_path.write_bytes(b"corrupt")
    elif change == "path":
        manifest["artifacts"]["../escape"] = "0" * 64
    elif change == "cases":
        manifest["records"]["candidate"] = [{}]
    else:
        raw = json.loads(raw_path.read_text())
        if change == "server":
            raw["server_revision"] = "c" * 40
        elif change == "duplicate_sessions":
            raw["sessions"][1] = copy.deepcopy(raw["sessions"][0])
        elif change == "duplicate_start":
            raw = json.loads((root.parent / "pair-1/candidate/1/boardwalk-42/benchmark.json").read_text())
        elif change == "policy":
            raw["config"]["warmup_chunks"] = 0
        else:
            raw["sessions"].pop()
        write_json(raw_path, raw)
        manifest["artifacts"][str(raw_path.relative_to(root))] = sha256(raw_path)
    if change != "missing_pair":
        write_json(path, manifest)
    with pytest.raises(RegressionError):
        runner.aggregate_measurements(settings)
    assert not json.loads((settings["output"] / "generation.json").read_text())["passed"]


@pytest.mark.parametrize("regression", ["performance", "precision"])
def test_complete_but_regressed_population_fails_aggregate_gate(measurement_campaign, regression):
    settings = measurement_campaign
    root = settings["generation"] / "pair-2"
    manifest_path = root / "measurement.json"
    manifest = json.loads(manifest_path.read_text())
    if regression == "performance":
        path = root / "candidate/2/boardwalk-42/benchmark.json"
        raw = document(interval=0.8, num_chunks=600)
        raw["sessions"] = [
            {**copy.deepcopy(raw["sessions"][0]), "session_index": index, "request_id": f"slower-{index}"}
            for index in range(2)
        ]
        raw["git_revision"] = raw["server_revision"] = manifest["candidate_revision"]
        write_json(path, raw)
    else:
        path = root / "candidate/2/precision.npz"
        save_tensor_sample(path, inputs={"latent": np.ones(2)}, outputs={"decoded": np.full(2, 2)})
    manifest["artifacts"][str(path.relative_to(root))] = sha256(path)
    write_json(manifest_path, manifest)
    report = runner.aggregate_measurements(settings)
    assert not report["passed"]
    if regression == "performance":
        assert not report["cases"]["boardwalk-42"]["checks"][2]["passed"]
    else:
        assert any(p["status"] != "passed" for p in report["precision"])


@pytest.mark.parametrize("precision_failure", [False, True])
def test_unstable_reference_retains_complete_failed_manifest_and_precision(measurement_campaign, precision_failure):
    settings = measurement_campaign
    root = settings["generation"] / "pair-2"
    path = root / "reference/2/boardwalk-42/benchmark.json"
    raw = json.loads(path.read_text())
    for session in raw["sessions"]:
        session["metrics"]["wall_seconds"] *= 2
        for chunk in session["chunks"]:
            chunk["arrival_s"] *= 2
            chunk["inter_arrival_ms"] *= 2
    write_json(path, raw)
    manifest_path = root / "measurement.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][str(path.relative_to(root))] = sha256(path)
    if precision_failure:
        sample = root / "candidate/2/precision.npz"
        save_tensor_sample(sample, inputs={"latent": np.ones(2)}, outputs={"decoded": np.full(2, 2)})
        manifest["artifacts"][str(sample.relative_to(root))] = sha256(sample)
    write_json(manifest_path, manifest)

    report = runner.aggregate_measurements(settings)
    assert report["passed"] is False
    assert "Unstable reference" in report["cases"]["boardwalk-42"]["calibration_error"]
    assert report["cases"]["boardwalk-42"]["baseline"] is None
    assert len(report["precision"]) == 7
    assert any(row["status"] != "passed" for row in report["precision"]) is precision_failure
    for role in ("reference", "candidate"):
        for repeat in range(3):
            relative = f"{role}/{repeat}/boardwalk-42/rollout.mp4"
            assert report["artifacts"][relative] == sha256(settings["output"] / relative)
    assert json.loads((settings["output"] / "generation.json").read_text()) == report


def test_final_gate_rejects_missing_candidate_start(measurement_campaign):
    settings = measurement_campaign
    records: dict[str, list[dict[str, dict[str, float]]]] = {
        role: [{"boardwalk-42": {}}] * 3 for role in ("reference", "candidate")
    }
    records["candidate"].pop()
    with pytest.raises(RegressionError, match="three independent"):
        runner._finalize_measurements(settings, {}, copy.deepcopy(records))


def test_aggregate_launcher_executes_pytest_and_returns_nonzero_for_regression(measurement_campaign):
    settings = measurement_campaign
    # Source-only CI images intentionally omit .git; use the same checked
    # Buildkite SHA fallback as the real artifact consumer.
    revision = source_revision(runner.ROOT)
    for repeat in range(3):
        root = settings["generation"] / f"pair-{repeat}"
        manifest_path = root / "measurement.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["candidate_revision"] = revision
        for role in ("reference", "candidate"):
            path = root / role / str(repeat) / "boardwalk-42/benchmark.json"
            raw = json.loads(path.read_text())
            raw["git_revision"] = revision
            if role == "candidate":
                raw["server_revision"] = revision
            write_json(path, raw)
            manifest["artifacts"][str(path.relative_to(root))] = sha256(path)
        write_json(manifest_path, manifest)
    environment = {**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]}
    environment.pop("BUILDKITE", None)
    environment.pop("LINGBOT_CASE_SHARD", None)
    for regressed in (False, True):
        if regressed:
            root = settings["generation"] / "pair-2"
            path = root / "candidate/2/boardwalk-42/benchmark.json"
            raw = document(interval=0.8, num_chunks=600)
            raw["sessions"] = [
                {**copy.deepcopy(raw["sessions"][0]), "session_index": i, "request_id": f"slower-{i}"} for i in range(2)
            ]
            raw["git_revision"] = raw["server_revision"] = revision
            write_json(path, raw)
            manifest_path = root / "measurement.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"][str(path.relative_to(root))] = sha256(path)
            write_json(manifest_path, manifest)
        output = settings["output"].with_name("regressed" if regressed else "passed")
        result = subprocess.run(
            [
                "bash",
                str(runner.ROOT / ".buildkite/common/scripts/run-lingbot-ci.sh"),
                "aggregate",
                str(settings["profile_path"]),
                str(output),
                str(settings["generation"]),
            ],
            cwd=runner.ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == (1 if regressed else 0), result.stdout + result.stderr
        suites = list(ET.parse(output / "pytest.xml").getroot().iter("testsuite"))
        assert len(suites) == 1 and suites[0].get("tests") == "1" and suites[0].get("skipped") == "0"
        assert suites[0].get("failures") == ("1" if regressed else "0")


@pytest.mark.parametrize(("selector", "expected_exit"), [("core_model and cpu", 5), ("full_model and cpu", 0)])
def test_full_campaign_gate_is_selected_only_by_explicit_full_model_launcher(selector, expected_exit, tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "tests/dfx/lingbot_world/test_aggregate.py",
            "-m",
            selector,
            "--lingbot-output",
            str(tmp_path / "output"),
        ],
        cwd=runner.ROOT,
        env={**os.environ, "BUILDKITE": "true"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == expected_exit, result.stdout + result.stderr
    nodeid = "test_aggregate.py::test_all_npu_measurement_pairs_pass_regression"
    assert (nodeid in result.stdout) == (expected_exit == 0)
