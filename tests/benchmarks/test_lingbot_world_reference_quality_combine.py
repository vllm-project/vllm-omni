# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Join real per-case schemas while retaining their immutable evidence."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from benchmarks.lingbot_world.regression import RegressionError, read_json, sha256, write_json
from tests.benchmarks import test_lingbot_world_calibration as calibration_tests
from tests.dfx.lingbot_world import calibrate as builder
from tests.dfx.lingbot_world.combine_reference_quality import combine_reference_quality

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
campaign = calibration_tests.campaign


def _split(quality: Path) -> list[Path]:
    complete = read_json(quality / "quality.json")
    lock = read_json(builder.ROOT / "tests/dfx/lingbot_world/evaluators.lock.json")
    sources = []
    for index, case in enumerate(complete["contract"]["profile"]["cases"]):
        source = quality.parent / f"{quality.name}-case-{index}"
        source.mkdir()
        report = copy.deepcopy(complete)
        report.update(
            generation_passed=False,
            quality_case_index=index,
            evaluated_cases=[case["id"]],
            quality_collection_id=str(uuid4()),
            quality_collection_backend="buildkite",
            quality_buildkite_job_id=f"{quality.name}-evaluator-{index}",
            evaluator_hardware={"platform": "cuda", "devices": [{"name": "H100"}], "allocation": {"id": index}},
            pixels={case["id"]: {"reference_repeat": [{"ssim": 1}], "candidate": [], "checks": []}},
        )
        report["suites"] = {suite: {case["id"]: summary[case["id"]]} for suite, summary in complete["suites"].items()}
        for suite in report["suites"]:
            raw = read_json(quality / f"{suite}-raw.json")
            raw.update(
                suite=suite,
                dimensions=list(next(iter(raw["scores"].values()))),
                repo_revision=lock["repositories"][suite]["revision"],
            )
            for field in ("scores", "native_provenance"):
                raw[field] = {
                    f"case0_reference{repeat}": raw[field][f"case{index}_reference{repeat}"] for repeat in range(3)
                }
            write_json(source / f"{suite}-raw.json", raw)
        (source / "worker.log").write_text(f"native scoring case {index}\n")
        report["artifacts"] = {str(path.relative_to(source)): sha256(path) for path in source.iterdir()}
        write_json(source / "quality.json", report)
        (source / "pytest.xml").write_text('<testsuite tests="1"><testcase name="reference_quality"/></testsuite>')
        sources.append(source)
    return sources


@pytest.mark.parametrize(
    "campaign", [(platform + "_weekly", shard, 6) for platform in ("cuda", "npu") for shard in range(6)], indirect=True
)
def test_combined_shard_preserves_sources_and_is_consumed_by_calibration(campaign, monkeypatch):
    # CI wrappers can forward an empty variable into a local collection.
    monkeypatch.setenv("BUILDKITE_JOB_ID", "")
    outputs = []
    for generation, quality in zip(campaign["generations"], campaign["qualities"]):
        sources = _split(quality)
        before = {path: sha256(path) for source in sources for path in source.rglob("*") if path.is_file()}
        output = quality.parent / f"{quality.name}-combined"
        report = combine_reference_quality(generation, sources[::-1], output=output)
        assert report["complete"] and report["reference_only"] and report["measurement_only"]
        assert not report["passed"] and not report["quality_passed"]
        assert report["quality_collection_backend"] == "local" and report["quality_buildkite_job_id"] is None
        assert report["contract"] == read_json(generation / "generation.json")["contract"]
        assert report["evaluated_cases"] == [case["id"] for case in campaign["profile"]["cases"]]
        assert len(report["quality_jobs"]) == 3
        for index, source in enumerate(sources):
            retained = output / "sources" / f"case-{index}"
            for path in source.rglob("*"):
                if path.is_file():
                    assert sha256(retained / path.relative_to(source)) == before[path] == sha256(path)
        for suite in report["suites"]:
            raw = read_json(output / f"{suite}-raw.json")
            assert set(raw["scores"]) == {f"case{index}_reference{repeat}" for index in range(3) for repeat in range(3)}
        with pytest.raises(RegressionError, match="never overwrite"):
            combine_reference_quality(generation, sources, output=output)
        outputs.append(output)
    result = builder.build_calibration(
        campaign["generations"],
        outputs,
        protocol=builder.PROTOCOL,
        python=Path(sys.executable),
        output=campaign["output"],
    )
    assert result["complete"] and not result["full_acceptance"]
    assert len(campaign["calls"]) == 45


@pytest.mark.parametrize("campaign", [("cuda_weekly", 0, 6)], indirect=True)
@pytest.mark.parametrize(
    "mutation",
    [
        "missing_case",
        "duplicate_case",
        "duplicate_job",
        "missing_job",
        "generation",
        "protocol",
        "candidate",
        "acceptance",
        "calibration",
        "policy",
        "evaluator",
        "hardware",
        "raw_hash",
        "unhashed",
        "symlink",
        "native_candidate",
        "native_summary",
        "native_provenance",
        "raw_protocol",
        "candidate_pixels",
        "extra_case",
        "native_revision",
        "native_dimensions",
        "failed_junit",
        "skipped_junit",
        "missing_junit_case",
        "invalid_junit",
        "setup_failure",
    ],
)
def test_rejects_mixed_incomplete_or_unverifiable_quality_sources(campaign, mutation):
    generation, quality = campaign["generations"][0], campaign["qualities"][0]
    sources = _split(quality)
    path = sources[1] / "quality.json"
    report = read_json(path)
    raw_path = sources[1] / "vbench-raw.json"
    raw = read_json(raw_path)
    if mutation == "missing_case":
        sources.pop()
    elif mutation == "duplicate_case":
        sources[1] = sources[0]
    elif mutation == "duplicate_job":
        report["quality_buildkite_job_id"] = read_json(sources[0] / "quality.json")["quality_buildkite_job_id"]
    elif mutation == "missing_job":
        report.pop("quality_collection_id")
    elif mutation == "generation":
        report["generation_manifest_sha256"] = "a" * 64
    elif mutation == "protocol":
        report["reference_protocol_sha256"] = "a" * 64
    elif mutation == "candidate":
        report["candidate_revision"] = "a" * 40
    elif mutation == "acceptance":
        report["passed"] = True
    elif mutation == "calibration":
        report["reference_population_sha256"] = "a" * 64
    elif mutation == "policy":
        report["quality_policy_sha256"] = "a" * 64
    elif mutation == "evaluator":
        report["evaluator_packages"]["numpy"] = "2.0"
    elif mutation == "hardware":
        report["evaluator_hardware"]["devices"][0]["name"] = "A100"
    elif mutation == "unhashed":
        (sources[1] / "unhashed.log").write_text("untracked")
    elif mutation == "symlink":
        target = sources[1] / "worker.log"
        (sources[1] / "alias.log").symlink_to(target)
        report["artifacts"]["alias.log"] = sha256(target)
    elif mutation == "failed_junit":
        (sources[1] / "pytest.xml").write_text("<testsuite><testcase><failure/></testcase></testsuite>")
    elif mutation == "skipped_junit":
        (sources[1] / "pytest.xml").write_text("<testsuite><testcase><skipped/></testcase></testsuite>")
    elif mutation == "missing_junit_case":
        (sources[1] / "pytest.xml").write_text('<testsuite tests="0"/>')
    elif mutation == "invalid_junit":
        (sources[1] / "pytest.xml").write_text("unfinished")
    elif mutation == "setup_failure":
        write_json(sources[1] / "setup-failure.json", {"passed": False})
    elif mutation == "native_candidate":
        raw["scores"]["case0_candidate0"] = raw["scores"]["case0_reference0"]
    elif mutation == "native_summary":
        case = report["evaluated_cases"][0]
        report["suites"]["vbench"][case]["reference_scores"][0]["subject_consistency"] = 0.1
    elif mutation == "native_provenance":
        raw["native_provenance"].pop("case0_reference0")
    elif mutation == "native_revision":
        raw["repo_revision"] = "a" * 40
    elif mutation == "native_dimensions":
        raw["dimensions"].append("unlocked")
    elif mutation == "raw_protocol":
        raw["protocol"]["dynamic_degree_backend"] = "unknown"
    elif mutation == "candidate_pixels":
        report["pixels"][report["evaluated_cases"][0]]["candidate"] = [{"ssim": 1}]
    elif mutation == "extra_case":
        report["evaluated_cases"].append("extra")
    write_json(raw_path, raw if mutation != "raw_hash" else {})
    if mutation != "raw_hash":
        report["artifacts"]["vbench-raw.json"] = sha256(raw_path)
    write_json(path, report)
    output = quality.parent / "combined"
    with pytest.raises(RegressionError):
        combine_reference_quality(generation, sources, output=output)
    assert not output.exists()
