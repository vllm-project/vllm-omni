# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Join independently scored reference cases without establishing acceptance."""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from uuid import uuid4

from benchmarks.lingbot_world.regression import RegressionError, read_json, sha256, write_json
from tests.dfx.lingbot_world.calibrate import (
    POLICY,
    PROTOCOL,
    ROOT,
    _collection_identity,
    _load_cohort,
    _semantic_contract,
    _verified_artifacts,
)


def _quality_identity(report: dict) -> str:
    return _collection_identity(
        {name: report.get(f"quality_{name}") for name in ("collection_id", "collection_backend", "buildkite_job_id")},
        {name: report.get(f"quality_{name}") for name in ("collection_id", "collection_backend", "buildkite_job_id")},
    )


def _source_report(directory: Path, manifest: dict, generation_sha: str, policy: dict) -> tuple[dict, dict]:
    report = read_json(directory / "quality.json")
    _verified_artifacts(directory, report)
    files = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()}
    if (
        files - {"pytest.xml"} != {*report["artifacts"], "quality.json"} - {"pytest.xml"}
        or "setup-failure.json" in files
        or any(path.is_symlink() for path in directory.rglob("*"))
    ):
        raise RegressionError("Quality source must retain exactly its hashed artifacts without symlinks")
    # pytest finalizes this report after scoring returns, so scoring cannot hash
    # it. Require a successful selection and hash its final bytes in the join.
    if "pytest.xml" in files:
        try:
            junit = ET.parse(directory / "pytest.xml")
        except ET.ParseError as error:
            raise RegressionError("Invalid final quality JUnit report") from error
        if not list(junit.iter("testcase")) or any(list(junit.iter(tag)) for tag in ("failure", "error", "skipped")):
            raise RegressionError("Incomplete or failed final quality JUnit selection")
    for key in ("reference_only", "measurement_only", "complete"):
        if report.get(key) is not True:
            raise RegressionError("Only complete reference quality sources can be combined")
    if (
        report.get("passed") is not False
        or report.get("quality_passed") is not False
        or report.get("generation_passed") is not False
        or "candidate_revision" in report
        or "reference_population_sha256" in report
        or report.get("calibration_errors")
        or report.get("generation_manifest_sha256") != generation_sha
        or report.get("contract") != manifest["contract"]
        or any(
            report.get(key) != manifest.get(key)
            for key in (
                "reference_revision",
                "harness_revision",
                "reference_protocol_sha256",
                "collection_id",
                "collection_backend",
                "buildkite_job_id",
                "measurement_jobs",
            )
        )
    ):
        raise RegressionError("Quality source differs from the reference generation or contains acceptance inputs")
    _quality_identity(report)
    cases = manifest["contract"]["profile"]["cases"]
    index = report.get("quality_case_index")
    if type(index) is not int or not 0 <= index < len(cases):
        raise RegressionError("Each quality source must select exactly one generation case")
    case = cases[index]["id"]
    if report.get("evaluated_cases") != [case] or set(report.get("suites", {})) != set(policy):
        raise RegressionError("Quality source case or native suite coverage differs")
    pixels = report.get("pixels", {})
    if (
        set(pixels) != {case}
        or pixels[case].get("candidate")
        or pixels[case].get("checks")
        or pixels[case].get("reference_checks")
    ):
        raise RegressionError("Reference quality source contains candidate pixel measurements")
    raw_reports = {}
    lock = read_json(ROOT / "tests/dfx/lingbot_world/evaluators.lock.json")
    labels = {f"case0_reference{repeat}" for repeat in range(3)}
    for suite in policy:
        if set(report["suites"][suite]) != {case}:
            raise RegressionError("Quality source native summary covers the wrong case")
        summary = report["suites"][suite][case]
        raw = read_json(directory / f"{suite}-raw.json")
        if (
            f"{suite}-raw.json" not in report["artifacts"]
            or raw.get("status") != "passed"
            or raw.get("errors")
            or raw.get("suite") != suite
            or set(raw.get("dimensions", [])) != set(policy[suite])
            or raw.get("repo_revision") != lock["repositories"][suite]["revision"]
            or set(raw.get("scores", {})) != labels
            or set(raw.get("native_provenance", {})) != labels
            or summary.get("checks") != []
            or summary.get("reference_scores") != [raw["scores"][f"case0_reference{repeat}"] for repeat in range(3)]
        ):
            raise RegressionError("Reference quality source raw scores or provenance are incomplete")
        raw_reports[suite] = raw
    return report, raw_reports


def combine_reference_quality(generation: Path, qualities: list[Path], *, output: Path) -> dict:
    """Preserve source reports, join their labels, then apply the builder's verifier."""
    generation = generation.resolve()
    qualities = [path.resolve() for path in qualities]
    output = output.resolve()
    if output.exists():
        raise RegressionError("Combined quality output is immutable; never overwrite an existing directory")
    if any(path == output or output in path.parents or path in output.parents for path in [generation, *qualities]):
        raise RegressionError("Combined quality output must be separate from its source directories")
    manifest = read_json(generation / "generation.json")
    profile = manifest["contract"]["profile"]
    cases = profile["cases"]
    if profile.get("num_chunks") != 160 or profile.get("case_shard", {}).get("count") != 6 or len(cases) != 3:
        raise RegressionError("Case-quality combination requires one exact three-case weekly shard")
    if len(qualities) != len(cases) or len(set(qualities)) != len(cases):
        raise RegressionError("Exactly three distinct per-case quality sources are required")
    policy = read_json(POLICY)["candidate_tolerances"][profile["platform"]]
    sources = [_source_report(path, manifest, sha256(generation / "generation.json"), policy) for path in qualities]
    source_paths = {report["quality_case_index"]: path for path, (report, _) in zip(qualities, sources)}
    sources.sort(key=lambda pair: pair[0]["quality_case_index"])
    if [row[0]["quality_case_index"] for row in sources] != list(range(len(cases))):
        raise RegressionError("Per-case quality sources must cover each shard case exactly once")
    first = sources[0][0]
    common = (
        "quality_policy_sha256",
        "evaluator_lock_sha256",
        "evaluator_packages",
        "evaluator_opencv_provider",
    )
    identities, collection_ids = set(), set()
    for report, _ in sources:
        identity = _quality_identity(report)
        if identity in identities or report["quality_collection_id"] in collection_ids:
            raise RegressionError("Per-case evaluator producers must be independent")
        identities.add(identity)
        collection_ids.add(report["quality_collection_id"])
        if any(report.get(key) != first.get(key) or report.get(key) is None for key in common):
            raise RegressionError("Per-case evaluator environment or policy differs")
        first_hardware = {"hardware": first.get("evaluator_hardware", {})}
        hardware = {"hardware": report.get("evaluator_hardware", {})}
        if not first_hardware["hardware"] or _semantic_contract(hardware) != _semantic_contract(first_hardware):
            raise RegressionError("Per-case evaluator hardware differs")
    combined = copy.deepcopy(first)
    job_id = os.environ.get("BUILDKITE_JOB_ID") or None
    combined.update(
        quality_case_index=None,
        evaluated_cases=[case["id"] for case in cases],
        status="reference_quality_collection_complete",
        quality_collection_id=str(uuid4()),
        quality_collection_backend="buildkite" if job_id else "local",
        quality_buildkite_job_id=job_id,
        suites={suite: {} for suite in policy},
        pixels={},
        quality_jobs=[],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reference-quality-", dir=output.parent) as temporary:
        staged = Path(temporary)
        raw_outputs = {
            suite: {"status": "passed", "suite": suite, "errors": [], "scores": {}, "native_provenance": {}}
            for suite in policy
        }
        for index, (report, raw_reports) in enumerate(sources):
            case = cases[index]["id"]
            source = source_paths[index]
            retained = staged / "sources" / f"case-{index}"
            shutil.copytree(source, retained)
            relative = str((retained / "quality.json").relative_to(staged))
            combined["quality_jobs"].append(
                {
                    "case_index": index,
                    "case_id": case,
                    "quality_collection_id": report["quality_collection_id"],
                    "quality_collection_backend": report["quality_collection_backend"],
                    "quality_buildkite_job_id": report["quality_buildkite_job_id"],
                    "manifest_path": relative,
                    "manifest_sha256": sha256(retained / "quality.json"),
                }
            )
            combined["pixels"][case] = report["pixels"][case]
            for suite, raw in raw_reports.items():
                combined["suites"][suite][case] = report["suites"][suite][case]
                if index == 0:
                    for field in ("protocol", "repo_revision", "dimensions", "mode", "scope"):
                        raw_outputs[suite][field] = raw.get(field)
                elif any(raw_outputs[suite][field] != raw.get(field) for field in ("protocol", "mode", "scope")):
                    raise RegressionError("Per-case native evaluator protocol differs")
                for repeat in range(3):
                    old, new = f"case0_reference{repeat}", f"case{index}_reference{repeat}"
                    raw_outputs[suite]["scores"][new] = raw["scores"][old]
                    raw_outputs[suite]["native_provenance"][new] = raw["native_provenance"][old]
        for suite, raw in raw_outputs.items():
            write_json(staged / f"{suite}-raw.json", raw)
        combined["artifacts"] = {
            str(path.relative_to(staged)): sha256(path) for path in staged.rglob("*") if path.is_file()
        }
        write_json(staged / "quality.json", combined)
        _load_cohort(generation, staged, sha256(PROTOCOL), read_json(POLICY))
        output.mkdir()
        shutil.copytree(staged, output, dirs_exist_ok=True)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", required=True, type=Path)
    parser.add_argument("--quality", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        combine_reference_quality(args.generation, args.quality, output=args.output)
    except (RegressionError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"Reference quality combination failed: {error}\n")


if __name__ == "__main__":
    main()
