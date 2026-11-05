# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Calibration builds from hashed reference-only cohorts, never candidates."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from benchmarks.lingbot_world.encoder_inference import BACKGROUND_CLIP_BATCH_SIZE
from benchmarks.lingbot_world.regression import RegressionError, read_json, sha256, write_json
from tests.dfx.lingbot_world import calibrate as builder
from vllm_omni.benchmarks.lingbot_world.workload import ChunkRecord, compute_metrics

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def campaign(tmp_path, monkeypatch, request):
    selection = getattr(request, "param", "cuda_nightly")
    profile_name = selection[0] if isinstance(selection, tuple) else selection
    source_profile = builder.ROOT / f"tests/dfx/lingbot_world/configs/{profile_name}.json"
    profile = read_json(source_profile)
    if isinstance(selection, tuple):
        profile = builder.profile_for_case_shard(profile, f"{selection[1]}/{selection[2]}")
        profile["case_shard"]["source_sha256"] = sha256(source_profile)
    policy = read_json(builder.POLICY)
    packages = {"numpy": "1.26.4", "opencv-python": "4.11.0.86", "scikit-image": "0.25.2"}
    contract = {
        "profile": profile,
        "deploy_sha256": sha256(builder.ROOT / profile["deploy_config"]),
        "engine_environment": {"optimization": "0"},
        "hardware": {
            "platform": profile["platform"],
            "cards": profile["cards"],
            "dies": profile["dies"],
            "devices": [{"name": "H100" if profile["platform"] == "cuda" else "Ascend", "total_bytes": 80_000_000_000}]
            * profile["dies"],
            "packages": dict(profile["inference_runtime"]["packages"]),
        },
    }
    snapshot = profile["inference_runtime"].get("source_snapshot_sha256")
    if snapshot is not None:
        lock = read_json(builder.ROOT / "tests/dfx/lingbot_world/runtime_sources.lock.json")
        contract["hardware"]["source_attestations"] = {
            "schema_version": 1,
            "source_snapshot_sha256": snapshot,
            "packages": copy.deepcopy(lock["packages"]),
        }
    generations, qualities = [], []
    for cohort in range(2):
        generation, quality = tmp_path / f"generation-{cohort}", tmp_path / f"quality-{cohort}"
        generation.mkdir()
        quality.mkdir()
        records = []
        for repeat in range(3):
            current = {}
            precision = generation / f"reference/{repeat}/precision.npz"
            precision.parent.mkdir(parents=True)
            precision.write_bytes(f"precision-{cohort}-{repeat}".encode())
            for case in profile["cases"]:
                arrivals, chunks = 0.0, []
                for index in range(profile["num_chunks"]):
                    interval = 1 + cohort * 0.09 + repeat * 0.001 if index == 0 else 0.8
                    arrivals += interval
                    chunks.append(ChunkRecord(index, arrivals, interval, 10, 9 if index == 0 else 12))
                metrics = compute_metrics(
                    chunks,
                    fps=profile["target_fps"],
                    warmup_chunks=profile["warmup_chunks"],
                    ttfc_s=chunks[0].arrival_s,
                    session_wall_s=arrivals,
                )
                raw = {
                    "git_revision": "c" * 40,
                    "server_revision": profile["reference_revision"],
                    "config": {
                        "target_fps": profile["target_fps"],
                        "warmup_chunks": profile["warmup_chunks"],
                        "chunk_slo_ms": None,
                        "playback_buffer_chunks": 1,
                    },
                    "sessions": [
                        {
                            "session_index": session,
                            "request_id": f"request-{cohort}-{repeat}-{case['id']}-{session}",
                            "metrics": metrics,
                            "chunks": [row.to_dict() for row in chunks],
                        }
                        for session in range(profile.get("measured_sessions", 1))
                    ],
                }
                directory = generation / f"reference/{repeat}/{case['id']}"
                directory.mkdir()
                write_json(directory / "benchmark.json", raw)
                (directory / "rollout.mp4").write_bytes(f"video-{cohort}-{repeat}-{case['id']}".encode())
                current[case["id"]] = builder._measured_metrics(raw, profile)
            records.append(current)
        manifest = {
            "reference_only": True,
            "measurement_only": True,
            "complete": True,
            "passed": False,
            "reference_protocol_sha256": sha256(builder.PROTOCOL),
            "contract": copy.deepcopy(contract),
            "records": {"reference": records},
            "reference_revision": profile["reference_revision"],
            "harness_revision": "c" * 40,
            "buildkite_job_id": f"job-{cohort}",
            "collection_id": str(uuid4()),
            "collection_backend": "buildkite",
            "artifacts": {
                str(path.relative_to(generation)): sha256(path) for path in generation.rglob("*") if path.is_file()
            },
        }
        write_json(generation / "generation.json", manifest)
        report = {
            "reference_only": True,
            "measurement_only": True,
            "complete": True,
            "passed": False,
            "quality_passed": False,
            "reference_protocol_sha256": sha256(builder.PROTOCOL),
            "contract": copy.deepcopy(contract),
            "reference_revision": profile["reference_revision"],
            "harness_revision": "c" * 40,
            "buildkite_job_id": manifest["buildkite_job_id"],
            "collection_id": manifest["collection_id"],
            "collection_backend": manifest["collection_backend"],
            "generation_manifest_sha256": sha256(generation / "generation.json"),
            "quality_policy_sha256": sha256(builder.POLICY),
            "evaluator_lock_sha256": sha256(builder.ROOT / "tests/dfx/lingbot_world/evaluators.lock.json"),
            "evaluator_packages": packages,
            "evaluator_opencv_provider": {"opencv_build_sha256": "b" * 64},
            "suites": {},
        }
        frames = (profile["num_chunks"] * 3 - 1) * 4 + 1
        coverage = {
            "gradients": False,
            "batch_size": BACKGROUND_CLIP_BATCH_SIZE,
            "encoded_images": frames,
            "forward_batches": (frames + BACKGROUND_CLIP_BATCH_SIZE - 1) // BACKGROUND_CLIP_BATCH_SIZE,
            "max_forward_batch": BACKGROUND_CLIP_BATCH_SIZE,
        }
        for suite, dimensions in policy["candidate_tolerances"][profile["platform"]].items():
            raw = {
                "status": "passed",
                "protocol": {"dynamic_degree_backend": "raft"},
                "scores": {},
                "native_provenance": {},
            }
            report["suites"][suite] = {}
            for index, case in enumerate(profile["cases"]):
                rows = []
                for repeat in range(3):
                    row = {
                        dimension: 0.0 if dimension == "dynamic_degree" else 0.8 + 0.001 * cohort + 0.0001 * repeat
                        for dimension in dimensions
                    }
                    rows.append(row)
                    raw["scores"][f"case{index}_reference{repeat}"] = row
                    raw["native_provenance"][f"case{index}_reference{repeat}"] = {"background_clip_inference": coverage}
                report["suites"][suite][case["id"]] = {"reference_scores": rows, "checks": []}
            write_json(quality / f"{suite}-raw.json", raw)
        report["artifacts"] = {str(path.relative_to(quality)): sha256(path) for path in quality.iterdir()}
        write_json(quality / "quality.json", report)
        generations.append(generation)
        qualities.append(quality)
    calls = []

    def pixels(first, second, output, **kwargs):
        assert "/reference/" in str(first) and "/reference/" in str(second)
        assert first != second
        calls.append((first, second))
        profile = contract["profile"]
        frames = (profile["num_chunks"] * 3 - 1) * 4 + 1
        return {
            "reference": {
                "frame_count": frames,
                "width": profile["width"],
                "height": profile["height"],
                "fps": profile["mux_fps"],
            },
            "candidate": {
                "frame_count": frames,
                "width": profile["width"],
                "height": profile["height"],
                "fps": profile["mux_fps"],
            },
            "decoded_frames": frames,
            "pixel_environment": {"packages": packages, "opencv_build_sha256": "b" * 64},
            "ssim": {"mean": 0.99},
            "total": {"psnr_db": 45, "psnr_is_infinite": False},
            "windows": [
                {
                    "index": index,
                    "start_frame": start,
                    "frame_count": min(32, frames - start),
                    "psnr_db": 45,
                    "psnr_is_infinite": False,
                }
                for index, start in enumerate(range(0, frames, 32))
            ],
            "uniform_frame_indices": {"reference": [], "candidate": []},
        }

    monkeypatch.setattr(builder, "compare_video_pair_in_environment", pixels)
    monkeypatch.setattr(builder, "verify_environment", lambda *args: packages)
    return {
        "generations": generations,
        "qualities": qualities,
        "calls": calls,
        "profile": profile,
        "output": tmp_path / "frozen",
    }


def _build(campaign):
    return builder.build_calibration(
        campaign["generations"],
        campaign["qualities"],
        protocol=builder.PROTOCOL,
        python=Path(sys.executable),
        output=campaign["output"],
    )


def test_freezes_both_cohorts_and_all_fifteen_pairs_without_claiming_acceptance(campaign):
    result = _build(campaign)
    assert result["complete"] and not result["full_acceptance"] and not result["independent_validation_complete"]
    assert len(campaign["calls"]) == 15 * len(campaign["profile"]["cases"])
    population = read_json(campaign["output"] / Path(campaign["profile"]["quality_reference_population"]).name)
    assert population["schema_version"] == 4
    assert not population["candidate_inputs_used"] and not population["candidate_allowances_changed"]
    for case in population["cases"].values():
        assert len(case["pixels"]) == len(case["pixel_pairs"]) == 15
        assert len(case["reference_videos"]) == 6
        assert all(len(rows) == 6 for rows in case["suites"].values())
    reference = read_json(campaign["output"] / Path(campaign["profile"]["reference_calibration"]).name)
    assert reference["ttfc_calibration"]["selected_floor_ms"] == 115
    assert all(row["check"]["passed"] for row in reference["performance_reference_null_checks"])
    frozen_hashes = dict(result["artifacts"])
    with pytest.raises(RegressionError, match="never overwrite"):
        _build(campaign)
    assert {name: sha256(campaign["output"] / name) for name in frozen_hashes} == frozen_hashes


@pytest.mark.parametrize(
    "mutation",
    [
        "candidate",
        "unhashed",
        "raw_corrupt",
        "duplicate_cohort",
        "runtime",
        "protocol",
        "native_candidate",
        "quality_mismatch",
    ],
)
def test_rejects_invalid_training_evidence_before_pixel_measurement(campaign, mutation):
    generation, quality = campaign["generations"][1], campaign["qualities"][1]
    path = generation / "generation.json"
    manifest = read_json(path)
    if mutation == "candidate":
        manifest["records"]["candidate"] = []
    elif mutation == "unhashed":
        manifest["artifacts"].pop(next(key for key in manifest["artifacts"] if key.endswith("rollout.mp4")))
    elif mutation == "raw_corrupt":
        (generation / next(key for key in manifest["artifacts"] if key.endswith("benchmark.json"))).write_text("{}")
    elif mutation == "duplicate_cohort":
        manifest["buildkite_job_id"] = "job-0"
    elif mutation == "runtime":
        manifest["contract"]["hardware"]["packages"]["vllm"] = "0.32.0"
    elif mutation == "protocol":
        manifest["reference_protocol_sha256"] = "f" * 64
    elif mutation == "native_candidate":
        raw_path = quality / "vbench-raw.json"
        raw = read_json(raw_path)
        raw["scores"]["case0_candidate0"] = next(iter(raw["scores"].values()))
        write_json(raw_path, raw)
    elif mutation == "quality_mismatch":
        manifest["harness_revision"] = "d" * 40
    write_json(path, manifest)
    report_path = quality / "quality.json"
    report = read_json(report_path)
    report["generation_manifest_sha256"] = sha256(path)
    if mutation == "runtime":
        report["contract"] = manifest["contract"]
    if mutation == "native_candidate":
        report["artifacts"]["vbench-raw.json"] = sha256(quality / "vbench-raw.json")
    write_json(report_path, report)
    with pytest.raises(RegressionError):
        _build(campaign)
    assert campaign["calls"] == []
    assert not campaign["output"].exists()


@pytest.mark.parametrize(
    "campaign", ["npu_nightly", "cuda_weekly", "npu_weekly", "cuda_long", "npu_long"], indirect=True
)
def test_full_profile_calibration_remains_bound_to_its_own_contract(campaign):
    result = _build(campaign)
    assert result["complete"] and not result["independent_validation_complete"]
    population = read_json(campaign["output"] / Path(campaign["profile"]["quality_reference_population"]).name)
    assert population["contract"]["profile"] == campaign["profile"]
    assert len(campaign["calls"]) == 15 * len(campaign["profile"]["cases"])


@pytest.mark.parametrize("mutation", ["coverage", "windows", "decoder", "uniform"])
def test_rejects_incomplete_or_changed_pixel_evidence(campaign, monkeypatch, mutation):
    compare = builder.compare_video_pair_in_environment

    def invalid_pixels(*args, **kwargs):
        row = compare(*args, **kwargs)
        if mutation == "coverage":
            row["decoded_frames"] -= 1
        elif mutation == "windows":
            row["windows"].pop()
        elif mutation == "decoder":
            row["pixel_environment"]["opencv_build_sha256"] = "f" * 64
        else:
            row["uniform_frame_indices"]["reference"] = [0]
        return row

    monkeypatch.setattr(builder, "compare_video_pair_in_environment", invalid_pixels)
    with pytest.raises(RegressionError, match="pixel|Pixel"):
        _build(campaign)
    assert not (campaign["output"] / Path(campaign["profile"]["reference_calibration"]).name).exists()


@pytest.mark.parametrize(
    "campaign",
    [(f"{platform}_weekly", index, 6) for platform in ("cuda", "npu") for index in range(6)],
    indirect=True,
)
def test_weekly_shards_freeze_their_own_three_case_population(campaign):
    result = _build(campaign)
    profile = campaign["profile"]
    shard = profile["case_shard"]
    assert len(profile["cases"]) == 3
    assert result["complete"] and not result["independent_validation_complete"]
    assert len(campaign["calls"]) == 45
    for key in ("reference_calibration", "quality_reference_population"):
        assert f"_shard{shard['index']}of{shard['count']}.json" in profile[key]
        assert (campaign["output"] / Path(profile[key]).name).is_file()
    population = read_json(campaign["output"] / Path(profile["quality_reference_population"]).name)
    assert population["contract"]["profile"] == profile
    assert set(population["cases"]) == {case["id"] for case in profile["cases"]}
    assert all(len(case["reference_videos"]) == 6 for case in population["cases"].values())


@pytest.mark.parametrize("campaign", [("cuda_weekly", 0, 6), ("npu_weekly", 5, 6)], indirect=True)
@pytest.mark.parametrize("mutation", ["source_hash", "full_profile_records", "different_shard_cases"])
def test_shard_calibration_rejects_unqualified_source_or_other_population(campaign, mutation):
    profile = copy.deepcopy(campaign["profile"])
    source = builder.ROOT / f"tests/dfx/lingbot_world/configs/{profile['platform']}_weekly.json"
    base = read_json(source)
    if mutation == "source_hash":
        profile["case_shard"]["source_sha256"] = "f" * 64
    elif mutation == "full_profile_records":
        for key in ("reference_calibration", "quality_reference_population"):
            profile[key] = base[key]
    else:
        profile["cases"] = base["cases"][1::6]
    for generation, quality in zip(campaign["generations"], campaign["qualities"]):
        manifest_path = generation / "generation.json"
        manifest = read_json(manifest_path)
        manifest["contract"]["profile"] = profile
        write_json(manifest_path, manifest)
        quality_path = quality / "quality.json"
        report = read_json(quality_path)
        report["contract"]["profile"] = profile
        report["generation_manifest_sha256"] = sha256(manifest_path)
        write_json(quality_path, report)
    with pytest.raises(RegressionError, match="frozen source profile"):
        _build(campaign)
    assert campaign["calls"] == []
    assert not campaign["output"].exists()


def _rewrite_collection_provenance(campaign, transform):
    for index, (generation, quality) in enumerate(zip(campaign["generations"], campaign["qualities"])):
        manifest_path = generation / "generation.json"
        quality_path = quality / "quality.json"
        manifest, report = read_json(manifest_path), read_json(quality_path)
        transform(index, manifest, report)
        write_json(manifest_path, manifest)
        report["generation_manifest_sha256"] = sha256(manifest_path)
        write_json(quality_path, report)


def test_local_reference_cohorts_retain_honest_uuid_provenance(campaign):
    def local(index, manifest, report):
        for document in (manifest, report):
            document.update(collection_backend="local", buildkite_job_id=None)

    _rewrite_collection_provenance(campaign, local)
    result = _build(campaign)
    assert result["complete"] and not result["independent_validation_complete"]
    population = read_json(campaign["output"] / Path(campaign["profile"]["quality_reference_population"]).name)
    identities = {f"local:{record['collection_id']}" for record in population["inputs"]}
    assert len(identities) == 2
    assert all(
        record["collection_backend"] == "local" and record["buildkite_job_id"] is None
        for record in population["inputs"]
    )
    for case in population["cases"].values():
        assert {video["generation_job_id"] for video in case["reference_videos"]} == identities


@pytest.mark.parametrize("mutation", ["malformed", "reused", "inconsistent_backend", "quality_identity", "missing"])
def test_reference_cohort_identity_cannot_be_missing_forged_or_reused(campaign, mutation):
    first_id = read_json(campaign["generations"][0] / "generation.json")["collection_id"]

    def invalidate(index, manifest, report):
        if index == 0:
            return
        if mutation == "malformed":
            manifest["collection_id"] = report["collection_id"] = "not-a-uuid"
        elif mutation == "reused":
            manifest["collection_id"] = report["collection_id"] = first_id
        elif mutation == "inconsistent_backend":
            manifest["collection_backend"] = report["collection_backend"] = "local"
        elif mutation == "quality_identity":
            report["collection_id"] = str(uuid4())
        else:
            del manifest["collection_id"]
            del report["collection_id"]

    _rewrite_collection_provenance(campaign, invalidate)
    with pytest.raises(RegressionError):
        _build(campaign)
    assert campaign["calls"] == []
    assert not campaign["output"].exists()


def _add_measurement_producers(campaign):
    def aggregate(index, manifest, report):
        generation = campaign["generations"][index]
        jobs = []
        for repeat in range(3):
            source = {
                key: value
                for key, value in manifest.items()
                if key not in {"records", "artifacts", "collection_id", "buildkite_job_id"}
            }
            source.update(
                repeat=repeat,
                collection_id=str(uuid4()),
                buildkite_job_id=f"producer-{index}-{repeat}",
                records={"reference": [manifest["records"]["reference"][repeat]]},
                artifacts={
                    path: digest
                    for path, digest in manifest["artifacts"].items()
                    if path.startswith(f"reference/{repeat}/")
                },
            )
            relative = f"reference/provenance/measurement-{repeat}.json"
            write_json(generation / relative, source)
            digest = sha256(generation / relative)
            manifest["artifacts"][relative] = digest
            jobs.append(
                {
                    "repeat": repeat,
                    "collection_id": source["collection_id"],
                    "collection_backend": source["collection_backend"],
                    "buildkite_job_id": source["buildkite_job_id"],
                    "manifest_path": relative,
                    "manifest_sha256": digest,
                }
            )
        manifest["measurement_jobs"] = jobs
        report["measurement_jobs"] = copy.deepcopy(jobs)

    _rewrite_collection_provenance(campaign, aggregate)


def test_aggregate_cohort_preserves_three_hashed_real_producer_identities(campaign):
    _add_measurement_producers(campaign)
    assert _build(campaign)["complete"]
    population = read_json(campaign["output"] / Path(campaign["profile"]["quality_reference_population"]).name)
    assert all(len(cohort["measurement_jobs"]) == 3 for cohort in population["inputs"])
    expected_jobs = {f"producer-{cohort}-{repeat}" for cohort in range(2) for repeat in range(3)}
    for case in population["cases"].values():
        assert {video["producer_job_id"] for video in case["reference_videos"]} == expected_jobs
        assert len({video["generation_job_id"] for video in case["reference_videos"]}) == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_uuid",
        "duplicate_job",
        "source_manifest",
        "unhashed_manifest",
        "quality_producers",
        "source_payload",
        "reused_producer",
        "missing_summary",
    ],
)
def test_aggregate_producer_provenance_cannot_be_invented_or_reused(campaign, mutation):
    _add_measurement_producers(campaign)
    first = read_json(campaign["generations"][0] / "generation.json")["measurement_jobs"][0]

    def invalidate(index, manifest, report):
        if index == 0:
            return
        if mutation == "missing_summary":
            manifest.pop("measurement_jobs")
            report.pop("measurement_jobs")
            return
        jobs = manifest["measurement_jobs"]
        job = jobs[0]
        path = campaign["generations"][index] / job["manifest_path"]
        if mutation == "duplicate_uuid":
            jobs[1]["collection_id"] = jobs[0]["collection_id"]
        elif mutation == "duplicate_job":
            jobs[1]["buildkite_job_id"] = jobs[0]["buildkite_job_id"]
        elif mutation == "unhashed_manifest":
            manifest["artifacts"].pop(job["manifest_path"])
        elif mutation == "quality_producers":
            report["measurement_jobs"].pop()
            return
        else:
            source = read_json(path)
            if mutation == "source_manifest":
                source["reference_revision"] = "e" * 40
            elif mutation == "source_payload":
                source["artifacts"].pop(next(iter(source["artifacts"])))
            else:
                for key in ("collection_id", "buildkite_job_id"):
                    source[key] = job[key] = first[key]
            write_json(path, source)
            manifest["artifacts"][job["manifest_path"]] = job["manifest_sha256"] = sha256(path)
        report["measurement_jobs"] = copy.deepcopy(jobs)

    _rewrite_collection_provenance(campaign, invalidate)
    with pytest.raises(RegressionError):
        _build(campaign)
    assert campaign["calls"] == []
    assert not campaign["output"].exists()


def test_reference_cohorts_cannot_mix_internally_consistent_harness_revisions(campaign):
    def change_harness(index, manifest, report):
        if index == 0:
            return
        manifest["harness_revision"] = report["harness_revision"] = "d" * 40
        generation = campaign["generations"][index]
        for relative in manifest["artifacts"]:
            if relative.endswith("benchmark.json"):
                raw = read_json(generation / relative)
                raw["git_revision"] = manifest["harness_revision"]
                write_json(generation / relative, raw)
                manifest["artifacts"][relative] = sha256(generation / relative)

    _rewrite_collection_provenance(campaign, change_harness)
    with pytest.raises(RegressionError, match="different benchmark harness revisions"):
        _build(campaign)
    assert campaign["calls"] == []
    assert not campaign["output"].exists()


@pytest.mark.parametrize("campaign", ["npu_nightly"], indirect=True)
@pytest.mark.parametrize("kind", ["performance", "quality"])
@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "missing_observed",
        "missing_frozen",
        "missing_both",
        "wrong_digest_both",
        "wrong_head_both",
        "changed_patch",
        "changed_flags",
        "bool_as_integer",
        "invalid_schema_both",
        "missing_files_both",
        "unknown_nested_both",
        "bool_as_integer_both",
        "unverified_both",
        "changed_patch_both",
    ],
)
def test_frozen_npu_records_bind_actual_source_snapshot(campaign, kind, mutation):
    _build(campaign)
    profile = campaign["profile"]
    population = read_json(campaign["output"] / Path(profile["quality_reference_population"]).name)
    observed = copy.deepcopy(population["contract"])
    if kind == "performance":
        frozen = read_json(campaign["output"] / Path(profile["reference_calibration"]).name)
        expected_hardware = frozen["hardware"]
    else:
        frozen = population
        expected_hardware = frozen["contract"]["hardware"]
    current_hardware = observed["hardware"]
    if mutation in {"missing_observed", "missing_both"}:
        current_hardware.pop("source_attestations")
    if mutation in {"missing_frozen", "missing_both"}:
        expected_hardware.pop("source_attestations")
    if mutation == "wrong_digest_both":
        for hardware in (current_hardware, expected_hardware):
            hardware["source_attestations"]["source_snapshot_sha256"] = "f" * 64
    if mutation == "wrong_head_both":
        for hardware in (current_hardware, expected_hardware):
            hardware["source_attestations"]["packages"]["vllm-ascend"]["head"] = "f" * 40
    if mutation == "changed_patch":
        current_hardware["source_attestations"]["packages"]["vllm-ascend"]["files"]["Dockerfile"]["worktree"][
            "sha256"
        ] = "f" * 64
    if mutation == "changed_flags":
        current_hardware["source_attestations"]["packages"]["vllm-ascend"]["index_flags_clean"] = False
    if mutation == "bool_as_integer":
        current_hardware["source_attestations"]["packages"]["vllm-ascend"]["index_flags_clean"] = 1
    if mutation == "invalid_schema_both":
        for hardware in (current_hardware, expected_hardware):
            hardware["source_attestations"]["schema_version"] = True
    for hardware in (current_hardware, expected_hardware):
        source = hardware.get("source_attestations", {}).get("packages", {}).get("vllm-ascend", {})
        if mutation == "missing_files_both":
            source.pop("files")
        elif mutation == "unknown_nested_both":
            source["unknown"] = True
        elif mutation == "bool_as_integer_both":
            source["index_flags_clean"] = 1
        elif mutation == "unverified_both":
            source["exact_snapshot_verified"] = False
        elif mutation == "changed_patch_both":
            source["files"]["Dockerfile"]["worktree"]["sha256"] = "f" * 64

    def verify():
        if kind == "performance":
            builder.verify_reference_calibration_contract(observed, frozen)
        else:
            report = read_json(campaign["qualities"][0] / "quality.json")
            builder.verify_quality_population_contract(
                frozen,
                observed,
                evaluator_packages=report["evaluator_packages"],
                evaluator_lock_sha256=frozen["evaluator_lock_sha256"],
                asset_lock_sha256=frozen["asset_lock_sha256"],
            )

    if mutation is None:
        verify()
    else:
        with pytest.raises(RegressionError, match="source"):
            verify()


@pytest.mark.parametrize("campaign", ["npu_nightly"], indirect=True)
@pytest.mark.parametrize("mutation", ["missing", "digest", "patch", "flags", "clean"])
def test_builder_rejects_forged_npu_source_snapshot_before_fitting(campaign, mutation):
    generation, quality = campaign["generations"][1], campaign["qualities"][1]
    path = generation / "generation.json"
    manifest = read_json(path)
    hardware = manifest["contract"]["hardware"]
    if mutation == "missing":
        hardware.pop("source_attestations")
    elif mutation == "digest":
        hardware["source_attestations"]["source_snapshot_sha256"] = "f" * 64
    else:
        source = hardware["source_attestations"]["packages"]["vllm-ascend"]
        if mutation == "patch":
            source["files"]["Dockerfile"]["worktree"]["sha256"] = "f" * 64
        elif mutation == "flags":
            source["index_flags_clean"] = False
        else:
            source["clean"] = True
    write_json(path, manifest)
    report_path = quality / "quality.json"
    report = read_json(report_path)
    report["contract"] = copy.deepcopy(manifest["contract"])
    report["generation_manifest_sha256"] = sha256(path)
    write_json(report_path, report)
    with pytest.raises(RegressionError, match="source"):
        _build(campaign)
    assert campaign["calls"] == []
    assert not campaign["output"].exists()
