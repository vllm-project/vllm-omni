# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for comparison provenance, independent of server/GPU dependencies."""

import hashlib
import json
from pathlib import Path

import pytest

from tests.dfx.perf.scripts import experiment_metadata, sglang_omni_server

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_provenance_uses_actual_checkout_and_files(monkeypatch):
    # A Buildkite commit can differ from the checked-out runner/image sources.
    monkeypatch.setenv("BUILDKITE_COMMIT", "b" * 40)
    monkeypatch.setenv("BUILDKITE_PIPELINE_SLUG", "approved-comparison-pipeline")
    monkeypatch.setattr(experiment_metadata.subprocess, "check_output", lambda *args, **kwargs: "a" * 40 + "\n")
    metadata = experiment_metadata.comparison_provenance(
        test_config="tests/dfx/perf/tests/test_qwen3_omni_sglang.json",
        server_config="tests/dfx/perf/deploy/sgl_qwen3_omni.yaml",
        server_revision="22d4ace88f9547777b819a69d7691204bb9ce847",
    )
    assert metadata["schema_version"] == 1
    assert metadata["runner_commit"] == "a" * 40
    buildkite = metadata["buildkite"]
    assert isinstance(buildkite, dict)
    assert buildkite["commit"] == "b" * 40
    assert buildkite["pipeline_slug"] == "approved-comparison-pipeline"
    assert metadata["server_type"] == "sglang-omni"
    runners = metadata["runner_files"]
    patches = metadata["server_patches"]
    assert isinstance(runners, list) and isinstance(patches, list)
    assert len(runners) == 4
    assert all(len(item["sha256"]) == 64 for item in runners)
    assert patches[0]["path"] == "docker/patches/sglang-omni-ignore-eos.patch"
    assert "HF_TOKEN" not in json.dumps(metadata)


def test_fingerprint_detects_changed_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_metadata, "_REPO_ROOT", tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("chunk: 4\n", encoding="utf-8")
    before = experiment_metadata.file_fingerprint("config.yaml")
    config.write_text("chunk: 25\n", encoding="utf-8")
    after = experiment_metadata.file_fingerprint(config)
    assert before["path"] == after["path"] == "config.yaml"
    assert before["sha256"] != after["sha256"]


def test_diffusers_provenance_records_applied_patches_and_audit(monkeypatch):
    monkeypatch.setenv("COMPARISON_VIDEO_PATCH", "1")
    monkeypatch.setenv("COMPARISON_DIFFUSERS_PATCHES", "1")
    monkeypatch.setattr(experiment_metadata.subprocess, "check_output", lambda *args, **kwargs: "a" * 40 + "\n")
    metadata = experiment_metadata.comparison_provenance(
        test_config="tests/dfx/perf/tests/test_hunyuanvideo15_sglang.json",
        server_config=None,
        server_revision="0.5.22",
        server_type="sglang-diffusion",
    )
    patches = metadata["server_patches"]
    runners = metadata["runner_files"]
    assert isinstance(patches, list) and isinstance(runners, list)
    assert [item["path"] for item in patches] == [
        "docker/patches/sglang-video-image-reference.patch",
        "docker/patches/sglang-diffusion-config-only-guider.patch",
        "docker/patches/sglang-diffusers-revision.patch",
    ]
    assert "tests/dfx/perf/scripts/audit_diffusers_snapshot.py" in {item["path"] for item in runners}


@pytest.fixture
def weekly_environment(tmp_path, monkeypatch):
    lock = {
        "schema_version": 1,
        "runner_commit": "a" * 40,
        "build_id": "build-id",
        "source_identity": {"pipeline_slug": "vllm-omni", "build_number": "20000"},
        "image_digest": "sha256:" + "d" * 64,
        "sglang_omni_revision": "c" * 40,
        "installed_packages": {"sglang": {"sglang": "0.5.23"}},
    }
    path = tmp_path / "comparison-environment.json"
    path.write_text(json.dumps(lock) + "\n")
    monkeypatch.setenv("COMPARISON_ENVIRONMENT_LOCK", str(path))
    for key, value in {
        "BUILDKITE_BUILD_ID": "build-id",
        "BUILDKITE_PIPELINE_SLUG": "vllm-omni",
        "BUILDKITE_BUILD_NUMBER": "20000",
        "COMPARISON_IMAGE_DIGEST": lock["image_digest"],
        "COMPARISON_SGLANG_OMNI_REVISION": lock["sglang_omni_revision"],
        "COMPARISON_SGLANG_REVISION": "0.5.23",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(experiment_metadata.subprocess, "check_output", lambda *args, **kwargs: "a" * 40 + "\n")
    return path


@pytest.mark.parametrize("server_type,revision", [("sglang-omni", "c" * 40), ("sglang-diffusion", "0.5.23")])
def test_weekly_result_carries_exact_environment_artifact(weekly_environment, server_type, revision):
    metadata = experiment_metadata.comparison_provenance(
        test_config="tests/dfx/perf/tests/test_qwen3_omni_sglang.json",
        server_config=None,
        server_revision=revision,
        server_type=server_type,
    )
    assert metadata["environment_lock"] == json.loads(weekly_environment.read_bytes())
    assert metadata["environment_lock_sha256"] == hashlib.sha256(weekly_environment.read_bytes()).hexdigest()
    assert metadata["image_digest"] == "sha256:" + "d" * 64


@pytest.mark.parametrize(
    "field,value",
    [("build_id", "other-build"), ("runner_commit", "b" * 40), ("image_digest", "sha256:" + "f" * 64)],
)
def test_weekly_rejects_environment_from_another_execution(weekly_environment, field, value):
    lock = json.loads(weekly_environment.read_bytes())
    lock[field] = value
    weekly_environment.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="does not belong"):
        experiment_metadata.comparison_environment("a" * 40)


@pytest.mark.parametrize("server_type,revision", [("sglang-omni", "e" * 40), ("sglang-diffusion", "0.5.22")])
def test_weekly_rejects_server_different_from_environment_lock(weekly_environment, server_type, revision):
    with pytest.raises(ValueError, match="Running server revision"):
        experiment_metadata.comparison_provenance(
            test_config="tests/dfx/perf/tests/test_qwen3_omni_sglang.json",
            server_config=None,
            server_revision=revision,
            server_type=server_type,
        )


@pytest.mark.parametrize("direct_url", [{"vcs_info": {"commit_id": "c" * 40}}, {}])
def test_revision_comes_from_server_python(monkeypatch, direct_url):
    calls = []

    def capture(command, **kwargs):
        calls.append(command)
        return json.dumps(direct_url)

    monkeypatch.setattr(sglang_omni_server.subprocess, "check_output", capture)
    monkeypatch.setattr(sglang_omni_server.shutil, "which", lambda _: "/opt/sglang/bin/sgl-omni")
    revision = sglang_omni_server._installed_revision("sgl-omni")
    assert revision == direct_url.get("vcs_info", {}).get("commit_id")
    assert calls[0][0] == "/opt/sglang/bin/python3"
    assert "distribution('sglang-omni')" in calls[0][2]


def test_comparison_config_has_explicit_reproducible_workload():
    root = Path(__file__).parents[4]
    config = json.loads((root / "tests/dfx/perf/tests/test_qwen3_omni_sglang.json").read_text())[0]
    params = config["benchmark_params"][0]
    assert experiment_metadata.configured_num_gpus(config) == 2
    workload = experiment_metadata.resolved_workload(params, num_prompts=10, num_warmups=2, request_rate=0.1)
    assert workload["seed"] == 0
    assert workload["num_warmups"] == 2
    assert workload["num_prompts"] == 10
    assert workload["request_rate"] == 0.1
    assert workload["max_concurrency"] is None
    extra_body = workload["extra_body"]
    assert isinstance(extra_body, dict)
    assert extra_body["seed"] == 42
    assert extra_body["audio"] == {"voice": "Chelsie", "format": "wav"}
    assert params["num_prompts"] == [10]


def test_gpu_count_without_hardware_declaration_is_unknown():
    assert experiment_metadata.configured_num_gpus({"mark": ["full_model", "omni"]}) is None


def test_resolved_sweep_preserves_unknown_seed_and_excludes_baselines():
    workload = experiment_metadata.resolved_workload(
        {"num_prompts": [10, 20], "request_rate": [1, 2], "baseline": {"H100": {"throughput": 1}}},
        num_prompts=20,
        num_warmups=8,
        request_rate="inf",
        max_concurrency=8,
    )
    assert workload == {
        "num_prompts": 20,
        "request_rate": "inf",
        "max_concurrency": 8,
        "num_warmups": 8,
        "seed": None,
    }
