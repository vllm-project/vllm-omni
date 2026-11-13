# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise snapshot freezing and the CPU-to-GPU handoff without network or Docker."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 40
RUNNER = "b" * 40
BASE = "lmsysorg/sglang@sha256:" + "c" * 64
PROJECT = """[project]
dependencies = ["sglang==0.5.23; sys_platform != 'darwin'", "torch==2.14.1"]
[tool.uv]
override-dependencies = ["protobuf>=6.31.1,<7.0.0"]
"""
DOCKERFILE = f"ARG SGLANG_IMAGE={BASE}\nFROM ${{SGLANG_IMAGE}}\nARG UCX_COMMIT={'d' * 40}\n"


@pytest.fixture
def producer(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "weekly_builder", ROOT / ".buildkite/cuda/scripts/build_weekly_comparison.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("BUILDKITE_PIPELINE_SLUG", "vllm-omni")
    monkeypatch.setenv("BUILDKITE_BUILD_NUMBER", "123")
    return module


def test_resolve_main_once_and_read_only_the_frozen_snapshot(producer, monkeypatch):
    calls = []
    snapshots = []

    def output(*command):
        calls.append(command)
        return f"{SHA}\trefs/heads/main"

    def snapshot_file(revision, path):
        snapshots.append((revision, path))
        return {"pyproject.toml": PROJECT, "docker/Dockerfile": DOCKERFILE}[path]

    monkeypatch.setattr(producer, "output", output)
    monkeypatch.setattr(producer, "snapshot_file", snapshot_file)
    lock = producer.source_lock("build-id", RUNNER, producer.resolve_main())
    assert calls == [("git", "ls-remote", producer.REPOSITORY, "refs/heads/main")]
    assert snapshots == [(SHA, "pyproject.toml"), (SHA, "docker/Dockerfile")]
    assert lock["sglang_omni_revision"] == SHA
    assert lock["sglang_base_image"] == BASE
    assert lock["sglang_requirement"].startswith("sglang==0.5.23")
    assert lock["upstream_pyproject_sha256"] == hashlib.sha256(PROJECT.encode()).hexdigest()
    assert lock["upstream_dockerfile_sha256"] == hashlib.sha256(DOCKERFILE.encode()).hexdigest()
    assert lock["source_identity"] == {"pipeline_slug": "vllm-omni", "build_number": "123"}
    assert "build-id" in lock["image_tag"] and SHA[:12] in lock["image_tag"]


@pytest.mark.parametrize("base", ["lmsysorg/sglang:latest", "lmsysorg/sglang@sha256:123"])
def test_reject_mutable_or_incomplete_upstream_base(producer, monkeypatch, base):
    monkeypatch.setattr(producer, "output", lambda *args: f"{SHA}\trefs/heads/main")
    monkeypatch.setattr(
        producer,
        "snapshot_file",
        lambda revision, path: PROJECT if path == "pyproject.toml" else DOCKERFILE.replace(BASE, base),
    )
    with pytest.raises(ValueError, match="immutable"):
        producer.source_lock("build-id", RUNNER, SHA)


def test_missing_metadata_is_distinct_from_agent_failure(producer, monkeypatch):
    monkeypatch.setattr(producer.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 100))
    assert producer.metadata("key") is None
    monkeypatch.setattr(producer.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 1))
    with pytest.raises(subprocess.CalledProcessError):
        producer.metadata("key")


def test_retry_reuses_source_and_completed_image_before_uploading_jobs(producer, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BUILDKITE_BUILD_ID", "build-id")
    monkeypatch.setenv("BACKEND_COMPARISON_COMMIT", RUNNER)
    records: dict[str, dict] = {}
    events = []
    source = {"build_id": "build-id", "runner_commit": RUNNER, "sglang_omni_revision": SHA}
    environment = {**source, "image_reference": "image@sha256:" + "c" * 64}
    monkeypatch.setattr(producer, "metadata", records.get)
    monkeypatch.setattr(producer, "save_metadata", lambda key, value: records.update({key: value}))

    def resolve_main():
        events.append("resolve")
        return SHA

    monkeypatch.setattr(producer, "resolve_main", resolve_main)
    monkeypatch.setattr(producer, "source_lock", lambda *args: source)

    def build_environment(lock):
        assert records[producer.SOURCE_KEY] == source  # persisted before any Docker work
        events.append("build")
        return environment

    monkeypatch.setattr(producer, "build_environment", build_environment)
    monkeypatch.setattr(producer, "run", lambda *args: events.append(args))
    monkeypatch.setattr(producer, "output", lambda *args: RUNNER if args[0] == "git" else '{"steps": []}')
    monkeypatch.setattr(producer.subprocess, "run", lambda *args, **kwargs: events.append("upload"))
    producer.main()
    producer.main()
    assert events.count("resolve") == events.count("build") == 1
    assert json.loads(producer.LOCK_PATH.read_text()) == environment
    assert events.index("build") < events.index("upload")


def test_build_records_packages_and_digest_and_checks_installed_commit(producer, monkeypatch):
    commands = []
    source = {
        "image_tag": producer.IMAGE_REPOSITORY + ":unique-build",
        "sglang_omni_revision": SHA,
        "sglang_base_image": BASE,
        "ucx_commit": "d" * 40,
        "sglang_requirement": "sglang==0.5.23; sys_platform != 'darwin'",
    }
    packages = {"sglang": "0.5.23", "sglang-omni": "0.1", "torch": "2.14.1", "diffusers": "0.37.0"}
    installed_sha = SHA

    def output(*command):
        commands.append(command)
        if "inspect" in command:
            return json.dumps([producer.IMAGE_REPOSITORY + "@sha256:" + "e" * 64])
        if "cat" in command:
            return '{"schema_version": 1, "patches": []}'
        if "source_commit" in command[-1]:
            return json.dumps({"direct_url": None, "source_commit": "f" * 40})
        if "find_spec" in command[-1]:
            return "[]"
        if "direct_url.json" in command[-1]:
            return json.dumps({"vcs_info": {"commit_id": installed_sha}})
        return json.dumps(packages)

    monkeypatch.setattr(producer, "output", output)
    monkeypatch.setattr(producer, "run", lambda *args: commands.append(args))
    environment = producer.build_environment(source)
    assert environment["installed_packages"]["sglang"] == packages
    assert environment["sglang_installation"] == {"direct_url": None, "source_commit": "f" * 40}
    assert environment["image_digest"] == "sha256:" + "e" * 64
    assert environment["sglang_omni_installation"]["vcs_info"]["commit_id"] == SHA
    assert f"SGLANG_OMNI_COMMIT={SHA}" in commands[0]
    assert f"SGLANG_BASE_IMAGE={BASE}" in commands[0]
    assert all("--gpus" not in command for command in commands)
    installed_sha = "f" * 40
    commands.clear()
    with pytest.raises(ValueError, match="Installed SGLang-Omni commit"):
        producer.build_environment(source)
    assert not any("push" in command for command in commands)


def test_snapshot_download_failure_retries_the_original_resolved_main(producer, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BUILDKITE_BUILD_ID", "build-id")
    monkeypatch.setenv("BACKEND_COMPARISON_COMMIT", RUNNER)
    records: dict[str, dict] = {}
    resolutions = []
    snapshots = []
    monkeypatch.setattr(producer, "metadata", records.get)
    monkeypatch.setattr(producer, "save_metadata", lambda key, value: records.update({key: value}))

    def resolve_main():
        resolutions.append(SHA)
        return SHA

    monkeypatch.setattr(producer, "resolve_main", resolve_main)
    monkeypatch.setattr(producer, "output", lambda *args: RUNNER if args[0] == "git" else '{"steps": []}')
    monkeypatch.setattr(producer, "run", lambda *args: None)
    monkeypatch.setattr(producer.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(producer, "build_environment", lambda source: source)

    def snapshot_file(revision, path):
        assert records[producer.RESOLUTION_KEY]["sglang_omni_revision"] == SHA
        snapshots.append((revision, path))
        if len(snapshots) == 1:
            raise OSError("transient snapshot download failure")
        return {"pyproject.toml": PROJECT, "docker/Dockerfile": DOCKERFILE}[path]

    monkeypatch.setattr(producer, "snapshot_file", snapshot_file)
    with pytest.raises(OSError, match="transient"):
        producer.main()
    producer.main()
    assert resolutions == [SHA]
    assert snapshots == [(SHA, "pyproject.toml"), (SHA, "pyproject.toml"), (SHA, "docker/Dockerfile")]
