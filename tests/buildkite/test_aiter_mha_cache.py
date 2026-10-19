# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Behavior tests for the build-scoped BF16 AITER MHA artifact."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SCRIPT = Path(".buildkite/amd/scripts/aiter_mha_cache.py")
HEAD = "1" * 40
BASE = "2" * 40
CHECKOUT = "3" * 40


def _load_cache():
    spec = importlib.util.spec_from_file_location("aiter_mha_cache", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cache = _load_cache()


def _write_module(path: Path, payload: bytes) -> None:
    path.write_bytes(payload + cache.ARCH_MARKER + b"\n")


def _identity() -> dict[str, str]:
    identity = {field: f"value-{field}" for field in cache.KEY_FIELDS}
    identity["schema"] = cache.SCHEMA
    identity["module"] = cache.MODULE_NAME
    identity["arch"] = cache.ARCH
    return identity


def _env(**updates: str) -> dict[str, str]:
    values = {
        "BUILDKITE_BUILD_ID": "build-pr-8848",
        "BUILDKITE_BUILD_NUMBER": "14000",
        "BUILDKITE_JOB_ID": "job-z-image",
        "BUILDKITE_COMMIT": CHECKOUT,
        "BUILDKITE_BRANCH": "haic0:ci/amd-aiter-mha-cache",
        "BUILDKITE_PULL_REQUEST": "8848",
        "BUILDKITE_PULL_REQUEST_HEAD_COMMIT": HEAD,
        "BUILDKITE_PULL_REQUEST_USING_MERGE_REFSPEC": "true",
        "BUILDKITE_REPO": "https://github.com/vllm-project/vllm-omni.git",
        "VLLM_OMNI_AITER_MHA_TRUSTED_COMMIT": BASE,
    }
    values.update(updates)
    return values


def _provenance(identity: dict[str, str], **updates: str) -> dict[str, str]:
    value = cache.provenance_from_env(identity, _env(**updates), checkout=CHECKOUT)
    assert value is not None
    return value


def _details(
    *,
    build_id: str = "build-pr-8848",
    number: int = 14000,
    commit: str = HEAD,
    branch: str = "haic0:ci/amd-aiter-mha-cache",
    pull_request: str = "8848",
) -> dict:
    return {
        "id": build_id,
        "number": number,
        "commit_id": commit,
        "branch_name": branch,
        "pull_request": {"id": pull_request} if pull_request else None,
        "project": {
            "slug": cache.BUILDKITE_PIPELINE_SLUG,
            "provider": {"url": "https://github.com/vllm-project/vllm-omni"},
        },
    }


def test_cache_key_changes_when_the_runtime_identity_changes() -> None:
    original = cache.cache_key(_identity())
    changed = _identity()
    changed["hip"] = "6.5.0"
    changed_abi = _identity()
    changed_abi["python_cache_tag"] = "cpython-313"
    changed_core = _identity()
    changed_core["core"] = "different-core"

    assert cache.cache_key(_identity()) == original
    assert cache.cache_key(changed) != original
    assert cache.cache_key(changed_abi) != original
    assert cache.cache_key(changed_core) != original
    with pytest.raises(ValueError, match="missing"):
        cache.cache_key({**_identity(), "aiter": ""})


def test_pr_source_head_is_distinct_from_the_test_merge_checkout() -> None:
    env = _env()

    assert cache.source_commit(env) == HEAD
    assert cache.checkout_commit(env) == CHECKOUT
    provenance = cache.provenance_from_env(_identity(), env)
    assert provenance is not None
    assert provenance["commit"] == HEAD
    assert provenance["checkout_commit"] == CHECKOUT


def test_ldd_uses_runtime_library_paths_and_rejects_missing_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = tmp_path / cache.module_filename()
    module.touch()
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/lib")
    calls: list[tuple[list[str], dict[str, str]]] = []

    def resolved_runner(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs["env"]))
        return subprocess.CompletedProcess(command, 0, "libtorch_cpu.so => /torch/lib/libtorch_cpu.so\n", "")

    assert cache.ldd_resolves(
        module,
        runner=resolved_runner,
        library_paths=[Path("/torch/lib"), Path("/opt/rocm/lib")],
    )
    command, environment = calls[0]
    assert command == ["ldd", str(module)]
    assert environment["LD_LIBRARY_PATH"] == "/torch/lib:/opt/rocm/lib:/existing/lib"

    def missing_runner(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, "libtorch_hip.so => not found\n", "")

    assert not cache.ldd_resolves(module, runner=missing_runner, library_paths=[])


def test_image_dependency_identity_tracks_the_rocm_image_inputs(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    for relative in cache._IMAGE_DEPENDENCY_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    requirements = root / "requirements"
    requirements.mkdir()
    (requirements / "rocm.txt").write_text("torch\n", encoding="utf-8")

    original = cache.dependency_identity(root)
    assert len(original) == 64
    assert cache.dependency_identity(root) == original
    (requirements / "rocm.txt").write_text("torch==2\n", encoding="utf-8")
    assert cache.dependency_identity(root) != original


def test_stage_and_restore_verify_module_identity_and_provenance(tmp_path: Path) -> None:
    identity = _identity()
    job_dir = tmp_path / "job"
    restore_dir = tmp_path / "restore"
    job_dir.mkdir()
    restore_dir.mkdir()
    _write_module(job_dir / cache.module_filename(), b"compiled-module")

    published = cache.stage_publication(job_dir, tmp_path / "stage", identity, _provenance(identity))
    assert published is not None
    assert cache.publication_matches(published, identity, ldd_ok=lambda _path: True)

    source = cache.BuildSource(
        "build-pr-8848",
        14000,
        HEAD,
        "haic0:ci/amd-aiter-mha-cache",
        "8848",
        True,
    )
    assert cache.provenance_matches_source(
        published,
        source,
        identity,
        _env(),
        current_checkout=CHECKOUT,
    )
    assert cache.restore_module(published, restore_dir)
    assert (restore_dir / cache.module_filename()).read_bytes().startswith(b"compiled-module")

    module = published / cache.module_filename()
    module.write_bytes(module.read_bytes() + b"tampered")
    rejected = tmp_path / "rejected"
    rejected.mkdir()
    assert not cache.restore_module(published, rejected)
    assert not (rejected / cache.module_filename()).exists()


def test_invalid_module_is_not_staged(tmp_path: Path) -> None:
    identity = _identity()
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / cache.module_filename()).write_bytes(b"host-only-object")

    assert cache.stage_publication(job_dir, tmp_path / "stage", identity, _provenance(identity)) is None


def test_same_pr_artifact_cannot_cross_head_base_or_pr_identity(tmp_path: Path) -> None:
    identity = _identity()
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    _write_module(job_dir / cache.module_filename(), b"compiled-module")
    published = cache.stage_publication(job_dir, tmp_path / "stage", identity, _provenance(identity))
    assert published is not None
    source = cache.BuildSource(
        "build-pr-8848",
        14000,
        HEAD,
        "haic0:ci/amd-aiter-mha-cache",
        "8848",
        True,
    )

    assert not cache.provenance_matches_source(
        published,
        source,
        identity,
        _env(BUILDKITE_PULL_REQUEST_HEAD_COMMIT="4" * 40),
        current_checkout=CHECKOUT,
    )
    assert not cache.provenance_matches_source(
        published,
        source,
        identity,
        _env(BUILDKITE_PULL_REQUEST="8849"),
        current_checkout=CHECKOUT,
    )
    assert not cache.provenance_matches_source(
        published,
        source,
        identity,
        _env(VLLM_OMNI_AITER_MHA_TRUSTED_COMMIT="5" * 40),
        current_checkout=CHECKOUT,
    )
    assert not cache.provenance_matches_source(
        published,
        source,
        identity,
        _env(),
        current_checkout="6" * 40,
    )


def test_main_artifact_requires_the_exact_trusted_commit(tmp_path: Path) -> None:
    identity = _identity()
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    _write_module(job_dir / cache.module_filename(), b"main-module")
    main_env = _env(
        BUILDKITE_BUILD_ID="build-main",
        BUILDKITE_BUILD_NUMBER="13999",
        BUILDKITE_COMMIT=BASE,
        BUILDKITE_BRANCH="main",
        BUILDKITE_PULL_REQUEST="false",
        VLLM_OMNI_AITER_MHA_TRUSTED_COMMIT="7" * 40,
    )
    provenance = cache.provenance_from_env(identity, main_env, checkout=BASE)
    assert provenance is not None
    published = cache.stage_publication(job_dir, tmp_path / "stage", identity, provenance)
    assert published is not None
    source = cache.BuildSource("build-main", 13999, BASE, "main", "", False)

    assert cache.provenance_matches_source(published, source, identity, _env())
    assert not cache.provenance_matches_source(
        published,
        source,
        identity,
        _env(VLLM_OMNI_AITER_MHA_TRUSTED_COMMIT="8" * 40),
    )


def test_discovery_uses_server_metadata_and_excludes_the_current_build() -> None:
    listing = (
        '<a href="/vllm/vllm-omni-amd-ci/builds/14001">current</a>'
        '<a href="/vllm/vllm-omni-amd-ci/builds/14000">previous</a>'
    )

    def fetch(url: str) -> str:
        if "?" in url:
            return listing
        if url.endswith("/14001.json"):
            return json.dumps(_details(build_id="current", number=14001))
        if url.endswith("/14000.json"):
            return json.dumps(_details())
        raise AssertionError(url)

    sources = cache.discover_build_sources(
        branch="haic0:ci/amd-aiter-mha-cache",
        commit=HEAD,
        pull_request="8848",
        current_build_id="current",
        exact_checkout=True,
        fetch=fetch,
    )

    assert sources == [
        cache.BuildSource(
            "build-pr-8848",
            14000,
            HEAD,
            "haic0:ci/amd-aiter-mha-cache",
            "8848",
            True,
        )
    ]


def test_candidate_sources_queries_the_pr_head_not_the_test_merge_commit() -> None:
    requested: list[str] = []

    def fetch(url: str) -> str:
        requested.append(url)
        if url.endswith("/14000.json"):
            return json.dumps(_details())
        if f"commit={HEAD}" in url:
            return '<a href="/vllm/vllm-omni-amd-ci/builds/14000">previous</a>'
        return ""

    sources = cache.candidate_sources(_env(BUILDKITE_BUILD_ID="current"), fetch=fetch)

    assert [source.number for source in sources] == [14000]
    assert any(f"commit={HEAD}" in url for url in requested)
    assert not any(f"commit={CHECKOUT}" in url for url in requested)


def test_discovery_rejects_forged_branch_pr_and_provider_metadata() -> None:
    listing = "".join(f'<a href="/vllm/vllm-omni-amd-ci/builds/{number}">build</a>' for number in (1, 2, 3))
    details = {
        1: _details(number=1, branch="main"),
        2: _details(number=2, pull_request="9999"),
        3: _details(number=3),
    }
    details[3]["project"]["provider"]["url"] = "https://github.com/example/vllm-omni"

    def fetch(url: str) -> str:
        if "?" in url:
            return listing
        number = int(url.removesuffix(".json").rsplit("/", 1)[-1])
        return json.dumps(details[number])

    assert not cache.discover_build_sources(
        branch="haic0:ci/amd-aiter-mha-cache",
        commit=HEAD,
        pull_request="8848",
        current_build_id="current",
        exact_checkout=True,
        fetch=fetch,
    )


def test_download_restores_only_the_verified_build_artifact(tmp_path: Path) -> None:
    identity = _identity()
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    _write_module(compiled / cache.module_filename(), b"compiled-module")
    source_root = tmp_path / "source"
    assert cache.stage_publication(compiled, source_root, identity, _provenance(identity)) is not None

    def runner(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        destination = Path(command[4])
        shutil.copytree(source_root / cache.CACHE_DIRNAME, destination / cache.CACHE_DIRNAME)
        return subprocess.CompletedProcess(command, 0, "", "")

    source = cache.BuildSource(
        "build-pr-8848",
        14000,
        HEAD,
        "haic0:ci/amd-aiter-mha-cache",
        "8848",
        True,
    )
    restored = tmp_path / "restored"
    restored.mkdir()

    assert cache.download_from_source(
        source,
        restored,
        identity,
        _env(),
        runner=runner,
        current_checkout=CHECKOUT,
        ldd_ok=lambda _path: True,
    )
    assert (restored / cache.module_filename()).read_bytes().startswith(b"compiled-module")


def test_upload_targets_only_the_current_build_scoped_artifact(tmp_path: Path) -> None:
    calls: list[tuple[list[str], Path]] = []

    def runner(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs["cwd"]))
        return subprocess.CompletedProcess(command, 0, "", "")

    assert cache.upload_publication(tmp_path, _identity(), runner=runner)
    command, cwd = calls[0]
    assert command[:3] == ["buildkite-agent", "artifact", "upload"]
    assert command[3] == f"{cache.artifact_prefix(_identity()).as_posix()}/*"
    assert "--job" not in command
    assert "--build" not in command
    assert cwd == tmp_path


def test_seed_skips_the_cached_mha_module(tmp_path: Path) -> None:
    jit_dir = tmp_path / "jit"
    job_dir = tmp_path / "job"
    jit_dir.mkdir()
    job_dir.mkdir()
    (jit_dir / "module_aiter_core.so").write_bytes(b"core")
    _write_module(jit_dir / cache.module_filename(), b"image-copy")

    cache.seed_prebuilt_modules(job_dir, jit_dir)

    assert (job_dir / "module_aiter_core.so").read_bytes() == b"core"
    assert not (job_dir / cache.module_filename()).exists()


def test_prepare_exports_only_a_job_local_directory(monkeypatch, capsys) -> None:
    monkeypatch.setenv("BUILDKITE_JOB_ID", "unit.job/1")
    monkeypatch.setattr(cache, "installed_jit_dir", lambda: None)
    monkeypatch.setattr(cache, "collect_identity", lambda: None)

    assert cache.prepare() == 0
    captured = capsys.readouterr()
    assert "export AITER_JIT_DIR=/tmp/vllm-omni-aiter-mha-unit.job_1\n" in captured.out
    assert "export GPU_ARCHS=gfx942\n" in captured.out
    assert "VLLM_OMNI_AITER_MHA_JOB_DIR" in captured.out
    assert "HF_HOME" not in captured.out
    assert "STORE" not in captured.out
