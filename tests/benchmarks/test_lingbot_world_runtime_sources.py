# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real Git source snapshots must reject unknown or concealed vendor changes."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from benchmarks.lingbot_world.regression import RegressionError
from tests.dfx.lingbot_world import runtime_sources as sources

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def git(root: Path, *arguments: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments],
        stderr=subprocess.PIPE,
        env={key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
    )


def commit(root: Path) -> None:
    git(root, "add", ".")
    git(root, "-c", "user.name=Snapshot Test", "-c", "user.email=snapshot@example.test", "commit", "-qm", "fixture")


@dataclass
class EditableDistribution:
    root: Path

    def read_text(self, name: str) -> str:
        assert name == "direct_url.json"
        return json.dumps({"url": self.root.as_uri(), "dir_info": {"editable": True}})


@dataclass
class Fixture:
    roots: dict[str, Path]
    expected: dict[str, str]
    digest: str
    lock: dict

    def observe(self) -> dict[str, object]:
        return sources.attest_runtime_sources(self.expected, expected_sha256=self.digest)


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    roots = {name: tmp_path / name for name in ("vllm", "vllm-ascend", "catlass")}
    for root in roots.values():
        root.mkdir()
        git(root, "init", "-q")
        (root / "runtime.py").write_text("original runtime\n")
        (root / "runtime.py").chmod(0o644)
        if root == roots["vllm-ascend"]:
            (root / "Dockerfile").write_text("FROM supplier\nRUN old-url\n")
            (root / "Dockerfile").chmod(0o644)
        commit(root)
    ascend = roots["vllm-ascend"]
    git(ascend, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(roots["catlass"]), "vendor/catlass")
    commit(ascend)
    (ascend / "Dockerfile").write_text("FROM supplier\nRUN new-url\n")
    expected = {
        name: git(root, "rev-parse", "HEAD").decode().strip() for name, root in roots.items() if name != "catlass"
    }
    records: dict[str, dict] = {}
    for name, revision in expected.items():
        root = roots[name]
        index = git(root, "ls-files", "--stage", "-z")
        records[name] = {
            "head": revision,
            "head_tree": git(root, "rev-parse", "HEAD^{tree}").decode().strip(),
            "index_sha256": hashlib.sha256(index).hexdigest(),
            "tracked_file_count": len(index.split(b"\0")) - 1,
            "status_porcelain_z": git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all").decode(),
            "files": {},
            "gitlinks": {},
            "clean": name == "vllm",
            "exact_snapshot_verified": True,
            "index_flags_clean": True,
        }
    old = git(ascend, "show", "HEAD:Dockerfile")
    blob = git(ascend, "rev-parse", "HEAD:Dockerfile").decode().strip()
    patched = (ascend / "Dockerfile").read_bytes()
    records["vllm-ascend"]["files"]["Dockerfile"] = {
        "head_mode": "100644",
        "head_blob": blob,
        "head_bytes": len(old),
        "head_sha256": hashlib.sha256(old).hexdigest(),
        "index_mode": "100644",
        "index_blob": blob,
        "index_stage": "0",
        "worktree": {
            "type": "regular",
            "bytes": len(patched),
            "mode": "0o644",
            "sha256": hashlib.sha256(patched).hexdigest(),
            "git_blob": git(ascend, "hash-object", "--no-filters", "Dockerfile").decode().strip(),
        },
    }
    records["vllm-ascend"]["gitlinks"] = {
        "vendor/catlass": {
            "head": git(roots["catlass"], "rev-parse", "HEAD").decode().strip(),
            "status_porcelain_z": "",
            "clean": True,
            "index_flags_clean": True,
        }
    }
    lock = {"schema_version": 1, "packages": records}
    path = tmp_path / "runtime_sources.lock.json"
    path.write_text(json.dumps(lock, sort_keys=True) + "\n")
    monkeypatch.setattr(sources, "LOCK_PATH", path)
    monkeypatch.setattr(
        sources.importlib.metadata, "distribution", lambda package: EditableDistribution(roots[package])
    )
    return Fixture(roots, expected, hashlib.sha256(path.read_bytes()).hexdigest(), lock)


def test_exact_snapshot_retains_dirty_supplier_state_and_stable_contract(snapshot):
    observed = snapshot.observe()
    assert observed == snapshot.observe()
    assert observed == {
        "schema_version": 1,
        "source_snapshot_sha256": snapshot.digest,
        "packages": snapshot.lock["packages"],
    }
    ascend = observed["packages"]["vllm-ascend"]
    assert ascend["clean"] is False and ascend["exact_snapshot_verified"] is True
    assert ascend["status_porcelain_z"] == " M Dockerfile\0"
    assert ascend["files"]["Dockerfile"]["head_blob"] != ascend["files"]["Dockerfile"]["worktree"]["git_blob"]
    assert str(snapshot.roots["vllm"].parent) not in json.dumps(observed)
    assert git(snapshot.roots["vllm-ascend"], "status", "--porcelain=v1", "-z") == b" M Dockerfile\0"


@pytest.mark.parametrize(
    "mutation",
    [
        "docker_content",
        "docker_cleaned",
        "docker_staged",
        "runtime_staged",
        "runtime_changed",
        "untracked",
        "docker_executable",
        "docker_permissions",
        "docker_symlink",
        "docker_directory",
        "wrong_head",
        "assume_unchanged",
        "skip_worktree",
        "mode_hidden",
        "nested_assume",
        "nested_skip",
        "nested_changed",
        "nested_untracked",
        "fsmonitor_valid",
    ],
)
def test_real_git_rejects_every_unreviewed_source_state(snapshot, mutation):
    root = snapshot.roots["vllm-ascend"]
    docker = root / "Dockerfile"
    if mutation == "docker_content":
        docker.write_text("FROM unknown\nRUN new-url\n")
    elif mutation == "docker_cleaned":
        docker.write_bytes(git(root, "show", "HEAD:Dockerfile"))
    elif mutation == "docker_staged":
        git(root, "add", "Dockerfile")
    elif mutation in ("runtime_changed", "runtime_staged"):
        (root / "runtime.py").write_text("unknown runtime\n")
        if mutation == "runtime_staged":
            git(root, "add", "runtime.py")
    elif mutation == "untracked":
        (root / "unexpected.txt").write_text("unknown\n")
    elif mutation == "docker_executable":
        docker.chmod(0o755)
    elif mutation == "docker_permissions":
        docker.chmod(0o600)
    elif mutation == "docker_symlink":
        payload = root.parent / "same-content"
        payload.write_bytes(docker.read_bytes())
        docker.unlink()
        docker.symlink_to(payload)
    elif mutation == "docker_directory":
        docker.unlink()
        docker.mkdir()
    elif mutation == "wrong_head":
        commit(root)
    elif mutation == "fsmonitor_valid":
        hook = root / ".git/fsmonitor-fixture"
        hook.write_text('#!/bin/sh\nprintf "fixture-token\\0"\n')
        hook.chmod(0o755)
        git(root, "config", "core.fsmonitor", str(hook))
        git(root, "config", "core.fsmonitorHookVersion", "2")
        git(root, "update-index", "--fsmonitor")
        git(root, "status", "--porcelain")
        assert b"h runtime.py\0" in git(root, "ls-files", "-f", "-z")
    elif mutation in ("assume_unchanged", "skip_worktree", "mode_hidden"):
        if mutation == "mode_hidden":
            git(root, "config", "core.filemode", "false")
            (root / "runtime.py").chmod(0o755)
        else:
            flag = "--assume-unchanged" if mutation == "assume_unchanged" else "--skip-worktree"
            git(root, "update-index", flag, "runtime.py")
            (root / "runtime.py").write_text("concealed runtime\n")
    elif mutation.startswith("nested_"):
        nested = root / "vendor/catlass"
        if mutation == "nested_untracked":
            (nested / "unexpected.txt").write_text("unknown\n")
        else:
            if mutation in ("nested_assume", "nested_skip"):
                git(
                    nested,
                    "update-index",
                    "--assume-unchanged" if mutation == "nested_assume" else "--skip-worktree",
                    "runtime.py",
                )
            (nested / "runtime.py").write_text("unknown nested runtime\n")
    with pytest.raises(RegressionError, match="runtime source|Runtime source"):
        snapshot.observe()


@pytest.mark.parametrize(
    "field",
    [
        "head",
        "head_tree",
        "index_sha256",
        "tracked_file_count",
        "clean",
        "index_flags_clean",
        "extra",
        "package",
        "digest",
    ],
)
def test_pure_report_validator_rejects_forged_or_incomplete_attestations(snapshot, monkeypatch, field):
    observed = {
        "schema_version": 1,
        "source_snapshot_sha256": snapshot.digest,
        "packages": copy.deepcopy(snapshot.lock["packages"]),
    }
    record = observed["packages"]["vllm-ascend"]
    if field == "package":
        del observed["packages"]["vllm"]
    elif field == "digest":
        observed["source_snapshot_sha256"] = "0" * 64
    elif field == "extra":
        record["local_root"] = "/different/vendor/root"
    else:
        record[field] = not record[field] if isinstance(record[field], bool) else "wrong"
    monkeypatch.setattr(sources, "_git", lambda *args: pytest.fail("pure validator invoked Git"))
    monkeypatch.setattr(
        sources.importlib.metadata, "distribution", lambda name: pytest.fail("pure validator inspected runtime")
    )
    with pytest.raises(RegressionError, match="runtime source attestation"):
        sources.verify_source_attestations(observed, expected_sha256=snapshot.digest)


def test_lock_digest_and_original_revision_pins_are_required(snapshot):
    with pytest.raises(RegressionError, match="snapshot lock"):
        sources.attest_runtime_sources(snapshot.expected, expected_sha256="0" * 64)
    with pytest.raises(RegressionError, match="pinned revisions"):
        sources.attest_runtime_sources({**snapshot.expected, "vllm": "0" * 40}, expected_sha256=snapshot.digest)


@pytest.mark.parametrize(
    "provenance", [None, {}, {"url": "https://example.test/vendor"}, {"url": "file:///tmp/vendor"}]
)
def test_snapshot_requires_actual_editable_distribution_provenance(snapshot, monkeypatch, provenance):
    @dataclass
    class Distribution:
        def read_text(self, name: str) -> str | None:
            assert name == "direct_url.json"
            return None if provenance is None else json.dumps(provenance)

    monkeypatch.setattr(sources.importlib.metadata, "distribution", lambda name: Distribution())
    with pytest.raises(RegressionError, match="editable runtime source"):
        snapshot.observe()


def test_pure_validator_distinguishes_boolean_from_integer(snapshot):
    observed = {
        "schema_version": 1,
        "source_snapshot_sha256": snapshot.digest,
        "packages": copy.deepcopy(snapshot.lock["packages"]),
    }
    observed["packages"]["vllm"]["clean"] = 1
    with pytest.raises(RegressionError, match="runtime source attestation"):
        sources.verify_source_attestations(observed, expected_sha256=snapshot.digest)


def test_lock_schema_rejects_boolean_alias_for_integer(snapshot):
    malformed = {**snapshot.lock, "schema_version": True}
    sources.LOCK_PATH.write_text(json.dumps(malformed))
    digest = hashlib.sha256(sources.LOCK_PATH.read_bytes()).hexdigest()
    with pytest.raises(RegressionError, match="snapshot schema"):
        sources.attest_runtime_sources(snapshot.expected, expected_sha256=digest)


def test_lock_is_bound_to_observed_supplier_source_inventory():
    lock = json.loads(sources.LOCK_PATH.read_text())
    assert lock["supplier_image"].endswith("@sha256:4a666fd818058c26ce445fa890e31ba781d0d43c721631b0d43e2eb550fbef99")
    assert lock["supplier_inventory_sha256"] == "345502e0d69adb00dcf2bbb7b6143940c86e8e32a9649f9b420c360c792929e4"
    ascend = lock["packages"]["vllm-ascend"]
    assert ascend["head"] == "221fb096cd1cdf9f0b273da9b9624fbf4bb7f200"
    assert ascend["status_porcelain_z"] == " M Dockerfile\0" and ascend["clean"] is False
    assert set(ascend["files"]) == {"Dockerfile"}
    assert ascend["files"]["Dockerfile"]["worktree"]["git_blob"] == "93d5f0e8fbee4cf2effe2d8dcc8d143ba0458cc8"
