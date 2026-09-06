# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Repository-root resolution for MiniMax Music 3's multi-component checkpoint.

The acoustic stage declares no ``model_subdir`` because it reads the repo root
itself, so on a hub-id deployment its model path is ``MiniMaxAI/MiniMax-Music3``
rather than a directory. Resolving that as a relative path silently points at
the server's working directory (issue #6638).
"""

from __future__ import annotations

import pytest

from vllm_omni.model_executor.models.minimax_music3.weights import (
    _COMPONENT_WEIGHT_NAME,
    _REQUIRED_COMPONENTS,
    _ROOT_MARKERS,
    resolve_repo_root,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_root(path):
    # A real snapshot carries weights for every component a stage loads, not
    # just the folders that identify the root. Building only ``_ROOT_MARKERS``
    # here would let a snapshot missing ``condition_encoder`` pass as complete,
    # which is the gap this suite regresses.
    for component in _REQUIRED_COMPONENTS:
        folder = path / component
        folder.mkdir(parents=True, exist_ok=True)
        (folder / _COMPONENT_WEIGHT_NAME).write_bytes(b"x")
    return path


def test_resolve_repo_root_accepts_the_root_itself(tmp_path):
    root = _make_root(tmp_path / "snapshot")
    assert resolve_repo_root(str(root)) == root


def test_resolve_repo_root_walks_up_from_a_model_subdir(tmp_path):
    root = _make_root(tmp_path / "snapshot")
    (root / "language_model").mkdir()
    assert resolve_repo_root(str(root / "language_model")) == root


def test_resolve_repo_root_resolves_a_hub_id_to_a_local_snapshot(monkeypatch, tmp_path):
    """A hub id must resolve through the cache, not against the working directory."""
    from huggingface_hub import HfApi

    root = _make_root(tmp_path / "snapshots" / "deadbeef")
    seen = []

    def fake_snapshot_download(self, repo_id, **kwargs):
        seen.append(kwargs)
        return str(root)

    monkeypatch.setattr(HfApi, "snapshot_download", fake_snapshot_download)

    assert resolve_repo_root("MiniMaxAI/MiniMax-Music3") == root
    # Cache-first, and only the component folders.
    assert seen[0]["local_files_only"] is True
    assert "transformer/*" in seen[0]["allow_patterns"]
    assert "qwen_7B/*" not in seen[0]["allow_patterns"]


def test_resolve_repo_root_falls_back_to_the_hub_when_the_cache_is_incomplete(monkeypatch, tmp_path):
    from huggingface_hub import HfApi

    root = _make_root(tmp_path / "snapshots" / "deadbeef")
    calls = []

    def fake_snapshot_download(self, repo_id, **kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise OSError("incomplete snapshot")
        return str(root)

    monkeypatch.setattr(HfApi, "snapshot_download", fake_snapshot_download)

    assert resolve_repo_root("MiniMaxAI/MiniMax-Music3") == root
    assert len(calls) == 2


def test_resolve_repo_root_retries_online_when_the_local_lookup_returns_a_partial_root(monkeypatch, tmp_path):
    """Hub clients differ on partial caches: some raise, some return the root.

    A returned partial root must fall through to the online call all the same,
    or startup fails without ever consulting the Hub.
    """
    from huggingface_hub import HfApi

    snapshot = tmp_path / "models--MiniMaxAI--MiniMax-Music3" / "snapshots" / "deadbeef"
    (snapshot / "language_model").mkdir(parents=True)
    calls = []

    def fake_snapshot_download(self, repo_id, **kwargs):
        calls.append(kwargs)
        if not kwargs.get("local_files_only"):
            _make_root(snapshot)
        return str(snapshot)

    monkeypatch.setattr(HfApi, "snapshot_download", fake_snapshot_download)

    assert resolve_repo_root("MiniMaxAI/MiniMax-Music3") == snapshot
    assert [bool(call.get("local_files_only")) for call in calls] == [True, False]


def test_resolve_repo_root_redownloads_weightless_marker_dirs(monkeypatch, tmp_path):
    """Marker folders without their component weights are an interrupted download."""
    from huggingface_hub import HfApi

    snapshot = tmp_path / "models--MiniMaxAI--MiniMax-Music3" / "snapshots" / "deadbeef"
    for marker in _ROOT_MARKERS:
        (snapshot / marker).mkdir(parents=True)
    calls = []

    def fake_snapshot_download(self, repo_id, **kwargs):
        calls.append(kwargs)
        if not kwargs.get("local_files_only"):
            _make_root(snapshot)
        return str(snapshot)

    monkeypatch.setattr(HfApi, "snapshot_download", fake_snapshot_download)

    assert resolve_repo_root("MiniMaxAI/MiniMax-Music3") == snapshot
    assert any(not call.get("local_files_only") for call in calls)


def test_resolve_repo_root_reports_the_original_reference_when_unresolvable(monkeypatch):
    from huggingface_hub import HfApi

    def fake_snapshot_download(self, repo_id, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(HfApi, "snapshot_download", fake_snapshot_download)

    with pytest.raises(FileNotFoundError, match="MiniMaxAI/MiniMax-Music3"):
        resolve_repo_root("MiniMaxAI/MiniMax-Music3")


def test_resolve_repo_root_downloads_components_for_a_cold_cache_model_subdir(monkeypatch, tmp_path):
    """Cold-cache stage 0: the talker gets an EXISTING ``language_model`` dir.

    Stage init pre-downloads only ``language_model/`` and ``tokenizer/``, so
    the marker walk fails although the repo id is recoverable from the cache
    layout and one snapshot_download away from working.
    """
    from huggingface_hub import HfApi

    snapshot = tmp_path / "models--MiniMaxAI--MiniMax-Music3" / "snapshots" / "deadbeef"
    (snapshot / "language_model").mkdir(parents=True)
    (snapshot / "tokenizer").mkdir()
    seen = []

    def fake_snapshot_download(self, repo_id, **kwargs):
        seen.append(repo_id)
        _make_root(snapshot)
        return str(snapshot)

    monkeypatch.setattr(HfApi, "snapshot_download", fake_snapshot_download)

    assert resolve_repo_root(str(snapshot / "language_model")) == snapshot
    assert seen == ["MiniMaxAI/MiniMax-Music3"]


def test_resolve_repo_root_does_not_guess_a_repo_id_for_plain_directories(monkeypatch, tmp_path):
    """A directory outside the HF cache has no recoverable repo id."""
    from huggingface_hub import HfApi

    def fake_snapshot_download(self, repo_id, **kwargs):
        raise AssertionError("must not reach the Hub for a plain directory")

    monkeypatch.setattr(HfApi, "snapshot_download", fake_snapshot_download)

    plain = tmp_path / "language_model"
    plain.mkdir()
    with pytest.raises(FileNotFoundError, match="expected sibling folders"):
        resolve_repo_root(str(plain))


def test_resolve_repo_root_rejects_a_snapshot_missing_condition_encoder(monkeypatch, tmp_path):
    """``condition_encoder`` is loaded by the acoustic stage but is not a root marker.

    A cache holding only the marker folders satisfies the name probe while the
    acoustic stage would still fail, so completeness must cover every required
    component rather than the identification markers alone.
    """
    from huggingface_hub import HfApi

    snapshot = tmp_path / "models--MiniMaxAI--MiniMax-Music3" / "snapshots" / "deadbeef"
    for marker in _ROOT_MARKERS:
        folder = snapshot / marker
        folder.mkdir(parents=True)
        (folder / _COMPONENT_WEIGHT_NAME).write_bytes(b"x")
    repaired = []

    def fake_snapshot_download(self, repo_id, **kwargs):
        if not kwargs.get("local_files_only"):
            repaired.append(repo_id)
            _make_root(snapshot)
        return str(snapshot)

    monkeypatch.setattr(HfApi, "snapshot_download", fake_snapshot_download)

    assert resolve_repo_root(str(snapshot)) == snapshot
    assert repaired == ["MiniMaxAI/MiniMax-Music3"]


def test_resolve_repo_root_rejects_a_partial_shard_run(monkeypatch, tmp_path):
    """One shard of a multi-shard component satisfies a bare glob but cannot load."""
    from huggingface_hub import HfApi

    stem = _COMPONENT_WEIGHT_NAME.removesuffix(".safetensors")
    snapshot = tmp_path / "models--MiniMaxAI--MiniMax-Music3" / "snapshots" / "deadbeef"
    for component in _REQUIRED_COMPONENTS:
        (snapshot / component).mkdir(parents=True)
        (snapshot / component / _COMPONENT_WEIGHT_NAME).write_bytes(b"x")
    # Replace one component with an interrupted 1-of-3 shard run.
    partial = snapshot / _ROOT_MARKERS[0]
    (partial / _COMPONENT_WEIGHT_NAME).unlink()
    (partial / f"{stem}-00001-of-00003.safetensors").write_bytes(b"x")
    repaired = []

    def fake_snapshot_download(self, repo_id, **kwargs):
        if not kwargs.get("local_files_only"):
            repaired.append(repo_id)
            for index in (2, 3):
                (partial / f"{stem}-0000{index}-of-00003.safetensors").write_bytes(b"x")
        return str(snapshot)

    monkeypatch.setattr(HfApi, "snapshot_download", fake_snapshot_download)

    assert resolve_repo_root(str(snapshot)) == snapshot
    assert repaired == ["MiniMaxAI/MiniMax-Music3"]


def test_resolve_repo_root_threads_revision_and_download_dir(monkeypatch, tmp_path):
    """A pinned revision must not be dropped when fetching the components.

    Resolving components against the default branch while the AR backbone is
    pinned elsewhere would combine weights from two different commits.
    """
    from huggingface_hub import HfApi

    snapshot = tmp_path / "snapshot"
    calls = []

    def fake_snapshot_download(self, repo_id, **kwargs):
        calls.append(kwargs)
        _make_root(snapshot)
        return str(snapshot)

    monkeypatch.setattr(HfApi, "snapshot_download", fake_snapshot_download)

    assert resolve_repo_root("MiniMaxAI/MiniMax-Music3", revision="abc123", download_dir="/tmp/hub-cache") == snapshot
    assert calls
    assert all(call.get("revision") == "abc123" for call in calls)
    assert all(call.get("cache_dir") == "/tmp/hub-cache" for call in calls)


def test_resolve_repo_root_reuses_the_snapshot_revision_from_the_cache_path(monkeypatch, tmp_path):
    """The commit is recoverable from ``snapshots/<sha>`` and must be reused.

    Re-resolving the repo id alone would fetch the default branch rather than
    the snapshot the caller is already sitting in.
    """
    from huggingface_hub import HfApi

    snapshot = tmp_path / "models--MiniMaxAI--MiniMax-Music3" / "snapshots" / "cafebabe"
    (snapshot / "language_model").mkdir(parents=True)
    calls = []

    def fake_snapshot_download(self, repo_id, **kwargs):
        calls.append(kwargs)
        _make_root(snapshot)
        return str(snapshot)

    monkeypatch.setattr(HfApi, "snapshot_download", fake_snapshot_download)

    assert resolve_repo_root(str(snapshot / "language_model")) == snapshot
    assert calls
    assert all(call.get("revision") == "cafebabe" for call in calls)


def test_resolve_repo_root_accepts_shard_names_the_loader_can_read(monkeypatch, tmp_path):
    """The completeness predicate must not be stricter than ``load_component_state``.

    The loader globs ``<stem>-*.safetensors`` and reads whatever it finds, so a
    component whose weights do not use the ``-<index>-of-<total>`` naming is
    loadable. Judging it incomplete would force a needless re-download and, with
    no network, fail on weights that are present and readable.
    """
    from huggingface_hub import HfApi

    stem = _COMPONENT_WEIGHT_NAME.removesuffix(".safetensors")
    root = tmp_path / "snapshot"
    for component in _REQUIRED_COMPONENTS:
        (root / component).mkdir(parents=True)
        (root / component / _COMPONENT_WEIGHT_NAME).write_bytes(b"x")
    unsharded = root / _ROOT_MARKERS[0]
    (unsharded / _COMPONENT_WEIGHT_NAME).unlink()
    (unsharded / f"{stem}-fp16.safetensors").write_bytes(b"x")

    def fake_snapshot_download(self, repo_id, **kwargs):
        raise AssertionError("must not reach the Hub for readable weights")

    monkeypatch.setattr(HfApi, "snapshot_download", fake_snapshot_download)

    assert resolve_repo_root(str(root)) == root


def test_resolve_repo_root_still_rejects_a_partial_run_beside_an_extra_file(monkeypatch, tmp_path):
    """An unparsable extra must not mask a genuinely short ``-of-`` run."""
    from huggingface_hub import HfApi

    stem = _COMPONENT_WEIGHT_NAME.removesuffix(".safetensors")
    root = tmp_path / "snapshot"
    for component in _REQUIRED_COMPONENTS:
        (root / component).mkdir(parents=True)
        (root / component / _COMPONENT_WEIGHT_NAME).write_bytes(b"x")
    partial = root / _ROOT_MARKERS[0]
    (partial / _COMPONENT_WEIGHT_NAME).unlink()
    (partial / f"{stem}-00001-of-00003.safetensors").write_bytes(b"x")
    (partial / f"{stem}-extra.safetensors").write_bytes(b"x")
    repaired = []

    def fake_snapshot_download(self, repo_id, **kwargs):
        repaired.append(repo_id)
        for index in (2, 3):
            (partial / f"{stem}-0000{index}-of-00003.safetensors").write_bytes(b"x")
        return str(root)

    monkeypatch.setattr(HfApi, "snapshot_download", fake_snapshot_download)

    with pytest.raises(FileNotFoundError, match="components are incomplete"):
        resolve_repo_root(str(root))
    assert repaired == []
