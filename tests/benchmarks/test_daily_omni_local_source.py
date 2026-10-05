# SPDX-License-Identifier: Apache-2.0
"""Offline Daily-Omni sources: local mirrors must not fall back to the Hub id."""

import json
import os
import tarfile
from pathlib import Path

import pytest

from vllm_omni.benchmarks.data_modules.daily_omni_dataset import (
    DailyOmniDataset,
    daily_omni_local_qa_json,
    daily_omni_local_videos_dir,
    ensure_daily_omni_hub_root,
    resolve_daily_omni_local_root,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]

_HUB_ID = "liarliar/Daily-Omni"


def _write_qa_json(root: Path) -> Path:
    qa = root / "qa.json"
    qa.write_text(json.dumps([{"Question": "q", "Answer": "A", "video_id": "vid0"}]), encoding="utf-8")
    return qa


def _stage_offline_hub_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, with_qa: bool = True) -> Path:
    """Build a hub cache holding ``liarliar/Daily-Omni`` and cut off all network access.

    Reproduces the air-gapped NPU runner: the dataset is pre-staged with ``hf download``, so it
    only exists in the *hub* cache -- never in the ``datasets`` cache that ``load_dataset`` reads.
    """
    constants = pytest.importorskip("huggingface_hub").constants

    cache = tmp_path / "hub"
    repo_dir = cache / "datasets--liarliar--Daily-Omni"
    snapshot = repo_dir / "snapshots" / "deadbeef"
    snapshot.mkdir(parents=True)
    if with_qa:
        _write_qa_json(snapshot)
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text("deadbeef", encoding="utf-8")

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    return snapshot


def test_resolve_local_root_returns_none_for_hub_id() -> None:
    assert resolve_daily_omni_local_root("liarliar/Daily-Omni") is None
    assert resolve_daily_omni_local_root("") is None
    assert resolve_daily_omni_local_root(None) is None


def test_plain_local_mirror(tmp_path: Path) -> None:
    _write_qa_json(tmp_path)
    (tmp_path / "Videos" / "vid0").mkdir(parents=True)

    root = resolve_daily_omni_local_root(str(tmp_path))
    assert root == tmp_path.resolve()
    assert daily_omni_local_qa_json(root) == (tmp_path / "qa.json").resolve()
    assert daily_omni_local_videos_dir(root) == root / "Videos"


def test_hf_cache_dir_resolves_to_snapshot(tmp_path: Path) -> None:
    cache_dir = tmp_path / "datasets--liarliar--Daily-Omni"
    snapshot = cache_dir / "snapshots" / "deadbeef"
    snapshot.mkdir(parents=True)
    _write_qa_json(snapshot)
    (cache_dir / "refs").mkdir()
    (cache_dir / "refs" / "main").write_text("deadbeef", encoding="utf-8")

    root = resolve_daily_omni_local_root(str(cache_dir))
    assert root == snapshot.resolve()
    assert daily_omni_local_qa_json(root) == (snapshot / "qa.json").resolve()


def test_hf_cache_dir_without_refs_picks_newest_snapshot(tmp_path: Path) -> None:
    cache_dir = tmp_path / "datasets--liarliar--Daily-Omni"
    older = cache_dir / "snapshots" / "aaa"
    newer = cache_dir / "snapshots" / "bbb"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    os.utime(older, (1, 1))
    os.utime(newer, (10, 10))

    assert resolve_daily_omni_local_root(str(cache_dir)) == newer.resolve()


def test_videos_tar_is_extracted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
    root = tmp_path / "mirror"
    (root / "Videos" / "vid0").mkdir(parents=True)
    (root / "Videos" / "vid0" / "vid0_video.mp4").write_bytes(b"fake")
    with tarfile.open(root / "Videos.tar", "w") as tf:
        tf.add(root / "Videos", arcname="Videos")
    # Only the tarball is left in the mirror.
    for child in (root / "Videos").rglob("*"):
        if child.is_file():
            child.unlink()
    (root / "Videos" / "vid0").rmdir()
    (root / "Videos").rmdir()

    videos_dir = daily_omni_local_videos_dir(root)
    assert videos_dir is not None
    assert (videos_dir / "vid0" / "vid0_video.mp4").is_file()

    # Second call reuses the extraction instead of redoing it.
    assert daily_omni_local_videos_dir(root) == videos_dir


def test_missing_qa_json_reports_none(tmp_path: Path) -> None:
    root = resolve_daily_omni_local_root(str(tmp_path))
    assert daily_omni_local_qa_json(root) is None
    assert daily_omni_local_videos_dir(root) is None


def test_hub_id_resolves_to_offline_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = _stage_offline_hub_cache(tmp_path, monkeypatch)

    root = ensure_daily_omni_hub_root(_HUB_ID)

    assert root == snapshot.resolve()
    assert daily_omni_local_qa_json(root) == (snapshot / "qa.json").resolve()


def test_dataset_loads_hub_id_without_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression this file exists for: a Hub id must not require ``load_dataset``."""
    snapshot = _stage_offline_hub_cache(tmp_path, monkeypatch)
    # Any use of `datasets` here means we went back to the path that cannot read the hub cache.
    monkeypatch.setattr("vllm_omni.benchmarks.data_modules.daily_omni_dataset.load_dataset", None)

    dataset = DailyOmniDataset(dataset_path=_HUB_ID, random_seed=0)

    assert dataset.qa_json_path == (snapshot / "qa.json").resolve()
    assert [row["video_id"] for row in dataset.data] == ["vid0"]


def test_uncached_hub_id_reports_both_attempts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stage_offline_hub_cache(tmp_path, monkeypatch, with_qa=False)
    monkeypatch.setattr("vllm_omni.benchmarks.data_modules.daily_omni_dataset.load_dataset", None)

    with pytest.raises(RuntimeError) as excinfo:
        DailyOmniDataset(dataset_path="someone-else/Daily-Omni", random_seed=0)

    message = str(excinfo.value)
    assert "someone-else/Daily-Omni" in message
    assert "snapshot_download failed" in message
    assert "--daily-omni-qa-json" in message
