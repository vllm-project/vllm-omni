# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline Video-MME sources: local mirrors must not fall back to a Hub id."""

from pathlib import Path

import pytest

from vllm_omni.benchmarks.data_modules.videomme_dataset import (
    VideoMMESampleRequest,
    ensure_videomme_videos_extracted,
    resolve_videomme_local_root,
    videomme_local_parquet,
    videomme_local_subtitle_dir,
    videomme_local_video_dir,
)
from vllm_omni.benchmarks.data_modules.videomme_eval import (
    compute_videomme_accuracy_metrics,
    extract_characters_regex,
    normalize_gold_answer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


def _write_parquet_placeholder(root: Path) -> Path:
    path = root / "videomme" / "test-00000-of-00001.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"parquet")
    return path


def test_resolve_local_root_returns_none_for_hub_id() -> None:
    assert resolve_videomme_local_root("lmms-eval/Video-MME") is None
    assert resolve_videomme_local_root("") is None
    assert resolve_videomme_local_root(None) is None


def test_plain_local_mirror(tmp_path: Path) -> None:
    pq = _write_parquet_placeholder(tmp_path)
    video = tmp_path / "video" / "abc.mp4"
    video.parent.mkdir()
    video.write_bytes(b"fake")
    (tmp_path / "subtitle").mkdir()
    (tmp_path / "subtitle" / "abc.srt").write_text("1\n", encoding="utf-8")

    root = resolve_videomme_local_root(str(tmp_path))
    assert root == tmp_path.resolve()
    assert videomme_local_parquet(root) == pq.resolve()
    assert videomme_local_video_dir(root) == tmp_path.resolve() / "video"
    assert videomme_local_subtitle_dir(root) == tmp_path.resolve() / "subtitle"


def test_hf_cache_dir_resolves_to_snapshot(tmp_path: Path) -> None:
    cache_dir = tmp_path / "datasets--lmms-eval--Video-MME"
    snapshot = cache_dir / "snapshots" / "deadbeef"
    snapshot.mkdir(parents=True)
    _write_parquet_placeholder(snapshot)
    (cache_dir / "refs").mkdir()
    (cache_dir / "refs" / "main").write_text("deadbeef", encoding="utf-8")

    root = resolve_videomme_local_root(str(cache_dir))
    assert root == snapshot.resolve()
    assert videomme_local_parquet(root) is not None


def test_hf_cache_dir_with_unusable_ref_falls_back_to_revision(tmp_path: Path) -> None:
    cache_dir = tmp_path / "datasets--lmms-eval--Video-MME"
    snapshot = cache_dir / "snapshots" / "deadbeef"
    snapshot.mkdir(parents=True)
    _write_parquet_placeholder(snapshot)
    (cache_dir / "refs").mkdir()
    # An empty ref must not resolve to snapshots/, which holds no dataset files.
    (cache_dir / "refs" / "main").write_text("", encoding="utf-8")

    assert resolve_videomme_local_root(str(cache_dir)) == snapshot.resolve()


def test_extract_videos_rejects_empty_tree(tmp_path: Path) -> None:
    (tmp_path / "video").mkdir()

    with pytest.raises(FileNotFoundError):
        ensure_videomme_videos_extracted(tmp_path)
    assert not (tmp_path / ".videomme_videos_extracted").exists()


def test_nested_unzipped_videos_are_discovered(tmp_path: Path) -> None:
    nested = tmp_path / "videos" / "videos_chunked_01" / "data" / "xyz.mp4"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"fake")

    assert videomme_local_video_dir(tmp_path) == tmp_path / "videos"


def test_extract_characters_regex_matches_official_prefixes() -> None:
    assert extract_characters_regex("The best answer is B.") == "B"
    assert extract_characters_regex("Answer: C") == "C"
    assert extract_characters_regex("I think D is correct") == "D"
    assert extract_characters_regex("no letter here at all in this long sentence") is None


def test_normalize_gold_answer() -> None:
    assert normalize_gold_answer("a") == "A"
    assert normalize_gold_answer("B. content") == "B"
    assert normalize_gold_answer("") is None


class _Out:
    def __init__(self, text: str, *, success: bool = True) -> None:
        self.generated_text = text
        self.success = success
        self.error = "" if success else "http fail"


def test_compute_videomme_accuracy_metrics() -> None:
    reqs = [
        VideoMMESampleRequest(
            prompt="q",
            prompt_len=1,
            expected_output_len=8,
            videomme_gold_answer="A",
            videomme_video_id="v1",
            videomme_question_id="q1",
            videomme_duration="short",
            videomme_domain="Knowledge",
            videomme_sub_category="Science",
            videomme_task_type="QA",
        ),
        VideoMMESampleRequest(
            prompt="q",
            prompt_len=1,
            expected_output_len=8,
            videomme_gold_answer="B",
            videomme_video_id="v2",
            videomme_question_id="q2",
            videomme_duration="long",
            videomme_domain="Knowledge",
            videomme_sub_category="Science",
            videomme_task_type="QA",
        ),
    ]
    metrics = compute_videomme_accuracy_metrics(reqs, [_Out("A"), _Out("The answer is C")])
    assert metrics is not None
    assert metrics["videomme_correct"] == 1
    assert metrics["videomme_evaluated_ok"] == 2
    assert metrics["videomme_accuracy"] == 0.5
    assert metrics["videomme_per_duration_accuracy"]["short"] == 1.0
    assert metrics["videomme_per_duration_accuracy"]["long"] == 0.0
