# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline Video-MME sources: local mirrors must not fall back to a Hub id."""

from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest
from PIL import Image

from tests.e2e.accuracy.qwen3_omni.run_qwen_omni_acc_benchmark import _validate_videomme
from vllm_omni.benchmarks.data_modules.videomme_dataset import (
    VideoMMEDataset,
    VideoMMESampleRequest,
    cached_videomme_snapshots,
    ensure_videomme_hub_root,
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
from vllm_omni.benchmarks.patch.patch import _looks_like_hf_dataset_id, _videomme_repo_from_args, get_samples

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


def _stub_failing_hub(monkeypatch: pytest.MonkeyPatch, hub_cache: Path) -> dict[str, object]:
    """Point the Hub cache at ``hub_cache`` and make every snapshot_download miss."""
    from huggingface_hub import constants as hf_constants
    from huggingface_hub.errors import LocalEntryNotFoundError
    from vllm.transformers_utils import repo_utils

    seen: dict[str, object] = {}

    class _Api:
        def snapshot_download(self, **kwargs: object) -> str:
            seen.update(kwargs)
            raise LocalEntryNotFoundError("no cached snapshot")

    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(hub_cache))
    monkeypatch.setattr(repo_utils, "hf_api", lambda: _Api())
    return seen


def test_hub_root_offline_without_cache_points_at_local_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _stub_failing_hub(monkeypatch, tmp_path)

    with pytest.raises(FileNotFoundError, match="--videomme-video-dir") as excinfo:
        ensure_videomme_hub_root("lmms-lab/Video-MME")

    # Offline runs must consult the cache directly instead of resolving over the network.
    assert seen["local_files_only"] is True
    assert "already cached under another repo id" not in str(excinfo.value)


def test_hub_root_offline_names_snapshot_cached_under_another_repo_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sibling = tmp_path / "datasets--lmms-eval--Video-MME" / "snapshots" / "deadbeef"
    sibling.mkdir(parents=True)
    _write_parquet_placeholder(sibling)
    _stub_failing_hub(monkeypatch, tmp_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        ensure_videomme_hub_root("lmms-lab/Video-MME")

    assert str(sibling.resolve()) in str(excinfo.value)


def test_cached_videomme_snapshots_skips_partial_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from huggingface_hub import constants as hf_constants

    (tmp_path / "datasets--lmms-eval--Video-MME" / "snapshots" / "deadbeef").mkdir(parents=True)
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))

    assert cached_videomme_snapshots() == []


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
    assert metrics["videomme_submitted"] == 2
    assert metrics["videomme_unique_question_ids"] == 2
    assert metrics["videomme_per_duration_accuracy"]["short"] == 1.0
    assert metrics["videomme_per_duration_accuracy"]["long"] == 0.0


def test_compute_videomme_accuracy_metrics_http_fail_excluded_from_default() -> None:
    reqs = [
        VideoMMESampleRequest(
            prompt="q",
            prompt_len=1,
            expected_output_len=8,
            videomme_gold_answer="A",
            videomme_question_id="q1",
        ),
        VideoMMESampleRequest(
            prompt="q",
            prompt_len=1,
            expected_output_len=8,
            videomme_gold_answer="B",
            videomme_question_id="q2",
        ),
    ]
    metrics = compute_videomme_accuracy_metrics(reqs, [_Out("A"), _Out("", success=False)])
    assert metrics is not None
    assert metrics["videomme_accuracy"] == 1.0
    assert metrics["videomme_accuracy_incl_http_fail"] == 0.5
    assert metrics["videomme_request_failed"] == 1
    errs = _validate_videomme(metrics, min_accuracy=0.68)
    assert any("videomme_request_failed" in err for err in errs)


def test_compute_videomme_accuracy_metrics_saves_eval_items() -> None:
    req = VideoMMESampleRequest(
        prompt="q",
        prompt_len=1,
        expected_output_len=8,
        videomme_gold_answer="A",
        videomme_question_id="q1",
    )
    metrics = compute_videomme_accuracy_metrics([req], [_Out("A")], include_per_item=True)
    assert metrics is not None
    assert metrics["videomme_eval_items"][0]["question_id"] == "q1"


def test_videomme_save_eval_items_cli_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.benchmarks import serve

    seen: dict[str, str | None] = {}

    async def fake_main_async(args: Namespace) -> dict[str, str]:
        seen["env"] = os.environ.get("VIDEOMME_SAVE_EVAL_ITEMS")
        return {"ok": "1"}

    monkeypatch.delenv("VIDEOMME_SAVE_EVAL_ITEMS", raising=False)
    monkeypatch.setattr(serve, "main_async", fake_main_async)

    args = Namespace(
        videomme_save_eval_items=True,
        seed_tts_wer_eval=False,
        seed_tts_wer_save_items=False,
        daily_omni_save_eval_items=False,
        omni_request_timeout_s=None,
        endpoint=None,
        backend="openai-chat-omni",
        explicit_keys=frozenset(),
        extra_body=None,
        print_stage=False,
        dataset_name="videomme",
    )
    assert serve.main(args) == {"ok": "1"}
    assert seen["env"] == "1"


def _file_uri_path(url: str) -> Path:
    parsed = urlparse(url)
    assert parsed.scheme == "file"
    return Path(unquote(parsed.path))


def test_relative_video_dir_emits_absolute_file_uris_cold_and_warm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "vid1.mp4").write_bytes(b"fake")

    def fake_extract(cls, video_path, *, include_audio, max_num_frames):  # noqa: ANN001
        return [Image.new("RGB", (4, 4), color="red")], []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(VideoMMEDataset, "load_data", lambda self: None)
    monkeypatch.setattr(VideoMMEDataset, "_extract_frames_and_audio", classmethod(fake_extract))

    ds = VideoMMEDataset(parquet_path="dummy.parquet", video_dir="videos", max_frames=1)
    assert ds.video_dir == videos.resolve()

    cold = ds._get_minicpm_frame_parts("vid1", include_audio=False)
    assert cold is not None
    cold_path = _file_uri_path(cold[0]["image_url"]["url"])
    assert cold_path.is_absolute()
    assert cold_path.is_file()

    warm_ds = VideoMMEDataset(parquet_path="dummy.parquet", video_dir="videos", max_frames=1)
    warm = warm_ds._get_minicpm_frame_parts("vid1", include_audio=False)
    assert warm is not None
    warm_path = _file_uri_path(warm[0]["image_url"]["url"])
    assert warm_path.is_absolute()
    assert warm_path == cold_path


def test_videomme_repo_from_args_honors_custom_hub_id() -> None:
    args = Namespace(dataset_name="videomme", dataset_path="my-org/custom-videomme", hf_name=None)
    assert _videomme_repo_from_args(args, explicit=True) == "my-org/custom-videomme"


def test_videomme_repo_from_args_defaults_when_path_omitted() -> None:
    args = Namespace(dataset_name="videomme", dataset_path=None, hf_name=None)
    assert _videomme_repo_from_args(args, explicit=True) is None


def test_videomme_repo_from_args_rejects_unsupported_path() -> None:
    args = Namespace(dataset_name="videomme", dataset_path="not-a-local-dir-or-hub-id", hf_name=None)
    with pytest.raises(ValueError, match="Unsupported Video-MME"):
        _videomme_repo_from_args(args, explicit=True)


def test_get_samples_rejects_unsupported_videomme_dataset_path() -> None:
    args = Namespace(
        dataset_name="videomme",
        dataset_path="not-a-local-dir-or-hub-id",
        hf_name=None,
        backend="openai-chat-omni",
        seed=0,
    )
    with pytest.raises(ValueError, match="Unsupported Video-MME"):
        get_samples(args, None)


def test_hf_dataset_name_does_not_auto_detect_custom_repo() -> None:
    args = Namespace(dataset_name="hf", dataset_path="my-org/custom-videomme", hf_name=None)
    assert _videomme_repo_from_args(args) is None
    assert _looks_like_hf_dataset_id("my-org/custom-videomme")
    assert not _looks_like_hf_dataset_id("./videos")
    official = Namespace(dataset_name="hf", dataset_path="lmms-eval/Video-MME", hf_name=None)
    assert _videomme_repo_from_args(official) == "lmms-eval/Video-MME"
