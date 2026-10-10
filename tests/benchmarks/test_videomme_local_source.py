# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline Video-MME sources: local mirrors must not fall back to a Hub id."""

from __future__ import annotations

import os
import zipfile
from argparse import Namespace
from pathlib import Path
from typing import Any
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

    from vllm_omni.transformers_utils import repo_utils

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
    assert _validate_videomme(metrics, min_accuracy=0.4) == []
    assert "videomme_eval_items" not in metrics
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


@pytest.mark.parametrize("max_frames", [0, -1])
def test_rejects_nonpositive_max_frames(max_frames: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(VideoMMEDataset, "load_data", lambda self: None)
    with pytest.raises(ValueError, match="max_frames must be positive"):
        VideoMMEDataset(parquet_path="unused.parquet", max_frames=max_frames)


def test_frame_cache_rebuild_overwrites_incomplete_files(tmp_path: Path) -> None:
    frame = tmp_path / "frame_0000.jpg"
    frame.write_bytes(b"interrupted JPEG")
    VideoMMEDataset._emit_media(tmp_path, frame.name, b"complete JPEG", "image_url", "image/jpeg")
    assert frame.read_bytes() == b"complete JPEG"


def test_frame_cache_rejects_empty_media(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text('{"count": 1}', encoding="utf-8")
    (tmp_path / "frame_0000.jpg").touch()
    dataset = object.__new__(VideoMMEDataset)
    assert dataset._cached_parts_from_disk(tmp_path, include_audio=False) is None


@pytest.mark.parametrize("case", ["duplicates", "missing_gold"])
def test_accuracy_gate_rejects_incomplete_question_coverage(case: str) -> None:
    def request(question_id: str, gold: str) -> VideoMMESampleRequest:
        return VideoMMESampleRequest(
            prompt="q",
            prompt_len=1,
            expected_output_len=8,
            videomme_question_id=question_id,
            videomme_gold_answer=gold,
        )

    requests = [
        request("q1", "A"),
        request("q1" if case == "duplicates" else "q2", "" if case == "missing_gold" else "A"),
    ]
    metrics = compute_videomme_accuracy_metrics(requests, [_Out("A"), _Out("A")])
    assert metrics is not None
    assert metrics["videomme_accuracy"] == 1.0
    errors = _validate_videomme(metrics, min_accuracy=0.68)
    assert any(("unique" if case == "duplicates" else "no_gold") in error for error in errors)


@pytest.fixture
def videomme_mirror(tmp_path: Path) -> Path:
    import pandas as pd

    parquet = tmp_path / "videomme" / "test-00000-of-00001.parquet"
    parquet.parent.mkdir()
    pd.DataFrame(
        [
            {
                "videoID": f"v{i}",
                "question_id": f"q{i}",
                "question": "Which option is correct?",
                "options": ["A. first", "B. second", "C. third", "D. fourth"],
                "answer": "A",
                "duration": duration,
            }
            for i, duration in enumerate(("short", "long"), start=1)
        ]
    ).to_parquet(parquet)
    videos = tmp_path / "video"
    videos.mkdir()
    for i in (1, 2):
        (videos / f"v{i}.mp4").write_bytes(b"video")
    return tmp_path


def _sample_video_urls(dataset: VideoMMEDataset, monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> list:
    monkeypatch.setattr(
        "vllm_omni.benchmarks.data_modules.videomme_dataset.get_cached_tokenizer", lambda tokenizer: tokenizer
    )
    return dataset.sample(Namespace(encode=lambda text: [1]), **kwargs)


@pytest.mark.parametrize("no_oversample", [True, False])
def test_missing_video_cannot_pass_accuracy_gate(
    videomme_mirror: Path, monkeypatch: pytest.MonkeyPatch, no_oversample: bool
) -> None:
    (videomme_mirror / "video" / "v2.mp4").unlink()
    dataset = VideoMMEDataset(dataset_path=str(videomme_mirror), pack_mode="video_url", disable_shuffle=True)
    requests = _sample_video_urls(dataset, monkeypatch, num_requests=2, no_oversample=no_oversample)
    metrics = compute_videomme_accuracy_metrics(requests, [_Out("A") for _ in requests])
    assert metrics is not None
    assert metrics["videomme_accuracy"] == 1.0
    assert metrics["videomme_submitted"] == (1 if no_oversample else 2)
    assert metrics["videomme_skipped_rows"] == 1
    assert any("videomme_skipped_rows" in error for error in _validate_videomme(metrics, min_accuracy=0.68))


def test_complete_duration_subset_can_pass_accuracy_gate(
    videomme_mirror: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = VideoMMEDataset(dataset_path=str(videomme_mirror), pack_mode="video_url", duration_filter="short")
    requests = _sample_video_urls(dataset, monkeypatch, num_requests=2700, no_oversample=True)
    metrics = compute_videomme_accuracy_metrics(requests, [_Out("A") for _ in requests])
    assert metrics is not None
    assert metrics["videomme_submitted"] == 1
    assert metrics["videomme_skipped_rows"] == 0
    assert _validate_videomme(metrics, min_accuracy=0.68) == []


def test_dataset_direct_hub_loading_resolves_video_assets(
    videomme_mirror: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloaded = []

    def download(repo_id: str) -> Path:
        downloaded.append(repo_id)
        return videomme_mirror

    monkeypatch.setattr("vllm_omni.benchmarks.data_modules.videomme_dataset.ensure_videomme_hub_root", download)
    dataset = VideoMMEDataset(dataset_path="my-org/custom-videomme", pack_mode="video_url")
    assert downloaded == ["my-org/custom-videomme"]
    assert dataset.video_dir == videomme_mirror / "video"
    assert len(dataset.data) == 2


def test_local_source_with_explicit_parquet_keeps_video_root(videomme_mirror: Path) -> None:
    dataset = VideoMMEDataset(
        dataset_path=str(videomme_mirror),
        parquet_path=str(videomme_local_parquet(videomme_mirror)),
        pack_mode="video_url",
    )
    assert dataset.video_dir == videomme_mirror / "video"
    assert len(dataset.data) == 2


def test_local_source_without_parquet_fails_locally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_hub(*args, **kwargs):
        pytest.fail("An explicitly selected local mirror must not fall back to Hub loading")

    monkeypatch.setattr("vllm_omni.benchmarks.data_modules.videomme_dataset.load_dataset", unexpected_hub)
    with pytest.raises(FileNotFoundError, match="No Video-MME parquet"):
        VideoMMEDataset(dataset_path=str(tmp_path))


def test_get_samples_prefers_local_root_over_hf_name(videomme_mirror: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = Namespace(
        dataset_name="videomme",
        dataset_path=str(videomme_mirror),
        hf_name="unused-hf-name",
        backend="openai-chat-omni",
        seed=0,
        videomme_pack_mode="video_url",
        num_prompts=2,
        request_id_prefix="",
        no_oversample=True,
    )
    monkeypatch.setattr(
        "vllm_omni.benchmarks.data_modules.videomme_dataset.get_cached_tokenizer", lambda tokenizer: tokenizer
    )
    requests = get_samples(args, Namespace(encode=lambda text: [1]))
    assert len(requests) == 2
    assert {request.videomme_question_id for request in requests} == {"q1", "q2"}


def test_extraction_replaces_truncated_video(tmp_path: Path) -> None:
    video = tmp_path / "video" / "v1.mp4"
    video.parent.mkdir()
    video.write_bytes(b"partial")
    with zipfile.ZipFile(tmp_path / "videos_chunked_01.zip", "w") as archive:
        archive.writestr("nested/v1.mp4", b"complete video")
    assert ensure_videomme_videos_extracted(tmp_path) == video.parent
    assert video.read_bytes() == b"complete video"


def test_failed_extraction_does_not_publish_partial_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with zipfile.ZipFile(tmp_path / "videos_chunked_01.zip", "w") as archive:
        archive.writestr("v1.mp4", b"complete video")

    def interrupted_copy(src, dst):
        dst.write(b"partial")
        raise OSError("interrupted")

    monkeypatch.setattr("vllm_omni.benchmarks.data_modules.videomme_dataset.shutil.copyfileobj", interrupted_copy)
    with pytest.raises(OSError, match="interrupted"):
        ensure_videomme_videos_extracted(tmp_path)
    assert list((tmp_path / "video").iterdir()) == []
    assert not (tmp_path / ".videomme_videos_extracted").exists()
