"""Tests for Daily-Omni ``--daily-omni-pack-mode`` packing recipes.

The ``minicpm-interleave`` mode is an accuracy-critical protocol: MiniCPM-o only reaches
OpenBMB's reported Daily-Omni score when frames and 1s audio segments arrive strictly
interleaved, with ``max_slice_nums=1`` / ``use_image_id=False`` and an empty MiniCPM
system message (not the Qwen system prompt). These tests pin that contract so a refactor
cannot silently regress the score.

vllm stubs are installed by tests/benchmarks/conftest.py before collection.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Load the data module directly (bypasses vllm_omni.__init__ heavy imports).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "vllm_omni" / "benchmarks" / "data_modules" / "daily_omni_dataset.py"
_MODULE_NAME = "vllm_omni.benchmarks.data_modules.daily_omni_dataset"

if _MODULE_NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_MODULE_NAME] = _mod
    _spec.loader.exec_module(_mod)

from vllm_omni.benchmarks.data_modules.daily_omni_dataset import (  # noqa: E402
    DailyOmniDataset,
    _uniform_sample_indices,
)

_VIDEO_ID = "vid001"
_NUM_FRAMES = 5


@pytest.fixture
def video_dir(tmp_path: Path) -> Path:
    """Official Daily-Omni ``Videos/`` layout with placeholder media files."""
    per_video = tmp_path / "Videos" / _VIDEO_ID
    per_video.mkdir(parents=True)
    (per_video / f"{_VIDEO_ID}_video.mp4").write_bytes(b"fake-mp4")
    (per_video / f"{_VIDEO_ID}_audio.wav").write_bytes(b"fake-wav")
    return tmp_path / "Videos"


@pytest.fixture
def qa_json(tmp_path: Path) -> Path:
    path = tmp_path / "qa.json"
    path.write_text(
        json.dumps(
            [
                {
                    "video_id": _VIDEO_ID,
                    "question": "What happens first?",
                    "choice": {"A": "a", "B": "b", "C": "c", "D": "d"},
                    "answer": "A",
                    "task_type": "AV Event Alignment",
                    "video_duration": "30s",
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


def _make_dataset(qa_json: Path, video_dir: Path, **kwargs) -> DailyOmniDataset:
    return DailyOmniDataset(
        qa_json_path=str(qa_json),
        video_dir=str(video_dir),
        disable_shuffle=True,
        **kwargs,
    )


def _stub_extract(monkeypatch, dataset: DailyOmniDataset, counter: list[int]) -> None:
    """Replace decord/ffmpeg extraction with deterministic in-memory frames + segments."""
    from PIL import Image

    def fake_extract(video_path, *, audio_path, include_audio, **_kw):
        counter.append(1)
        frames = [Image.new("RGB", (16, 16), color=(i, i, i)) for i in range(_NUM_FRAMES)]
        segments = [np.zeros(1600, dtype=np.float32) for _ in range(_NUM_FRAMES)] if include_audio else []
        return frames, segments

    monkeypatch.setattr(dataset, "_extract_minicpm_frame_audio_segments", fake_extract)


# ---------------------------------------------------------------------------
# _uniform_sample_indices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "k"), [(10, 4), (100, 64), (1000, 64), (7, 7), (3, 5), (1, 1)])
def test_uniform_sample_indices_matches_openbmb_midpoint(n: int, k: int) -> None:
    """Pin OpenBMB midpoint-bin sampling (not np.linspace endpoints)."""
    got = _uniform_sample_indices(n, k)
    if n <= k:
        assert got == list(range(n))
    else:
        gap = n / k
        expected = [int(i * gap + gap / 2) for i in range(k)]
        assert got == expected
        # Endpoints differ from linspace for typical Daily-Omni downsample sizes.
        linspace = [int(i) for i in np.linspace(0, n - 1, k, dtype=int)]
        if n == 1000 and k == 64:
            assert got[0] == 7 and got[-1] == 992
            assert linspace[0] == 0 and linspace[-1] == 999
            assert got != linspace


def test_uniform_sample_indices_non_positive_k() -> None:
    assert _uniform_sample_indices(10, 0) == []
    assert _uniform_sample_indices(10, -1) == []


# ---------------------------------------------------------------------------
# minicpm-interleave packing
# ---------------------------------------------------------------------------


def test_minicpm_interleave_alternates_image_and_audio(monkeypatch, qa_json, video_dir) -> None:
    ds = _make_dataset(qa_json, video_dir, pack_mode="minicpm-interleave", input_mode="all")
    _stub_extract(monkeypatch, ds, [])

    parts, extra, position = ds._compose_daily_omni_multimodal(_VIDEO_ID, None)

    assert position == "first"
    assert [p["type"] for p in parts] == ["image_url", "audio_url"] * _NUM_FRAMES
    assert extra["mm_processor_kwargs"] == {
        "use_audio_in_video": False,
        "max_slice_nums": 1,
        "use_image_id": False,
    }


def test_minicpm_interleave_visual_mode_emits_frames_only(monkeypatch, qa_json, video_dir) -> None:
    ds = _make_dataset(qa_json, video_dir, pack_mode="minicpm-interleave", input_mode="visual")
    _stub_extract(monkeypatch, ds, [])

    parts, _extra, _position = ds._compose_daily_omni_multimodal(_VIDEO_ID, None)

    assert [p["type"] for p in parts] == ["image_url"] * _NUM_FRAMES


def test_minicpm_interleave_writes_file_urls_under_video_dir(monkeypatch, qa_json, video_dir) -> None:
    ds = _make_dataset(qa_json, video_dir, pack_mode="minicpm-interleave", input_mode="all")
    _stub_extract(monkeypatch, ds, [])

    parts, _extra, _position = ds._compose_daily_omni_multimodal(_VIDEO_ID, None)

    segment_dir = video_dir / ".minicpm_daily_omni_interleave" / _VIDEO_ID
    for part in parts:
        url = part[part["type"]]["url"]
        assert url.startswith("file://")
        assert Path(url[len("file://") :]).parent == segment_dir.resolve()
    assert sorted(p.name for p in segment_dir.glob("frame_*.jpg")) == [f"frame_{i:04d}.jpg" for i in range(_NUM_FRAMES)]


def test_minicpm_interleave_inline_mode_emits_data_urls(monkeypatch, qa_json, video_dir) -> None:
    ds = _make_dataset(
        qa_json,
        video_dir,
        pack_mode="minicpm-interleave",
        input_mode="all",
        inline_local_video=True,
    )
    _stub_extract(monkeypatch, ds, [])

    parts, _extra, _position = ds._compose_daily_omni_multimodal(_VIDEO_ID, None)

    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert parts[1]["audio_url"]["url"].startswith("data:audio/wav;base64,")
    assert not (video_dir / ".minicpm_daily_omni_interleave").exists()


def test_minicpm_interleave_caches_file_parts_but_not_inline_payloads(monkeypatch, qa_json, video_dir) -> None:
    """Inline base64 must not be retained: ~700 Daily-Omni videos would pin GBs of RSS."""
    file_calls: list[int] = []
    ds_file = _make_dataset(qa_json, video_dir, pack_mode="minicpm-interleave", input_mode="all")
    _stub_extract(monkeypatch, ds_file, file_calls)
    ds_file._compose_daily_omni_multimodal(_VIDEO_ID, None)
    ds_file._compose_daily_omni_multimodal(_VIDEO_ID, None)
    assert len(file_calls) == 1
    assert ds_file._minicpm_interleave_cache

    inline_calls: list[int] = []
    ds_inline = _make_dataset(
        qa_json,
        video_dir,
        pack_mode="minicpm-interleave",
        input_mode="all",
        inline_local_video=True,
    )
    _stub_extract(monkeypatch, ds_inline, inline_calls)
    ds_inline._compose_daily_omni_multimodal(_VIDEO_ID, None)
    ds_inline._compose_daily_omni_multimodal(_VIDEO_ID, None)
    assert len(inline_calls) == 2
    assert not ds_inline._minicpm_interleave_cache


def test_minicpm_interleave_missing_local_video_returns_none(qa_json, tmp_path) -> None:
    empty = tmp_path / "empty_videos"
    empty.mkdir()
    ds = _make_dataset(qa_json, empty, pack_mode="minicpm-interleave", input_mode="all")

    parts, extra, _position = ds._compose_daily_omni_multimodal(_VIDEO_ID, None)

    assert parts is None
    assert extra is None


# ---------------------------------------------------------------------------
# System prompt selection
# ---------------------------------------------------------------------------


def test_minicpm_interleave_keeps_empty_system_message(monkeypatch, qa_json, video_dir) -> None:
    """Official MiniCPM omni mode still sends role=system with empty content."""
    ds = _make_dataset(qa_json, video_dir, pack_mode="minicpm-interleave", input_mode="all")
    _stub_extract(monkeypatch, ds, [])
    parts, _extra, _position = ds._compose_daily_omni_multimodal(_VIDEO_ID, None)

    messages = ds._build_daily_omni_openai_messages(parts, "q?", {"A": "a"})

    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"][0]["text"] == ""


def test_qwen_pack_mode_keeps_system_message(qa_json, video_dir) -> None:
    ds = _make_dataset(qa_json, video_dir, pack_mode="qwen", input_mode="all")

    messages = ds._build_daily_omni_openai_messages([], "q?", {"A": "a"})

    assert [m["role"] for m in messages] == ["system", "user"]
    assert "Qwen" in messages[0]["content"][0]["text"]


def test_invalid_pack_mode_rejected(qa_json, video_dir) -> None:
    with pytest.raises(ValueError, match="pack_mode must be"):
        _make_dataset(qa_json, video_dir, pack_mode="bogus")
