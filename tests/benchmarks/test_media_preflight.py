# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Media dependency failures must precede dataset loading and model startup."""

import builtins

import pytest

from vllm_omni.benchmarks.media_preflight import check_minicpm_media_dependencies, main

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("failure", [ModuleNotFoundError("No module named av"), OSError("libavcodec.so missing")])
def test_dependency_error_explains_installation(monkeypatch, failure):
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "av":
            raise failure
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(RuntimeError, match="pip install -e") as exc:
        check_minicpm_media_dependencies(include_audio=True)
    assert exc.value.__cause__ is failure


def test_visual_probe_does_not_require_audio_backend(monkeypatch, tmp_path):
    from vllm_omni.benchmarks.data_modules.daily_omni_dataset import DailyOmniDataset
    from vllm_omni.benchmarks.media_preflight import _write_fixture

    video, _ = _write_fixture(tmp_path)
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name in ("soundfile", "vllm.multimodal.media.audio"):
            raise ModuleNotFoundError(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    check_minicpm_media_dependencies(include_audio=False)
    frames, segments = DailyOmniDataset._extract_minicpm_frame_audio_segments(
        video, audio_path=None, include_audio=False
    )
    assert len(frames) == 2 and segments == []
    with pytest.raises(RuntimeError):
        check_minicpm_media_dependencies(include_audio=True)


def test_missing_dependency_fails_before_dataset_io(monkeypatch):
    from vllm_omni.benchmarks.data_modules.daily_omni_dataset import DailyOmniDataset

    def fail(**kwargs):
        raise RuntimeError("environment failure before any sample")

    monkeypatch.setattr("vllm_omni.benchmarks.media_preflight.check_minicpm_media_dependencies", fail)
    with pytest.raises(RuntimeError, match="environment failure"):
        DailyOmniDataset(qa_json_path="does-not-exist.json", pack_mode="minicpm-interleave")


def test_offline_media_preflight():
    assert main([]) == 0


def test_invalid_video_fails_preflight(tmp_path):
    import av

    video = tmp_path / "broken.mp4"
    video.write_bytes(b"not a video")
    with pytest.raises(av.error.InvalidDataError):
        main(["--video", str(video)])
