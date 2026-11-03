# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
import torch

from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.minimax_h3 import encoder_processing as processing
from vllm_omni.model_executor.models.minimax_h3 import reference_video

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def reference(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for real reference-audio tests")
    path = tmp_path / "reference.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:r=24",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100",
            "-t",
            "3",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
        timeout=30,
    )
    return {
        "original_path": str(path),
        "prepared_path": str(path),
        "input_has_audio": True,
        "audio_duration_seconds": 2.5,
    }


def _prepare(references):
    return processing.prepare_encoder_inputs(
        {"prompt": "reference", "multi_modal_data": {"video": [item["original_path"] for item in references]}},
        OmniDiffusionSamplingParams(height=64, width=64, extra_args={"task": "ref2va", "duration": 4.4}),
        prepared_reference_videos=references,
    )


def test_audio_overlap_preserves_reference_order_and_waveforms(reference):
    silent = {**reference, "input_has_audio": False}
    prepared = _prepare([reference, silent, reference])
    expected, rate = reference_video.load_video_audio(reference["original_path"], duration_seconds=2.5)
    assert prepared.media.video_audios[1] is None
    for index in (0, 2):
        waveform, actual_rate = prepared.media.video_audios[index]
        assert actual_rate == rate
        torch.testing.assert_close(waveform, expected, rtol=0, atol=0)
    assert prepared.condition_labels == [("audio", 1), ("video", 1), ("video", 2), ("audio", 2), ("video", 3)]


def test_frame_failure_cancels_queued_audio_and_joins_worker(reference, monkeypatch):
    # Hold the real executor's worker so both audio jobs remain queued.
    pool = ThreadPoolExecutor(max_workers=1)
    release = Event()
    started = Event()
    futures = []
    submit = pool.submit
    shutdown = pool.shutdown

    def hold_worker():
        started.set()
        assert release.wait(10)

    blocker = submit(hold_worker)
    assert started.wait(10)

    def record_submit(*args, **kwargs):
        future = submit(*args, **kwargs)
        futures.append(future)
        return future

    def release_and_shutdown(*, wait=True, cancel_futures=False):
        shutdown(wait=False, cancel_futures=cancel_futures)
        release.set()
        shutdown(wait=wait)

    def fail_frames(path):
        raise RuntimeError("frame decoding failed")

    monkeypatch.setattr(pool, "submit", record_submit)
    monkeypatch.setattr(pool, "shutdown", release_and_shutdown)
    monkeypatch.setattr(processing, "ThreadPoolExecutor", lambda **kwargs: pool)
    monkeypatch.setattr(processing, "load_video_frames", fail_frames)
    try:
        with pytest.raises(RuntimeError, match="frame decoding failed"):
            _prepare([reference, reference])
        assert len(futures) == 2
        assert all(future.cancelled() for future in futures)
        assert blocker.done()
    finally:
        release.set()
        shutdown(wait=True, cancel_futures=True)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires a POSIX FIFO")
def test_audio_extraction_timeout_propagates_and_cleans_tempdir(tmp_path, monkeypatch):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for the subprocess timeout test")
    fifo = tmp_path / "blocked_audio"
    getattr(os, "mkfifo")(fifo)
    monkeypatch.setattr(reference_video, "_AUDIO_EXTRACTION_TIMEOUT_SECONDS", 0.2)
    with pytest.raises(subprocess.TimeoutExpired) as error:
        reference_video.load_video_audio(str(fifo))
    assert error.value.timeout == pytest.approx(0.2, abs=0.02)
    assert not os.path.exists(os.path.dirname(error.value.cmd[-1]))
