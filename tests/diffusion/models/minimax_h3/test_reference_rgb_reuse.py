# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.models.minimax_h3 import encoder_processing, reference_video

pytestmark = [pytest.mark.cpu, pytest.mark.core_model, pytest.mark.diffusion]


def _descriptor_item(tmp_path):
    return {
        "original_path": "source.mp4",
        "prepared_path": "prepared.mp4",
        "input_has_audio": False,
        "width": 2,
        "height": 2,
        "start_time_seconds": 0.0,
        "duration_seconds": 2.0,
        "audio_duration_seconds": 2.0,
        "rgb_path": str(tmp_path / "prepared.rgb"),
        "frame_count": 2,
    }


def test_reference_rgb_reuse_gate_is_strict(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_H3_REF_VIDEO_DECODE_REUSE", raising=False)
    assert not reference_video.reference_video_transcode_rgb_reuse_enabled()
    monkeypatch.setenv("VLLM_OMNI_H3_REF_VIDEO_DECODE_REUSE", "1")
    assert reference_video.reference_video_transcode_rgb_reuse_enabled()
    monkeypatch.setenv("VLLM_OMNI_H3_REF_VIDEO_DECODE_REUSE", "true")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        reference_video.reference_video_transcode_rgb_reuse_enabled()


def test_prepared_reference_descriptor_round_trips_rgb_sidecar(tmp_path):
    item = _descriptor_item(tmp_path)
    encoded = reference_video.serialize_prepared_reference_videos([item], str(tmp_path))
    assert reference_video.deserialize_prepared_reference_videos(encoded) == (str(tmp_path), [item])

    payload = json.loads(encoded)
    del payload["videos"][0]["frame_count"]
    with pytest.raises(ValueError, match="invalid MiniMax H3"):
        reference_video.deserialize_prepared_reference_videos(json.dumps(payload))


def test_prepared_rgb_sidecar_validates_exact_size(tmp_path):
    item = _descriptor_item(tmp_path)
    expected = np.arange(24, dtype=np.uint8).reshape(2, 2, 2, 3)
    expected.tofile(item["rgb_path"])
    actual = reference_video.load_prepared_rgb_frames(item)
    np.testing.assert_array_equal(actual, expected)

    with open(item["rgb_path"], "ab") as output:
        output.write(b"x")
    with pytest.raises(OmniClientError, match="expected 24"):
        reference_video.load_prepared_rgb_frames(item)


@pytest.mark.parametrize("enabled", ["0", "1", "true"])
@pytest.mark.parametrize("has_sidecar", [False, True])
def test_reused_descriptor_selects_frames_independently_of_consumer_env(monkeypatch, tmp_path, enabled, has_sidecar):
    """Old descriptors remain usable; an existing sidecar is always consumed."""
    monkeypatch.setenv("VLLM_OMNI_H3_REF_VIDEO_DECODE_REUSE", enabled)
    item = _descriptor_item(tmp_path)
    item["prepared_path"] = str(tmp_path / "prepared.mp4")
    Path(item["prepared_path"]).touch()
    rgb = np.arange(24, dtype=np.uint8).reshape(2, 2, 2, 3)
    decoded = np.full_like(rgb, 123)
    if has_sidecar:
        rgb.tofile(item["rgb_path"])
    else:
        del item["rgb_path"], item["frame_count"]
    monkeypatch.setattr(encoder_processing, "load_video_frames", lambda path: decoded)
    monkeypatch.setattr(encoder_processing, "resolve_minimax_h3_shape", lambda *args: (32, 32, 96, 27, 160))
    result = encoder_processing.prepare_encoder_inputs(
        {"prompt": "reference clip", "multi_modal_data": {"video": "source.mp4"}},
        SimpleNamespace(extra_args={"task": "ref2va"}),
        prepared_reference_videos=[item],
    )
    expected = rgb if has_sidecar else decoded
    np.testing.assert_array_equal(result.media.videos[0].numpy(), expected)
    np.testing.assert_array_equal(result.qwen_videos[0][0], expected[:1])
    assert result.video_timestamps == [[0.0]]


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_reused_descriptor_does_not_fall_back_from_a_truncated_sidecar(monkeypatch, tmp_path, enabled):
    monkeypatch.setenv("VLLM_OMNI_H3_REF_VIDEO_DECODE_REUSE", enabled)
    item = _descriptor_item(tmp_path)
    item["prepared_path"] = str(tmp_path / "prepared.mp4")
    Path(item["prepared_path"]).touch()
    Path(item["rgb_path"]).write_bytes(b"truncated")
    monkeypatch.setattr(encoder_processing, "load_video_frames", lambda path: np.zeros((2, 2, 2, 3), dtype=np.uint8))
    monkeypatch.setattr(encoder_processing, "resolve_minimax_h3_shape", lambda *args: (32, 32, 96, 27, 160))
    with pytest.raises(OmniClientError, match="expected 24"):
        encoder_processing.prepare_encoder_inputs(
            {"prompt": "reference clip", "multi_modal_data": {"video": "source.mp4"}},
            SimpleNamespace(extra_args={"task": "ref2va"}),
            prepared_reference_videos=[item],
        )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
@pytest.mark.parametrize(
    ("target_frames", "duration", "start", "pad", "expected_count"),
    [(6, None, 0.0, False, 6), (0, 0.25, 0.125, False, 6), (20, None, 0.0, True, 20), (0, None, 0.0, False, 12)],
)
def test_transcode_sidecar_matches_prepared_video_pixels(tmp_path, target_frames, duration, start, pad, expected_count):
    """Catch divergence in RGB conversion, trimming, seeking or last-frame padding."""
    source = str(tmp_path / "source.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=32x32:rate=24",
            "-frames:v",
            "12",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            source,
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    rgb_path = str(tmp_path / "prepared.rgb")
    prepared = reference_video._transcode_reference_video(
        source,
        target_width=32,
        target_height=32,
        target_frame_count=target_frames,
        workdir=str(tmp_path),
        start_time_seconds=start,
        duration_seconds=duration,
        pad_last_frame=pad,
        rgb_output_path=rgb_path,
    )
    rgb = reference_video.load_prepared_rgb_frames(
        {"rgb_path": rgb_path, "frame_count": expected_count, "width": 32, "height": 32}
    )
    try:
        decoded = reference_video._decode_video_frames_ffmpeg(prepared, frame_count=expected_count, width=32, height=32)
        np.testing.assert_array_equal(rgb, decoded)
        sidecar_sample = reference_video.sample_reference_video_frames(prepared, decoded_frames=rgb)
        decoded_sample = reference_video.sample_reference_video_frames(prepared, decoded_frames=decoded)
        np.testing.assert_array_equal(sidecar_sample["frames"], decoded_sample["frames"])
        assert sidecar_sample["block_timestamps"] == decoded_sample["block_timestamps"]
        np.testing.assert_array_equal(
            encoder_processing._frames_to_tensor(rgb).numpy(), encoder_processing._frames_to_tensor(decoded).numpy()
        )
    finally:
        rgb._mmap.close()
