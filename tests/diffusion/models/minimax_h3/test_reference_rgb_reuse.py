# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.models.minimax_h3 import encoder_processing, reference_video

pytestmark = [pytest.mark.cpu, pytest.mark.core_model, pytest.mark.diffusion]


@dataclass
class _Sampling:
    extra_args: dict[str, object]


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


def test_reference_rgb_reuse_request_default_is_disabled():
    resolve = reference_video.resolve_reference_video_decode_reuse
    key = reference_video.MINIMAX_H3_REFERENCE_VIDEO_DECODE_REUSE_KEY
    assert resolve({}) is False
    assert resolve({key: False}) is False
    assert resolve({key: True}) is True


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


@pytest.mark.parametrize("has_sidecar", [False, True])
@pytest.mark.parametrize("request_override", [None, False, True])
def test_reused_descriptor_selects_frames_independently_of_request_option(
    monkeypatch, tmp_path, has_sidecar, request_override
):
    """Old descriptors remain usable; an existing sidecar is always consumed."""
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
    extra_args: dict[str, object] = {"task": "ref2va"}
    if request_override is not None:
        extra_args[reference_video.MINIMAX_H3_REFERENCE_VIDEO_DECODE_REUSE_KEY] = request_override
    result = encoder_processing.prepare_encoder_inputs(
        {"prompt": "reference clip", "multi_modal_data": {"video": "source.mp4"}},
        _Sampling(extra_args),
        prepared_reference_videos=[item],
    )
    expected = rgb if has_sidecar else decoded
    np.testing.assert_array_equal(result.media.videos[0].numpy(), expected)
    np.testing.assert_array_equal(result.qwen_videos[0][0], expected[:1])
    assert result.video_timestamps == [[0.0]]


def test_reused_descriptor_does_not_fall_back_from_a_truncated_sidecar(monkeypatch, tmp_path):
    item = _descriptor_item(tmp_path)
    item["prepared_path"] = str(tmp_path / "prepared.mp4")
    Path(item["prepared_path"]).touch()
    Path(item["rgb_path"]).write_bytes(b"truncated")
    monkeypatch.setattr(encoder_processing, "load_video_frames", lambda path: np.zeros((2, 2, 2, 3), dtype=np.uint8))
    monkeypatch.setattr(encoder_processing, "resolve_minimax_h3_shape", lambda *args: (32, 32, 96, 27, 160))
    with pytest.raises(OmniClientError, match="expected 24"):
        encoder_processing.prepare_encoder_inputs(
            {"prompt": "reference clip", "multi_modal_data": {"video": "source.mp4"}},
            _Sampling({"task": "ref2va"}),
            prepared_reference_videos=[item],
        )


@pytest.fixture(scope="module")
def ffmpeg_rgb_encoder():
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required")
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    if "libx264rgb" not in result.stdout:
        pytest.skip("ffmpeg with the libx264rgb encoder is required")


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", "1", [], {}])
def test_request_reuse_option_rejects_non_booleans(value):
    with pytest.raises(OmniClientError, match="reference_video_decode_reuse must be a boolean"):
        encoder_processing.prepare_encoder_inputs(
            {"prompt": "reference clip", "multi_modal_data": {"video": "not-decoded.mp4"}},
            _Sampling({"task": "ref2va", reference_video.MINIMAX_H3_REFERENCE_VIDEO_DECODE_REUSE_KEY: value}),
        )


def test_request_reuse_option_requires_ref2va():
    with pytest.raises(OmniClientError, match="requires the ref2va task"):
        encoder_processing.prepare_encoder_inputs(
            "text-only scene",
            _Sampling({"task": "t2va", reference_video.MINIMAX_H3_REFERENCE_VIDEO_DECODE_REUSE_KEY: True}),
        )


@pytest.fixture(scope="module")
def reference_clip(tmp_path_factory, ffmpeg_rgb_encoder):
    source = tmp_path_factory.mktemp("reference-rgb-option") / "source.mp4"
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
            "testsrc=size=256x256:rate=24",
            "-frames:v",
            "52",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            str(source),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    return source


@pytest.mark.parametrize(
    ("request_override", "expected_reuse"),
    [
        (None, False),
        (False, False),
        (True, True),
    ],
)
def test_request_option_reaches_real_rgb_producer(monkeypatch, reference_clip, request_override, expected_reuse):
    """Exercise request resolution, real ffmpeg output, and both live consumers."""
    monkeypatch.setattr(encoder_processing, "resolve_minimax_h3_shape", lambda *args: (256, 256, 52, 13, 160))
    original_transcode = reference_video._transcode_reference_video
    captures = []

    def record_transcode(*args, **kwargs):
        captures.append(kwargs["rgb_output_path"] is not None)
        return original_transcode(*args, **kwargs)

    monkeypatch.setattr(reference_video, "_transcode_reference_video", record_transcode)
    extra_args: dict[str, object] = {"task": "ref2va"}
    if request_override is not None:
        extra_args[reference_video.MINIMAX_H3_REFERENCE_VIDEO_DECODE_REUSE_KEY] = request_override
    result = encoder_processing.prepare_encoder_inputs(
        {"prompt": "reference clip", "multi_modal_data": {"video": str(reference_clip)}}, _Sampling(extra_args)
    )
    assert captures == [expected_reuse]
    expected = reference_video._decode_video_frames_ffmpeg(str(reference_clip), frame_count=52, width=256, height=256)
    np.testing.assert_array_equal(result.media.videos[0].numpy(), expected)
    np.testing.assert_array_equal(result.qwen_videos[0][0], expected[[0, 12, 24, 36, 48]])
    assert result.video_timestamps == [[0.25, 1.25, 2.0]]


def test_removed_environment_switch_does_not_override_consecutive_requests(monkeypatch, reference_clip):
    monkeypatch.setenv("VLLM_OMNI_H3_REF_VIDEO_DECODE_REUSE", "1")
    monkeypatch.setattr(encoder_processing, "resolve_minimax_h3_shape", lambda *args: (256, 256, 52, 13, 160))
    original_transcode = reference_video._transcode_reference_video
    captures = []

    def record_transcode(*args, **kwargs):
        captures.append(kwargs["rgb_output_path"] is not None)
        return original_transcode(*args, **kwargs)

    monkeypatch.setattr(reference_video, "_transcode_reference_video", record_transcode)
    outputs = []
    for override in (False, True, None):
        extra_args: dict[str, object] = {"task": "ref2va"}
        if override is not None:
            extra_args[reference_video.MINIMAX_H3_REFERENCE_VIDEO_DECODE_REUSE_KEY] = override
        result = encoder_processing.prepare_encoder_inputs(
            {"prompt": "reference clip", "multi_modal_data": {"video": str(reference_clip)}}, _Sampling(extra_args)
        )
        outputs.append(result.media.videos[0].numpy())
        assert os.environ["VLLM_OMNI_H3_REF_VIDEO_DECODE_REUSE"] == "1"
    assert captures == [False, True, False]
    np.testing.assert_array_equal(outputs[0], outputs[1])
    np.testing.assert_array_equal(outputs[0], outputs[2])


@pytest.mark.parametrize(
    ("target_frames", "duration", "start", "pad", "expected_count"),
    [(6, None, 0.0, False, 6), (0, 0.25, 0.125, False, 6), (20, None, 0.0, True, 20), (0, None, 0.0, False, 12)],
)
def test_transcode_sidecar_matches_prepared_video_pixels(
    tmp_path, ffmpeg_rgb_encoder, target_frames, duration, start, pad, expected_count
):
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
