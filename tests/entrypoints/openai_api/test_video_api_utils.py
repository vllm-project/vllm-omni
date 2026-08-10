# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OpenAI-compatible video API encoding helpers."""

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.postprocess import rife_interpolator
from vllm_omni.entrypoints.openai import video_api_utils

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _install_fake_video_mux(monkeypatch, mux_calls):
    def _fake_mux_video_audio_bytes(frames, audio, fps, audio_sample_rate, video_codec_options=None):
        mux_calls.append(
            {
                "frames": frames,
                "audio": audio,
                "fps": fps,
                "audio_sample_rate": audio_sample_rate,
                "video_codec_options": video_codec_options,
            }
        )
        return b"fake-video"

    monkeypatch.setattr(
        "vllm_omni.diffusion.utils.media_utils.mux_video_audio_bytes",
        _fake_mux_video_audio_bytes,
    )


def test_encode_video_bytes_exports_frames_without_interpolation(monkeypatch):
    mux_calls = []
    _install_fake_video_mux(monkeypatch, mux_calls)

    frames = [np.full((2, 2, 3), fill_value=i / 5, dtype=np.float32) for i in range(5)]
    video_bytes = video_api_utils._encode_video_bytes(
        frames,
        fps=8,
    )

    assert video_bytes == b"fake-video"
    assert mux_calls[0]["frames"].shape == (5, 2, 2, 3)
    assert mux_calls[0]["frames"].dtype == np.uint8
    assert mux_calls[0]["fps"] == 8.0
    assert mux_calls[0]["audio"] is None


def test_float_frames_are_converted_without_stacking_full_video(monkeypatch):
    frame = np.array(
        [
            [[0.0, 0.5, 1.0, 0.2], [1.5, -0.5, 0.5, 0.8]],
        ],
        dtype=np.float32,
    )
    original = frame.copy()

    def fail_stack(*args, **kwargs):
        raise AssertionError("float video conversion must not stack all frames")

    monkeypatch.setattr(video_api_utils.np, "stack", fail_stack)

    frames = video_api_utils._coerce_video_to_uint8_frames([frame, frame])

    expected = np.array([[[128, 191, 255], [255, 64, 191]]], dtype=np.uint8)
    assert frames.flags.c_contiguous
    np.testing.assert_array_equal(frames, np.array([expected, expected]))
    np.testing.assert_array_equal(frame, original)


def test_two_dimensional_width_four_frames_are_not_treated_as_rgba():
    frame = np.arange(8, dtype=np.float32).reshape(2, 4) / 8

    frames = video_api_utils._coerce_video_to_uint8_frames([frame, frame])

    expected = np.rint(frame * 255).astype(np.uint8)
    assert frames.shape == (2, 2, 4)
    np.testing.assert_array_equal(frames, np.stack([expected, expected]))


def test_mixed_float_dtypes_preserve_stacked_rounding_semantics():
    frames = [
        np.full((1, 1, 3), 0.1, dtype=np.float16),
        np.full((1, 1, 3), 0.2, dtype=np.float32),
    ]
    stacked = np.stack(frames)
    expected = np.rint(np.clip(stacked, 0.0, 1.0) * 255.0).astype(np.uint8)

    actual = video_api_utils._coerce_video_to_uint8_frames(frames)

    assert expected[0, 0, 0, 0] == 25
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("frame_count", [3, 4])
def test_channel_last_video_tensor_preserves_channel_sized_frame_count(frame_count):
    video = torch.arange(frame_count * 2 * 5 * 3, dtype=torch.uint8).reshape(frame_count, 2, 5, 3)

    frames = video_api_utils._coerce_video_to_uint8_frames(video)

    assert frames.shape == (frame_count, 2, 5, 3)
    np.testing.assert_array_equal(frames, video.numpy())


def test_channel_first_video_tensor_is_converted_to_channel_last():
    video = torch.arange(3 * 5 * 2 * 6, dtype=torch.uint8).reshape(3, 5, 2, 6)

    frames = video_api_utils._coerce_video_to_uint8_frames(video)

    assert frames.shape == (5, 2, 6, 3)
    np.testing.assert_array_equal(frames, video.permute(1, 2, 3, 0).numpy())


def test_fragmented_mp4_video_encoder_reuses_single_muxer(monkeypatch):
    muxers = []

    class FakeFragmentedMP4Muxer:
        def __init__(self, *, width, height, fps, video_codec_options=None):
            self.calls = []
            muxers.append(
                {
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "video_codec_options": video_codec_options,
                    "instance": self,
                }
            )

        def mux_video_frames(self, frames):
            self.calls.append(frames)
            return f"fragment-{len(self.calls)}".encode()

        def close(self):
            return b"trailer"

    monkeypatch.setattr(
        "vllm_omni.diffusion.utils.media_utils.FragmentedMP4Muxer",
        FakeFragmentedMP4Muxer,
    )

    encoder = video_api_utils.FragmentedMP4VideoEncoder(
        fps=12,
        video_codec_options={"preset": "ultrafast"},
    )
    assert encoder.encode([np.zeros((4, 6, 3), dtype=np.float32)]) == b"fragment-1"
    assert encoder.encode([np.ones((4, 6, 3), dtype=np.float32)]) == b"fragment-2"
    assert encoder.close() == b"trailer"

    assert len(muxers) == 1
    assert muxers[0]["width"] == 6
    assert muxers[0]["height"] == 4
    assert muxers[0]["fps"] == 12.0
    assert muxers[0]["video_codec_options"] == {"preset": "ultrafast"}
    assert len(muxers[0]["instance"].calls) == 2


def test_create_streaming_video_encoder_selects_requested_format():
    assert isinstance(
        video_api_utils.create_streaming_video_encoder(output_format="m4s", fps=12),
        video_api_utils.FragmentedMP4VideoEncoder,
    )


def test_finalize_streaming_mp4_bytes_produces_progressive_mp4():
    """Fragment MP4 chunks are remuxed into decodable progressive MP4 bytes."""
    import numpy as np

    from vllm_omni.diffusion.utils.media_utils import FragmentedMP4Muxer, finalize_streaming_video_bytes

    def _read_mp4_video_info(mp4_bytes: bytes) -> tuple[int, float, float | None, int, int]:
        import io

        import av

        with av.open(io.BytesIO(mp4_bytes), mode="r", format="mp4") as container:
            stream = container.streams.video[0]
            frame_count = sum(1 for _ in container.decode(stream))
            assert stream.average_rate is not None
            fps = float(stream.average_rate)
            duration_sec = None
            if stream.duration is not None:
                assert stream.time_base is not None
                duration_sec = float(stream.duration * stream.time_base)
            return frame_count, fps, duration_sec, stream.width, stream.height

    width = 32
    height = 32
    fps = 16.0
    input_frame_count = 2

    muxer = FragmentedMP4Muxer(width=width, height=height, fps=fps)
    frames = np.zeros((input_frame_count, height, width, 3), dtype=np.uint8)
    streamed = muxer.mux_video_frames(frames) + muxer.close()

    finalized = finalize_streaming_video_bytes(streamed, input_format="m4s", fps=fps)
    assert finalized
    assert finalized != streamed

    streamed_info = _read_mp4_video_info(streamed)
    final_info = _read_mp4_video_info(finalized)
    expected_duration = input_frame_count / fps

    assert streamed_info[0] == input_frame_count
    assert final_info[0] == input_frame_count
    assert streamed_info[1] == pytest.approx(fps)
    assert final_info[1] == pytest.approx(fps)
    assert streamed_info[3:] == (width, height)
    assert final_info[3:] == (width, height)

    assert final_info[2] == pytest.approx(expected_duration, rel=0.05)
    assert streamed_info[2] == pytest.approx(expected_duration, rel=0.05)


def test_rife_model_inference_runs_on_dummy_tensors():
    model = rife_interpolator.Model().eval()
    img0 = torch.rand(1, 3, 32, 32)
    img1 = torch.rand(1, 3, 32, 32)

    output = model.inference(img0, img1, scale=1.0)

    assert output.shape == (1, 3, 32, 32)
    assert torch.isfinite(output).all()


def test_frame_interpolator_runs_actual_torch_tensor_path(monkeypatch):
    model = rife_interpolator.Model().eval()
    interpolator = rife_interpolator.FrameInterpolator()
    monkeypatch.setattr(interpolator, "_ensure_model_loaded", lambda preferred_device=None: model)

    video = torch.zeros(1, 3, 2, 32, 32)
    output_video, multiplier = interpolator.interpolate_tensor(video, exp=1, scale=1.0)

    assert multiplier == 2
    assert output_video.shape == (1, 3, 3, 32, 32)
    assert torch.isfinite(output_video).all()


def test_frame_interpolator_uses_platform_device_when_tensor_is_cpu(monkeypatch):
    chosen_devices = []
    model = rife_interpolator.Model().eval()

    def _fake_ensure_model_loaded(*, preferred_device=None):
        chosen_devices.append(preferred_device)
        return model

    interpolator = rife_interpolator.FrameInterpolator()
    monkeypatch.setattr(interpolator, "_ensure_model_loaded", _fake_ensure_model_loaded)
    monkeypatch.setattr(model.flownet, "to", lambda device: model.flownet)
    monkeypatch.setattr(rife_interpolator, "_select_torch_device", lambda: torch.device("cuda"))

    video = torch.zeros(1, 3, 2, 32, 32)
    output_video, multiplier = interpolator.interpolate_tensor(video, exp=1, scale=1.0)

    assert chosen_devices == [torch.device("cuda")]
    assert multiplier == 2
    assert output_video.shape == (1, 3, 3, 32, 32)
