# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OpenAI-compatible video API encoding helpers."""

import base64
from io import BytesIO

import av
import httpx
import numpy as np
import pytest
import torch
from PIL import Image
from vllm import envs

from vllm_omni.diffusion.postprocess import rife_interpolator
from vllm_omni.diffusion.utils import media_utils
from vllm_omni.entrypoints.openai import video_api_utils
from vllm_omni.entrypoints.openai.errors import InvalidInputReferenceError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 1), color=(12, 34, 56)).save(buffer, format="PNG")
    return buffer.getvalue()


def _install_http_transport(monkeypatch, handler, client_kwargs):
    async_client = httpx.AsyncClient

    def _client_factory(*args, **kwargs):
        client_kwargs.append(kwargs.copy())
        return async_client(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(video_api_utils.httpx, "AsyncClient", _client_factory)


@pytest.mark.asyncio
async def test_decode_image_url_follows_redirects_when_allowed(monkeypatch):
    requested_paths = []
    client_kwargs = []

    def _handler(request):
        requested_paths.append(request.url.path)
        if request.url.path == "/redirect.png":
            return httpx.Response(302, headers={"location": "/image.png"})
        return httpx.Response(200, content=_png_bytes(), headers={"content-type": "image/png"})

    monkeypatch.setattr(envs, "VLLM_MEDIA_URL_ALLOW_REDIRECTS", True)
    _install_http_transport(monkeypatch, _handler, client_kwargs)

    image = await video_api_utils.decode_image_url("https://example.com/redirect.png")

    assert image.size == (2, 1)
    assert image.mode == "RGB"
    assert requested_paths == ["/redirect.png", "/image.png"]
    assert client_kwargs == [{"timeout": 60, "follow_redirects": True}]


@pytest.mark.asyncio
async def test_decode_image_url_rejects_redirects_when_disabled(monkeypatch):
    requested_paths = []
    client_kwargs = []

    def _handler(request):
        requested_paths.append(request.url.path)
        return httpx.Response(302, headers={"location": "/image.png"})

    monkeypatch.setattr(envs, "VLLM_MEDIA_URL_ALLOW_REDIRECTS", False)
    _install_http_transport(monkeypatch, _handler, client_kwargs)

    with pytest.raises(InvalidInputReferenceError, match="redirect.*VLLM_MEDIA_URL_ALLOW_REDIRECTS"):
        await video_api_utils.decode_image_url("https://example.com/redirect.png")

    assert requested_paths == ["/redirect.png"]
    assert client_kwargs == [{"timeout": 60, "follow_redirects": False}]


@pytest.mark.asyncio
async def test_decode_image_url_reports_http_status(monkeypatch):
    client_kwargs = []

    def _handler(request):
        return httpx.Response(404)

    monkeypatch.setattr(envs, "VLLM_MEDIA_URL_ALLOW_REDIRECTS", True)
    _install_http_transport(monkeypatch, _handler, client_kwargs)

    with pytest.raises(InvalidInputReferenceError, match="server returned HTTP 404"):
        await video_api_utils.decode_image_url("https://example.com/missing.png")


@pytest.mark.asyncio
async def test_decode_image_url_reports_connection_failure(monkeypatch):
    client_kwargs = []

    def _handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr(envs, "VLLM_MEDIA_URL_ALLOW_REDIRECTS", True)
    _install_http_transport(monkeypatch, _handler, client_kwargs)

    with pytest.raises(InvalidInputReferenceError, match="failed to download image"):
        await video_api_utils.decode_image_url("https://example.com/unreachable.png")


@pytest.mark.asyncio
async def test_decode_image_url_keeps_data_urls_local(monkeypatch):
    def _unexpected_http_client(*args, **kwargs):
        pytest.fail("data URLs must not create an HTTP client")

    monkeypatch.setattr(video_api_utils.httpx, "AsyncClient", _unexpected_http_client)
    encoded_image = base64.b64encode(_png_bytes()).decode()

    image = await video_api_utils.decode_image_url(f"data:image/png;base64,{encoded_image}")

    assert image.size == (2, 1)
    assert image.mode == "RGB"


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


def test_channel_first_video_tensor_uses_direct_planar_mux(monkeypatch):
    calls = []

    def fake_planar_mux(frames, width, height, audio_waveform, **kwargs):
        calls.append((list(frames), width, height, audio_waveform, kwargs))
        return b"planar-video"

    def fail_compat_mux(*args, **kwargs):
        raise AssertionError("CFHW-backed FHWC frames should use the planar path")

    monkeypatch.setattr(media_utils, "mux_av_video_audio_bytes", fake_planar_mux)
    monkeypatch.setattr(media_utils, "mux_video_audio_bytes", fail_compat_mux)

    video = torch.linspace(-1, 1, 3 * 2 * 4 * 6, dtype=torch.float32).reshape(3, 2, 4, 6)
    encoded = video_api_utils._encode_video_bytes(
        video,
        fps=12,
        encoding_config=video_api_utils.OPTIMIZED_VIDEO_RESPONSE_ENCODING,
    )

    assert encoded == b"planar-video"
    assert len(calls) == 1
    frames, width, height, audio, options = calls[0]
    assert (len(frames), width, height) == (2, 6, 4)
    assert all(frame.format.name == "gbrp" for frame in frames)
    assert audio is None
    assert options["fps"] == 12.0


def test_channel_first_video_tensor_keeps_legacy_path_by_default(monkeypatch):
    calls = []

    def fail_planar_mux(*args, **kwargs):
        raise AssertionError("the optimized path must remain opt-in")

    def fake_compat_mux(frames, audio, **kwargs):
        calls.append((frames, audio, kwargs))
        return b"legacy-video"

    monkeypatch.setattr(media_utils, "mux_av_video_audio_bytes", fail_planar_mux)
    monkeypatch.setattr(media_utils, "mux_video_audio_bytes", fake_compat_mux)

    video = torch.zeros((3, 2, 4, 6), dtype=torch.float32)
    encoded = video_api_utils._encode_video_bytes(video, fps=12)

    assert encoded == b"legacy-video"
    assert calls[0][0].shape == (2, 4, 6, 3)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("channels", [3, 4])
def test_planar_frames_decode_with_bounded_rgb_quantization(dtype, channels):
    channel_values = np.array(
        [
            [[-0.1, 0.0, 0.1, 0.5, 0.9, 1.0, 1.1]],
            [[1.0, 0.9, 0.5, 0.1, 0.0, -0.1, 1.1]],
            [[0.25, 0.75, 0.501, 0.499, 0.125, 0.875, 0.5]],
            [[0.2, 0.4, 0.6, 0.8, 1.0, 0.0, 0.5]],
        ],
        dtype=dtype,
    )[:channels]
    fhwc = np.transpose(channel_values[:, None, :, :], (1, 2, 3, 0))
    frames = list(video_api_utils._iter_planar_video_frames(list(fhwc), np.dtype(dtype)))

    decoded = frames[0].to_ndarray(format="rgb24")
    expected = np.rint(np.clip(fhwc[0, ..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(decoded, expected)


def test_planar_frame_writes_gbr_planes_and_clears_padding():
    width = 17
    planar = np.stack(
        [
            np.full((2, width), 11, dtype=np.uint8),
            np.full((2, width), 22, dtype=np.uint8),
            np.full((2, width), 33, dtype=np.uint8),
        ]
    )
    fhwc = np.transpose(planar[:, None, :, :], (1, 2, 3, 0))

    frame = next(video_api_utils._iter_planar_video_frames(list(fhwc), np.dtype(np.uint8)))

    for plane, expected in zip(frame.planes, (22, 33, 11)):
        plane_data = np.frombuffer(plane, dtype=np.uint8).reshape(plane.height, plane.line_size)
        np.testing.assert_array_equal(plane_data[:2, :width], expected)
        np.testing.assert_array_equal(plane_data[:2, width:], 0)
        np.testing.assert_array_equal(plane_data[2:], 0)


def test_planar_bool_frames_match_bounded_compatible_output():
    planar = np.array(
        [
            [[[False, True], [True, False]]],
            [[[True, False], [True, False]]],
            [[[False, False], [True, True]]],
        ],
        dtype=np.bool_,
    )
    direct_video = np.transpose(planar, (1, 2, 3, 0))
    compatible_video = np.ascontiguousarray(direct_video)

    prepared_frames, _, common_dtype = video_api_utils._prepare_video_frames(direct_video)
    direct_frames = list(video_api_utils._iter_planar_video_frames(prepared_frames, common_dtype))
    decoded_direct = np.stack([frame.to_ndarray(format="rgb24") for frame in direct_frames])
    bounded_compatible = video_api_utils._coerce_video_to_uint8_frames(compatible_video)

    np.testing.assert_array_equal(decoded_direct, bounded_compatible)


@pytest.mark.parametrize(
    ("plane_height", "line_size"),
    [(1, 2), (2, 1)],
)
def test_planar_frame_rejects_undersized_plane(monkeypatch, plane_height, line_size):
    class FakePlane:
        pass

    plane = FakePlane()
    plane.height = plane_height
    plane.line_size = line_size

    class FakeFrame:
        planes = [plane]

    monkeypatch.setattr(av, "VideoFrame", lambda *args, **kwargs: FakeFrame())
    planar = np.zeros((3, 1, 2, 2), dtype=np.uint8)
    fhwc = np.transpose(planar, (1, 2, 3, 0))

    with pytest.raises(ValueError, match="smaller than"):
        next(video_api_utils._iter_planar_video_frames(list(fhwc), np.dtype(np.uint8)))


def test_interleaved_video_uses_compatible_mux(monkeypatch):
    calls = []

    def fail_planar_mux(*args, **kwargs):
        raise AssertionError("interleaved FHWC frames must use the compatible path")

    def fake_compat_mux(frames, audio, **kwargs):
        calls.append((frames, audio, kwargs))
        return b"compatible-video"

    monkeypatch.setattr(media_utils, "mux_av_video_audio_bytes", fail_planar_mux)
    monkeypatch.setattr(media_utils, "mux_video_audio_bytes", fake_compat_mux)

    video = np.zeros((2, 4, 6, 3), dtype=np.float32)
    encoded = video_api_utils._encode_video_bytes(
        video,
        fps=12,
        encoding_config=video_api_utils.OPTIMIZED_VIDEO_RESPONSE_ENCODING,
    )

    assert encoded == b"compatible-video"
    assert calls[0][0].shape == (2, 4, 6, 3)
    assert calls[0][0].dtype == np.uint8


def test_unequal_frame_shapes_fail_before_output_allocation(monkeypatch):
    def fail_empty(*args, **kwargs):
        raise AssertionError("output allocation must happen after shape validation")

    monkeypatch.setattr(video_api_utils.np, "empty", fail_empty)

    with pytest.raises(ValueError, match="same shape"):
        video_api_utils._encode_video_bytes(
            [np.zeros((2, 3, 3), dtype=np.float32), np.zeros((3, 3, 3), dtype=np.float32)],
            fps=12,
            encoding_config=video_api_utils.OPTIMIZED_VIDEO_RESPONSE_ENCODING,
        )


@pytest.mark.parametrize("with_audio", [False, True])
def test_direct_and_compatible_paths_produce_equivalent_mp4(with_audio):
    rng = np.random.default_rng(7)
    planar = rng.random((3, 3, 32, 32), dtype=np.float32)
    direct_video = np.transpose(planar, (1, 2, 3, 0))
    compatible_video = np.ascontiguousarray(direct_video)
    audio = np.zeros((2, 2400), dtype=np.float32) if with_audio else None

    direct_bytes = video_api_utils._encode_video_bytes(
        direct_video,
        fps=12,
        audio=audio,
        audio_sample_rate=24000,
        video_codec_options={"preset": "ultrafast", "threads": "1"},
        encoding_config=video_api_utils.OPTIMIZED_VIDEO_RESPONSE_ENCODING,
    )
    compatible_bytes = video_api_utils._encode_video_bytes(
        compatible_video,
        fps=12,
        audio=audio,
        audio_sample_rate=24000,
        video_codec_options={"preset": "ultrafast", "threads": "1"},
        encoding_config=video_api_utils.OPTIMIZED_VIDEO_RESPONSE_ENCODING,
    )

    def decode(mp4_bytes):
        with av.open(BytesIO(mp4_bytes), mode="r", format="mp4") as container:
            stream_types = [stream.type for stream in container.streams]
            video_stream = container.streams.video[0]
            frames = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video_stream)])
        return stream_types, frames

    direct_streams, direct_frames = decode(direct_bytes)
    compatible_streams, compatible_frames = decode(compatible_bytes)
    assert direct_streams == compatible_streams
    assert ("audio" in direct_streams) is with_audio
    np.testing.assert_array_equal(direct_frames, compatible_frames)


def test_mux_closes_container_and_preserves_generator_error(monkeypatch):
    class GeneratorError(RuntimeError):
        pass

    class FakeStream:
        options = None

        def encode(self, frame=None):
            return []

    class FakeContainer:
        def __init__(self):
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.closed = True
            return False

        def add_stream(self, codec, rate):
            return FakeStream()

        def mux(self, packet):
            pass

    container = FakeContainer()
    monkeypatch.setattr(media_utils.av, "open", lambda *args, **kwargs: container)

    def failing_frames():
        yield object()
        raise GeneratorError("frame generation failed")

    with pytest.raises(GeneratorError, match="frame generation failed"):
        media_utils.mux_av_video_audio_bytes(failing_frames(), width=2, height=2)
    assert container.closed


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
