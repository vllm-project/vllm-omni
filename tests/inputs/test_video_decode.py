# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from concurrent.futures import ThreadPoolExecutor
from inspect import signature
from io import BytesIO
from pathlib import Path
from threading import Barrier

import av
import cv2
import numpy as np
import pytest
import vllm
from packaging.version import Version
from vllm.multimodal.video import Qwen2VLVideoBackend, VideoBackend
from vllm.multimodal.video_decoders import opencv

import vllm_omni  # noqa: F401  # Install the compatibility patch through its real entrypoint.
from vllm_omni.inputs.video import _AFFECTED_VLLM_VERSIONS, apply_opencv_capture_reuse_patch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

requires_capture_reuse_workaround = pytest.mark.skipif(
    Version(vllm.__version__).public not in _AFFECTED_VLLM_VERSIONS
    or "cap" in signature(opencv.OpenCVVideoBackendMixin.get_video_metadata).parameters,
    reason="Requires a reviewed vLLM version without the upstream fix; retirement checks remain unconditional.",
)


@pytest.fixture(params=["mp4", "webm", "h264", "trimmed_mp4"])
def video_bytes(request):
    buffer = BytesIO()
    trimmed = request.param == "trimmed_mp4"
    fmt = "mp4" if trimmed else request.param
    options = {"live": "1"} if fmt == "webm" else None
    with av.open(buffer, mode="w", format=fmt, options=options) as container:
        stream = container.add_stream("libvpx-vp9" if fmt == "webm" else "h264", rate=10)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        if trimmed:
            stream.codec_context.gop_size = 40
            stream.codec_context.max_b_frames = 0
            stream.codec_context.options = {"x264-params": "scenecut=0:keyint=40:min-keyint=40"}
        for index in range(40 if trimmed else 20):
            pixels = np.full((32, 32, 3), index * 5, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    if trimmed:
        # Hide 20 lead-in frames using an MP4 edit list, without re-encoding.
        output = BytesIO()
        with av.open(BytesIO(buffer.getvalue())) as source, av.open(output, mode="w", format="mp4") as target:
            source_stream = source.streams.video[0]
            target_stream = target.add_stream_from_template(source_stream)
            start_pts = round(2 / source_stream.time_base)
            for packet in source.demux(source_stream):
                if packet.pts is None or packet.dts is None:
                    continue
                packet.pts -= start_pts
                packet.dts -= start_pts
                packet.stream = target_stream
                target.mux(packet)
        return output.getvalue()
    return buffer.getvalue()


@pytest.fixture
def native_decoder():
    decode = getattr(opencv.decode_opencv, "__wrapped__", opencv.decode_opencv)
    open_capture = opencv.OpenCVVideoBackendMixin.__dict__["open_video_capture"].__func__
    open_capture = getattr(open_capture, "__wrapped__", open_capture)
    return decode, open_capture


def _restore_native(monkeypatch, native_decoder):
    decode, open_capture = native_decoder
    monkeypatch.setattr(opencv, "decode_opencv", decode)
    monkeypatch.setattr(opencv.OpenCVVideoBackendMixin, "open_video_capture", classmethod(open_capture))


@requires_capture_reuse_workaround
@pytest.mark.parametrize("loader", [VideoBackend, Qwen2VLVideoBackend])
@pytest.mark.parametrize("frame_recovery", [False, True])
def test_video_decode_reuses_capture_without_changing_frames(
    video_bytes, loader, frame_recovery, native_decoder, monkeypatch
):
    with monkeypatch.context() as control:
        _restore_native(control, native_decoder)
        expected, expected_metadata = loader.load_bytes(
            video_bytes, num_frames=4, backend="opencv", frame_recovery=frame_recovery
        )

    original = cv2.VideoCapture
    opens = []

    def track_capture(stream, backend, params):
        if params != [cv2.CAP_PROP_FORMAT, -1]:
            opens.append(stream)
        return original(stream, backend, params)

    monkeypatch.setattr(cv2, "VideoCapture", track_capture)
    actual, metadata = loader.load_bytes(video_bytes, num_frames=4, backend="opencv", frame_recovery=frame_recovery)
    np.testing.assert_array_equal(actual, expected)
    assert metadata == expected_metadata
    assert metadata["total_num_frames"] == 20
    assert actual.shape[0] == len(metadata["frames_indices"]) == 4
    assert len(opens) == 1


@pytest.mark.parametrize("dockerfile", ["Dockerfile.cuda", "Dockerfile.ci"])
def test_video_workaround_is_reviewed_when_vllm_pin_changes(dockerfile):
    lines = (Path(__file__).resolve().parents[2] / "docker" / dockerfile).read_text().splitlines()
    pin = next(line.split("=", 1)[1] for line in lines if line.startswith("ARG VLLM_VERSION="))
    assert pin in _AFFECTED_VLLM_VERSIONS, (
        "vLLM pin changed: review inputs/video.py against vllm#66466; remove the workaround "
        "when the fix is included, or reproduce the regression before extending its version guard."
    )


def test_video_workaround_is_retired_when_upstream_reuses_capture(video_bytes, native_decoder, monkeypatch):
    _restore_native(monkeypatch, native_decoder)
    original = cv2.VideoCapture
    opens = []

    def track_capture(stream, backend, params):
        if params != [cv2.CAP_PROP_FORMAT, -1]:
            opens.append(stream)
        return original(stream, backend, params)

    monkeypatch.setattr(cv2, "VideoCapture", track_capture)
    VideoBackend.load_bytes(video_bytes, num_frames=4, backend="opencv")
    assert len(opens) == 2, "Upstream decoder changed: review/remove inputs/video.py (vllm#66466)."


@pytest.mark.parametrize(
    "version",
    ["0.32.0", "0.33.1", "0.34.0rc0", "0.34.0rc2", "0.34.0rc4", "0.34.0", "0.33.0.post1", "0.33.0.dev1", "dev"],
)
def test_video_workaround_leaves_other_versions_untouched(version, native_decoder, monkeypatch):
    _restore_native(monkeypatch, native_decoder)
    monkeypatch.setattr(vllm, "__version__", version)
    apply_opencv_capture_reuse_patch()
    assert opencv.decode_opencv is native_decoder[0]
    assert opencv.OpenCVVideoBackendMixin.open_video_capture.__func__ is native_decoder[1]


@pytest.mark.parametrize("version", sorted(_AFFECTED_VLLM_VERSIONS))
def test_video_workaround_is_idempotent_for_local_builds(version, native_decoder, monkeypatch):
    _restore_native(monkeypatch, native_decoder)
    monkeypatch.setattr(vllm, "__version__", f"{version}+cu130")
    apply_opencv_capture_reuse_patch()
    decoder = opencv.decode_opencv
    opener = opencv.OpenCVVideoBackendMixin.open_video_capture
    apply_opencv_capture_reuse_patch()
    assert opencv.decode_opencv is decoder
    assert opencv.OpenCVVideoBackendMixin.open_video_capture == opener


def test_video_workaround_skips_upstream_backports(native_decoder, monkeypatch):
    _restore_native(monkeypatch, native_decoder)
    original = opencv.OpenCVVideoBackendMixin.get_video_metadata

    def fixed_metadata(data, *, cap=None):
        return original(data)

    monkeypatch.setattr(opencv.OpenCVVideoBackendMixin, "get_video_metadata", staticmethod(fixed_metadata))
    apply_opencv_capture_reuse_patch()
    assert opencv.decode_opencv is native_decoder[0]
    assert opencv.OpenCVVideoBackendMixin.open_video_capture.__func__ is native_decoder[1]


@requires_capture_reuse_workaround
def test_video_capture_is_isolated_between_threads(video_bytes, monkeypatch):
    expected, metadata = VideoBackend.load_bytes(video_bytes, num_frames=4, backend="opencv")
    original = cv2.VideoCapture
    barrier = Barrier(2)
    captures = []

    def concurrent_capture(stream, backend, params):
        cap = original(stream, backend, params)
        if params != [cv2.CAP_PROP_FORMAT, -1]:
            captures.append(cap)
            barrier.wait(timeout=10)
        return cap

    monkeypatch.setattr(cv2, "VideoCapture", concurrent_capture)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(VideoBackend.load_bytes, video_bytes, num_frames=4, backend="opencv") for _ in range(2)]
        for future in futures:
            frames, actual_metadata = future.result(timeout=20)
            np.testing.assert_array_equal(frames, expected)
            assert actual_metadata == metadata
    assert len(captures) == 2
    assert captures[0] is not captures[1]


@requires_capture_reuse_workaround
def test_video_capture_scope_is_reset_after_failure(video_bytes, monkeypatch):
    original = opencv.OpenCVVideoBackendMixin.get_video_metadata

    def fail_metadata(data):
        raise ValueError("metadata failure")

    with monkeypatch.context() as failure:
        failure.setattr(opencv.OpenCVVideoBackendMixin, "get_video_metadata", staticmethod(fail_metadata))
        with pytest.raises(ValueError, match="metadata failure"):
            VideoBackend.load_bytes(video_bytes, num_frames=4, backend="opencv")

    native_capture = cv2.VideoCapture
    opens = []

    def track_capture(*args):
        opens.append(args)
        return native_capture(*args)

    monkeypatch.setattr(cv2, "VideoCapture", track_capture)
    assert original(video_bytes).total_frames_num == 20
    assert opens, "A standalone metadata call must not reuse a failed decode's capture."
