# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import importlib.util
import os
import sys
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from types import ModuleType
from typing import Any

import av
import numpy as np
import pytest
import torch
from comfyui_vllm_omni.utils.api_client import MINIMAX_H3_CONTROL_VIDEO_ENCODING
from comfyui_vllm_omni.utils.format import video_to_bytes

pytestmark = [pytest.mark.local_model, pytest.mark.diffusion, pytest.mark.cpu]


class _ProgressBar:
    """Stand-in for comfy.utils.ProgressBar, which the real transcode path instantiates."""

    def __init__(self, total: int) -> None:
        self.total = total

    def update(self, value: int) -> None:
        pass

    def update_absolute(self, value: int, total: int | None = None) -> None:
        pass


def _find_comfyui_source() -> Path:
    candidates: list[Path] = []
    if configured := os.environ.get("COMFYUI_SOURCE_ROOT"):
        candidates.append(Path(configured))
    candidates.extend(parent / "ComfyUI" for parent in Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / "comfy_api/latest/_input_impl/video_types.py").is_file():
            return candidate
    return pytest.skip("Real ComfyUI source checkout not found; set COMFYUI_SOURCE_ROOT to run this encoding proof.")


def _package(monkeypatch: pytest.MonkeyPatch, name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_module(monkeypatch: pytest.MonkeyPatch, name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _load_real_video_types(
    monkeypatch: pytest.MonkeyPatch,
    comfyui_root: Path,
) -> tuple[type[Any], type[Any], type[Any], type[Any], type[Any]]:
    prefix = "ctrl02_real_comfy_api.latest"
    _package(monkeypatch, "ctrl02_real_comfy_api")
    _package(monkeypatch, prefix)
    input_package = _package(monkeypatch, f"{prefix}._input")
    util_package = _package(monkeypatch, f"{prefix}._util")
    _package(monkeypatch, f"{prefix}._input_impl")
    setattr(input_package, "ImageInput", torch.Tensor)
    setattr(input_package, "MaskInput", torch.Tensor)
    setattr(input_package, "AudioInput", dict)

    source_root = comfyui_root / "comfy_api/latest"
    util_types = _load_module(
        monkeypatch,
        f"{prefix}._util.video_types",
        source_root / "_util/video_types.py",
    )
    for name in ("VideoContainer", "VideoCodec", "VideoComponents", "normalize_crop_rect"):
        setattr(util_package, name, getattr(util_types, name))

    input_types = _load_module(
        monkeypatch,
        f"{prefix}._input.video_types",
        source_root / "_input/video_types.py",
    )
    setattr(input_package, "VideoInput", input_types.VideoInput)

    comfy_package = _package(monkeypatch, "comfy")
    comfy_utils = ModuleType("comfy.utils")
    setattr(comfy_utils, "ProgressBar", _ProgressBar)
    setattr(comfy_package, "utils", comfy_utils)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils)

    input_impl = _load_module(
        monkeypatch,
        f"{prefix}._input_impl.video_types",
        source_root / "_input_impl/video_types.py",
    )
    return (
        input_impl.VideoFromComponents,
        input_impl.VideoFromFile,
        util_types.VideoComponents,
        util_types.VideoContainer,
        util_types.VideoCodec,
    )


def test_real_comfyui_temporal_mask_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    video_cls, _, components_cls, container_cls, codec_cls = _load_real_video_types(
        monkeypatch,
        _find_comfyui_source(),
    )
    images = torch.zeros((3, 32, 32, 3), dtype=torch.float32)
    images[0, :, 16:, :] = 1.0
    images[1, 16:, :, :] = 1.0
    images[2, 8:24, 8:24, :] = 1.0
    video = video_cls(components_cls(images=images, frame_rate=Fraction(8, 1)))

    encoded = video_to_bytes(video, "mask.mp4", **MINIMAX_H3_CONTROL_VIDEO_ENCODING)

    assert encoded.tell() == 0
    with av.open(encoded) as container:
        stream = container.streams.video[0]
        decoded = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]) / 255.0
        assert container.format.name.startswith("mov,mp4")
        assert stream.codec_context.name == "h264"
    assert isinstance(video, video_cls)
    assert container_cls.MP4.value == "mp4"
    assert codec_cls.H264.value == "h264"
    assert decoded.shape == tuple(images.shape)
    assert np.array_equal(decoded[..., 0] > 0.5, images[..., 0].numpy() > 0.5)


def test_real_comfyui_structure_hint_survives_the_client_encode(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 1px structure hint must reach the server unchanged, which default CRF does not do."""
    video_cls, _, components_cls, _, _ = _load_real_video_types(monkeypatch, _find_comfyui_source())
    images = torch.zeros((6, 128, 128, 3), dtype=torch.float32)
    for index in range(images.shape[0]):
        images[index, :, 30 + index, :] = 1.0
        images[index, 60 + index, :, :] = 1.0
    video = video_cls(components_cls(images=images, frame_rate=Fraction(24, 1)))

    encoded = video_to_bytes(video, "control.mp4", **MINIMAX_H3_CONTROL_VIDEO_ENCODING)

    with av.open(encoded) as container:
        stream = container.streams.video[0]
        decoded = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(stream)])
        assert container.format.name.startswith("mov,mp4")
        assert stream.codec_context.name == "h264"
    assert np.array_equal(decoded, (images.numpy() * 255).astype(np.uint8))


def test_real_comfyui_non_mp4_source_is_uploaded_as_the_declared_mp4(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without pinned options ComfyUI reuses the source container, contradicting the declared upload type."""
    _, file_video_cls, _, _, _ = _load_real_video_types(monkeypatch, _find_comfyui_source())
    source = BytesIO()
    with av.open(source, mode="w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=Fraction(24, 1))
        stream.width, stream.height, stream.pix_fmt = 64, 64, "gray"
        for index in range(4):
            frame = np.zeros((64, 64), dtype=np.uint8)
            frame[:, 10 + index] = 255
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="gray")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    encoded = video_to_bytes(
        file_video_cls(BytesIO(source.getvalue())),
        "control.mp4",
        **MINIMAX_H3_CONTROL_VIDEO_ENCODING,
    )

    with av.open(encoded) as container:
        assert container.format.name.startswith("mov,mp4")
        assert container.streams.video[0].codec_context.name == "h264"


@pytest.mark.parametrize("channels", [0, 1, 2])
def test_real_comfyui_generated_audio_survives_decode_and_save(monkeypatch: pytest.MonkeyPatch, channels: int) -> None:
    from types import SimpleNamespace

    from comfyui_vllm_omni.utils import format as media_format

    video_cls, _, components_cls, _, _ = _load_real_video_types(monkeypatch, _find_comfyui_source())
    monkeypatch.setattr(media_format, "Types", SimpleNamespace(VideoComponents=components_cls))
    monkeypatch.setattr(media_format, "InputImpl", SimpleNamespace(VideoFromComponents=video_cls))
    sample_rate = 48000
    audio = None
    if channels:
        time = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
        waveform = torch.stack(
            [0.25 * torch.sin(2 * torch.pi * frequency * time) for frequency in (440, 880)[:channels]]
        )
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
    source = video_cls(components_cls(images=torch.zeros((24, 32, 32, 3)), frame_rate=Fraction(24), audio=audio))
    encoded = video_to_bytes(source, "generated.mp4", **MINIMAX_H3_CONTROL_VIDEO_ENCODING).getvalue()

    decoded = media_format.bytes_to_video(encoded)
    components = decoded.get_components()
    assert components.images.shape == (24, 32, 32, 3)
    assert components.frame_rate == 24
    if not channels:
        assert components.audio is None
        return
    assert components.audio is not None
    assert components.audio["sample_rate"] == sample_rate
    received = components.audio["waveform"]
    assert received.shape[:2] == (1, channels)
    assert abs(received.shape[-1] - sample_rate) <= 1024
    assert torch.mean(received.square()) > 0.01

    saved = video_to_bytes(decoded, "comfy-output.mp4", **MINIMAX_H3_CONTROL_VIDEO_ENCODING)
    with av.open(saved) as container:
        assert len(container.streams.audio) == 1
        frames = list(container.decode(audio=0))
        assert sum(frame.samples for frame in frames) >= sample_rate
