# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Temporary capture reuse for vLLM 0.33.0; remove after vllm#66466."""

from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
from inspect import signature
from typing import TYPE_CHECKING

from packaging.version import InvalidVersion, Version

if TYPE_CHECKING:
    import cv2
    import numpy as np
    import numpy.typing as npt
    from vllm.multimodal.video import VideoBackend
    from vllm.multimodal.video_decoders.base import VideoSourceMetadata, VideoTargetMetadata
    from vllm.multimodal.video_decoders.opencv import OpenCVVideoBackendMixin

_AFFECTED_VLLM_VERSION = "0.33.0"


def apply_opencv_capture_reuse_patch() -> None:
    import vllm

    try:
        version = Version(vllm.__version__)
    except InvalidVersion:
        return
    if version.public != _AFFECTED_VLLM_VERSION:
        return

    from vllm.multimodal.video_decoders import opencv

    original_decode = opencv.decode_opencv
    if getattr(original_decode, "_omni_capture_reuse", False):
        return
    mixin = opencv.OpenCVVideoBackendMixin
    # Also leave same-version builds carrying the upstream fix untouched.
    if "cap" in signature(mixin.get_video_metadata).parameters:
        return

    original_open = mixin.__dict__["open_video_capture"].__func__
    capture: ContextVar[tuple[bytes, cv2.VideoCapture] | None] = ContextVar("omni_opencv_capture", default=None)

    @wraps(original_open)
    def open_video_capture(cls: type[OpenCVVideoBackendMixin], data: bytes) -> cv2.VideoCapture:
        active = capture.get()
        if cls is mixin and active is not None and active[0] is data:
            return active[1]
        return original_open(cls, data)

    @wraps(original_decode)
    def decode_opencv(
        loader_cls: type[VideoBackend],
        data: bytes,
        target: VideoTargetMetadata,
        sampling_kwargs: dict[str, object],
        *,
        frame_recovery: bool = False,
    ) -> tuple[npt.NDArray[np.uint8], VideoSourceMetadata, list[int], list[int]]:
        # Only this synchronous decode shares the capture. Concurrent calls,
        # nested decodes and standalone metadata reads retain separate owners.
        token = capture.set((data, original_open(mixin, data)))
        try:
            return original_decode(loader_cls, data, target, sampling_kwargs, frame_recovery=frame_recovery)
        finally:
            capture.reset(token)

    setattr(decode_opencv, "_omni_capture_reuse", True)
    mixin.open_video_capture = classmethod(open_video_capture)
    opencv.decode_opencv = decode_opencv
