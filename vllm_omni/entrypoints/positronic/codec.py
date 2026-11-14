# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Positronic image markers layered over the existing NumPy/msgpack codec."""

from __future__ import annotations

import io
import time

import numpy as np
from PIL import Image

from vllm_omni.protocol.numpy_msgpack import MAX_PAYLOAD_BYTES
from vllm_omni.protocol.numpy_msgpack import pack as pack
from vllm_omni.protocol.numpy_msgpack import unpack as unpack_numpy

MAX_IMAGE_FRAMES = 25
MAX_IMAGE_BYTES = MAX_PAYLOAD_BYTES


IMAGE_CODECS: tuple[str, ...] = ("jpeg",)


def unpack(
    data: bytes, timing: dict | None = None, cameras: tuple[str, ...] = ("image.exterior", "image.wrist")
) -> dict:
    if len(data) > MAX_PAYLOAD_BYTES:
        raise ValueError("Positronic request payload too large")
    budget = MAX_IMAGE_BYTES
    frames_left = len(cameras) * MAX_IMAGE_FRAMES
    jpeg_seconds = 0.0

    def decode(value: object, image: bool = False) -> object:
        nonlocal budget, frames_left, jpeg_seconds
        if isinstance(value, np.ndarray):
            budget -= value.nbytes
        elif isinstance(value, list):
            return [decode(v, image) for v in value]
        elif isinstance(value, dict):
            if b"__h264__" in value:
                raise ValueError("H.264 decoding is unsupported; use JPEG")
            if b"__jpeg__" in value and not image:
                raise ValueError("Compressed images require a configured camera key")
            if b"__jpeg__" in value:
                ndim = value[b"ndim"]
                buffers = value[b"frames"]
                if type(ndim) is not int or ndim not in (3, 4) or not 1 <= len(buffers) <= MAX_IMAGE_FRAMES:
                    raise ValueError("Invalid JPEG stack")
                if ndim == 3 and len(buffers) != 1:
                    raise ValueError("A single JPEG image must contain exactly one frame")
                frames_left -= len(buffers)
                if frames_left < 0:
                    raise ValueError("Too many camera frames")
                started = time.monotonic()
                frames = []
                for buf in buffers:
                    with Image.open(io.BytesIO(buf)) as im:
                        if im.format != "JPEG" or im.width * im.height * 3 > budget:
                            raise ValueError("Invalid or oversized JPEG image")
                        frame = np.asarray(im.convert("RGB"))
                        budget -= frame.nbytes
                        frames.append(frame)
                value = np.stack(frames) if ndim == 4 else frames[0]
                jpeg_seconds += time.monotonic() - started
            else:
                return {k: decode(v) for k, v in value.items()}
        if budget < 0:
            raise ValueError("Decoded image budget exceeded")
        return value

    result = unpack_numpy(data)
    if not isinstance(result, dict) or any(not isinstance(k, str) for k in result):
        raise ValueError("An observation must be a msgpack map with string keys")
    result = {k: decode(v, k in cameras) for k, v in result.items() if not k.startswith("image.") or k in cameras}
    if timing is not None:
        timing.update(
            jpeg_decode_ms=round(jpeg_seconds * 1000, 1),
            video_decode_ms=0.0,
            image_codec="jpeg" if jpeg_seconds else "raw",
            wire_jpeg_quality_est=None,
        )
    return result
