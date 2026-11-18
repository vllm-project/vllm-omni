# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Helpers for ``/v1/videos/depth``: validate the pipeline payload and build the response."""

from __future__ import annotations

import io
import time
import zipfile
from fractions import Fraction
from http import HTTPStatus
from typing import Any

import av
import cv2
import numpy as np
import pybase64 as base64
from fastapi import HTTPException, Request

from vllm_omni.entrypoints.openai.app_state import _get_diffusion_od_config
from vllm_omni.entrypoints.openai.protocol.videos import VideoDepthResponse

# Uploads are streamed to request-scoped local storage that the worker decodes.
VIDEO_DEPTH_MAX_BYTES = 512 * 1024 * 1024
VIDEO_DEPTH_COLORMAPS = ("grayscale", "inferno")
_DEFAULT_DEPTH_VIDEO_FPS = 24


def supports_video_depth(raw_request: Request, engine_client: Any) -> bool:
    """Whether the served diffusion model declares ``/v1/videos/depth`` support."""
    from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata

    od_config = _get_diffusion_od_config(raw_request, engine_client)
    model_class_name = getattr(od_config, "model_class_name", None)
    return isinstance(model_class_name, str) and get_diffusion_model_metadata(model_class_name).supports_video_depth


def extract_video_depth(result: Any) -> dict[str, Any]:
    """The pipeline's ``video_depth`` payload, or a 500 when the model returned something else."""
    multimodal_output = getattr(result, "multimodal_output", None)
    payload = multimodal_output.get("video_depth") if isinstance(multimodal_output, dict) else None
    if not isinstance(payload, dict) or not isinstance(payload.get("depth"), np.ndarray):
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="The served model returned no video depth output.",
        )
    return payload


def encode_depth_video(depth: np.ndarray, fps: float | None, colormap: str) -> bytes:
    """One H.264 MP4 of [T, H, W] depth, normalized over the whole video like the reference visualizer."""
    frames, height, width = depth.shape
    low, high = float(depth.min()), float(depth.max())
    scale = 255.0 / (high - low) if high > low else 0.0
    # yuv420p needs even dimensions; an odd edge gets one black row/column.
    out_w, out_h = width + width % 2, height + height % 2
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        rate = Fraction(fps).limit_denominator(1001) if fps else Fraction(_DEFAULT_DEPTH_VIDEO_FPS)
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height, stream.pix_fmt = out_w, out_h, "yuv420p"
        # Frame threading and the veryfast preset encode a 720p clip ~5x faster than the defaults,
        # and the files come out no larger at this crf.
        stream.thread_type = "AUTO"
        stream.options = {"crf": "18", "preset": "veryfast"}
        channels = 1 if colormap == "grayscale" else 3
        canvas = np.zeros((out_h, out_w, channels), dtype=np.uint8)
        for index in range(frames):
            level = ((depth[index].astype(np.float32) - low) * scale).astype(np.uint8)
            if colormap == "inferno":
                level = cv2.cvtColor(cv2.applyColorMap(level, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
            canvas[:height, :width] = level.reshape(height, width, channels)
            frame = av.VideoFrame.from_ndarray(
                canvas[..., 0] if channels == 1 else canvas, format="gray" if channels == 1 else "rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def encode_depth_npz(depth: np.ndarray) -> bytes:
    """An NPZ archive with one float16 ``depth`` entry.

    Float16 depth barely compresses, so the fastest deflate level is used: on a 195-frame 720p
    clip it is about twice as fast as ``np.savez_compressed`` for a 15% larger archive.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        with archive.open("depth.npy", "w", force_zip64=True) as entry:
            np.lib.format.write_array(entry, np.ascontiguousarray(depth, dtype=np.float16))
    return buffer.getvalue()


def build_video_depth_response(
    payload: dict[str, Any], model_name: str | None, colormap: str, include_depth: bool
) -> VideoDepthResponse:
    depth = payload["depth"]
    fps = payload.get("fps")
    return VideoDepthResponse(
        created=int(time.time()),
        model=model_name,
        width=int(payload["width"]),
        height=int(payload["height"]),
        num_frames=int(payload["num_frames"]),
        fps=fps,
        metric=bool(payload["metric"]),
        depth_min=float(depth.min()),
        depth_max=float(depth.max()),
        video=base64.b64encode(encode_depth_video(depth, fps, colormap)).decode("ascii"),
        depth=base64.b64encode(encode_depth_npz(depth)).decode("ascii") if include_depth else None,
    )


def render_video_depth_response(
    payload: dict[str, Any], model_name: str | None, colormap: str, include_depth: bool
) -> bytes:
    """The response as JSON bytes, so encoding a long video never runs on the event loop."""
    return build_video_depth_response(payload, model_name, colormap, include_depth).model_dump_json().encode()
