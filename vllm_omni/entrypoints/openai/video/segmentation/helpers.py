# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Helpers for ``/v1/videos/segmentations``: validate the pipeline payload and build the response."""

from __future__ import annotations

import io
import time
from fractions import Fraction
from http import HTTPStatus
from typing import Any

import av
import numpy as np
import pybase64 as base64
from fastapi import HTTPException, Request

from vllm_omni.entrypoints.openai.app_state import _get_diffusion_od_config
from vllm_omni.entrypoints.openai.protocol.images import SegmentationRLE
from vllm_omni.entrypoints.openai.protocol.videos import (
    VideoSegmentationFrame,
    VideoSegmentationInstance,
    VideoSegmentationObject,
    VideoSegmentationResponse,
)

# Uploads are streamed to request-scoped local storage that the worker decodes.
VIDEO_SEGMENTATION_MAX_BYTES = 512 * 1024 * 1024
_DEFAULT_MASK_VIDEO_FPS = 24


def supports_video_segmentation(raw_request: Request, engine_client: Any) -> bool:
    """Whether the served diffusion model declares ``/v1/videos/segmentations`` support."""
    from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata

    od_config = _get_diffusion_od_config(raw_request, engine_client)
    model_class_name = getattr(od_config, "model_class_name", None)
    return (
        isinstance(model_class_name, str) and get_diffusion_model_metadata(model_class_name).supports_video_segmentation
    )


def extract_video_segmentation(result: Any) -> dict[str, Any]:
    """The pipeline's ``video_segmentation`` payload, or a 500 when the model returned something else."""
    multimodal_output = getattr(result, "multimodal_output", None)
    payload = multimodal_output.get("video_segmentation") if isinstance(multimodal_output, dict) else None
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail="The served model returned no video segmentation output.",
        )
    return payload


def _decode_rle(counts: list[int], height: int, width: int) -> np.ndarray:
    values = np.arange(len(counts)) % 2 == 1  # runs alternate background / foreground
    flat = np.repeat(values, counts)
    return flat.reshape((height, width), order="F")


def encode_mask_video(frames: list[dict[str, Any]], obj_id: int, width: int, height: int, fps: float | None) -> bytes:
    """One H.264 MP4 of ``obj_id``'s mask on every frame: white inside, black elsewhere or when absent."""
    # yuv420p needs even dimensions; an odd edge gets one black row/column.
    out_w, out_h = width + width % 2, height + height % 2
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        rate = Fraction(fps).limit_denominator(1001) if fps else Fraction(_DEFAULT_MASK_VIDEO_FPS)
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height, stream.pix_fmt = out_w, out_h, "yuv420p"
        canvas = np.zeros((out_h, out_w), dtype=np.uint8)
        for frame in frames:
            canvas[:] = 0
            if obj_id in frame["object_ids"]:
                counts = frame["rle"][frame["object_ids"].index(obj_id)]
                canvas[:height, :width] = _decode_rle(counts, height, width) * np.uint8(255)
            for packet in stream.encode(av.VideoFrame.from_ndarray(canvas, format="gray")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def build_video_segmentation_response(
    payload: dict[str, Any], model_name: str | None, include_mask_videos: bool
) -> VideoSegmentationResponse:
    width, height = int(payload["width"]), int(payload["height"])
    frames = payload["frames"]
    fps = payload.get("fps")
    objects = []
    for obj_id, prompt in sorted(payload["objects"].items()):
        mask_video = None
        if include_mask_videos:
            mask_video = base64.b64encode(encode_mask_video(frames, obj_id, width, height, fps)).decode("ascii")
        objects.append(VideoSegmentationObject(id=obj_id, prompt=prompt, mask_video=mask_video))
    return VideoSegmentationResponse(
        created=int(time.time()),
        model=model_name,
        width=width,
        height=height,
        num_frames=int(payload["num_frames"]),
        fps=fps,
        objects=objects,
        frames=[
            VideoSegmentationFrame(
                frame_index=index,
                instances=[
                    VideoSegmentationInstance(
                        id=obj_id, score=score, box=box, mask=SegmentationRLE(size=[height, width], counts=counts)
                    )
                    for obj_id, score, box, counts in zip(
                        frame["object_ids"], frame["scores"], frame["boxes"], frame["rle"]
                    )
                ],
            )
            for index, frame in enumerate(frames)
        ],
    )


def render_video_segmentation_response(
    payload: dict[str, Any], model_name: str | None, include_mask_videos: bool
) -> bytes:
    """The response as JSON bytes, so a long video's per-frame RLE never serializes on the event loop."""
    return build_video_segmentation_response(payload, model_name, include_mask_videos).model_dump_json().encode()
