# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OpenAI-compatible protocol definitions for video generation.

This module provides Pydantic models for a video generation endpoint that
mirrors the OpenAI Images API shape, with vllm-omni extensions for diffusion
video models (e.g., Wan2.2).
"""

import mimetypes
import time
import uuid
from enum import Enum
from functools import lru_cache
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from vllm_omni.entrypoints.openai.image_api_utils import parse_size


class VideoGenerationStatus(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


SizeStr = Annotated[str, StringConstraints(pattern=r"^\d+x\d+$")]
SecondStr = Annotated[str, StringConstraints(pattern=r"^[1-9]\d*$")]
DEFAULT_FPS = 24


@lru_cache
def file_extension(media_type: str):
    media_type = str(media_type).split(";", 1)[0].strip().lower()
    ext = mimetypes.guess_extension(media_type, strict=False)

    if ext is None:
        raise ValueError(f"No recognized file extension for media_type {media_type}")

    # Keep naming stable for unknown/unsupported MIME types.
    return ext.lstrip(".")


class VideoParams(BaseModel):
    """Optional block for video-specific parameters."""

    width: int | None = Field(default=None, ge=1, description="Video width in pixels")
    height: int | None = Field(default=None, ge=1, description="Video height in pixels")
    num_frames: int | None = Field(default=None, ge=1, description="Number of frames")
    fps: int | None = Field(default=None, ge=1, description="Frames per second for output video")

    @property
    def size(self) -> str | None:
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        else:
            return None


class FileImageReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_id: str


class UrlImageReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_url: str


ImageReference = UrlImageReference | FileImageReference


class VideoGenerationRequest(BaseModel):
    """
    OpenAI-style video generation request.

    Follows the OpenAI Images API conventions with extensions for video.
    """

    # OpenAI standard fields
    model: str | None = Field(
        default=None,
        description="Model to use (optional, uses server's configured model if omitted)",
    )
    prompt: str = Field(..., description="Text description of the desired video(s)")
    seconds: SecondStr | None = Field(
        default=None,
        description="Clip duration in seconds (OpenAI-compatible, e.g., 4, 8, 12)",
    )
    size: SizeStr | None = Field(
        default=None,
        description="Video dimensions in WIDTHxHEIGHT format (e.g., '1280x720')",
    )

    image_reference: ImageReference | None = Field(
        default=None,
        description="Optional JSON-safe image reference that guides generation. Provide either image_url or file_id.",
    )

    # Video params block for extensibility
    video_params: VideoParams | None = Field(default=None, description="Optional video-specific parameters")

    # User specific tracking field
    user: str | None = Field(default=None, description="User identifier for tracking")

    # Video-specific fields (top-level for OpenAI-style compatibility)
    width: int | None = Field(default=None, ge=1, description="Video width in pixels")
    height: int | None = Field(default=None, ge=1, description="Video height in pixels")
    fps: int | None = Field(default=None, ge=1, description="Frames per second for output video")
    num_frames: int | None = Field(default=None, ge=1, description="Number of frames to generate")

    # vllm-omni extensions for diffusion control
    negative_prompt: str | None = Field(default=None, description="Text describing what to avoid in the video")
    num_inference_steps: int | None = Field(
        default=None,
        ge=1,
        le=200,
        description="Number of diffusion sampling steps (uses model defaults if not specified)",
    )
    guidance_scale: float | None = Field(
        default=None,
        ge=0.0,
        le=20.0,
        description="Classifier-free guidance scale (uses model defaults if not specified)",
    )
    guidance_scale_2: float | None = Field(
        default=None,
        ge=0.0,
        le=20.0,
        description="High-noise CFG scale for video models (Wan2.2)",
    )
    boundary_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Boundary split ratio for low/high DiT (Wan2.2)",
    )
    flow_shift: float | None = Field(
        default=None,
        description="Scheduler flow_shift for video models (Wan2.2)",
    )
    true_cfg_scale: float | None = Field(
        default=None,
        ge=0.0,
        le=20.0,
        description="True CFG scale (model-specific parameter, may be ignored if not supported)",
    )
    seed: int | None = Field(default=None, description="Random seed for reproducibility")

    # vllm-omni extensions for post-generation frame interpolation.
    enable_frame_interpolation: bool = Field(
        default=False,
        description="Enable post-generation RIFE frame interpolation before MP4 encoding.",
    )
    frame_interpolation_exp: int = Field(
        default=1,
        ge=1,
        description="Interpolation exponent: 1=2x temporal resolution, 2=4x, etc.",
    )
    frame_interpolation_scale: float = Field(
        default=1.0,
        gt=0.0,
        description="RIFE inference scale. Use 0.5 for high-resolution inputs to save memory.",
    )
    frame_interpolation_model_path: str | None = Field(
        default=None,
        description=(
            "Local directory or Hugging Face repo ID containing RIFE flownet.pkl weights. "
            "Defaults to elfgum/RIFE-4.22.lite."
        ),
    )

    # vllm-omni extension for per-request LoRA.
    lora: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional LoRA adapter for this request. Expected shape: "
            "{name/path/scale/int_id}. Field names are flexible "
            "(e.g. name|lora_name|adapter, path|lora_path|local_path, "
            "scale|lora_scale, int_id|lora_int_id)."
        ),
    )

    # Generic model-specific parameters
    extra_params: dict[str, Any] | None = Field(
        default=None,
        description=("Optional model-specific parameters passed directly to the model's extra_args. "),
    )

    def resolve_video_params(self) -> VideoParams:
        vp = VideoParams(width=self.width, height=self.height, fps=self.fps, num_frames=self.num_frames)

        if self.video_params is not None:
            vp.width = vp.width or self.video_params.width
            vp.height = vp.height or self.video_params.height
            vp.fps = vp.fps or self.video_params.fps
            vp.num_frames = vp.num_frames or self.video_params.num_frames

        if self.size:
            vp.width, vp.height = parse_size(self.size)

        if vp.fps is None:
            vp.fps = DEFAULT_FPS

        if vp.num_frames is None and self.seconds is not None:
            vp.num_frames = int(self.seconds) * int(vp.fps)

        return vp


class VideoData(BaseModel):
    """Single generated video data."""

    b64_json: str | None = Field(default=None, description="Base64-encoded MP4 video")
    url: str | None = Field(default=None, description="Video URL (not implemented)")
    revised_prompt: str | None = Field(default=None, description="Revised prompt (OpenAI compatibility, always null)")


class VideoGenerationResponse(BaseModel):
    """OpenAI-style video generation response."""

    created: int = Field(..., description="Unix timestamp of when the generation completed")
    data: list[VideoData] = Field(..., description="Array of generated videos")
    stage_durations: dict[str, float] = Field(
        default_factory=dict,
        description="Profiler stage durations reported by the diffusion pipeline.",
    )
    peak_memory_mb: float = Field(
        default=0.0,
        description="Peak device memory usage in MB reported by the diffusion pipeline.",
    )


class VideoError(BaseModel):
    code: int | str = Field(..., description="A machine-readable error code that was returned.")
    message: str = Field(..., description="A human-readable description of the error that was returned.")


class VideoResponse(BaseModel):
    """Stored metadata for an async video generation job."""

    # OpenAI standard fields
    model: str = Field(..., description="Model name used for video generation.")
    prompt: str = Field(..., description="The prompt that was used to generate the video.")
    id: str = Field(
        default_factory=lambda: f"video_gen_{uuid.uuid4().hex}", description="Unique id for a video request"
    )
    object: Literal["video"] = Field(default="video", description="Object type identifier.")
    status: VideoGenerationStatus = Field(
        default=VideoGenerationStatus.QUEUED, description="Current lifecycle status of the video job."
    )
    size: SizeStr | None = Field(default=None, description="Requested output size in WIDTHxHEIGHT format.")
    progress: int = Field(default=0, description="Best-effort progress indicator from 0 to 100.")
    seconds: SecondStr = Field(default="4", description="Requested clip length in seconds.")
    quality: str = Field(default="default", description="Requested quality level for generation.")
    completed_at: int | None = Field(
        default=None, description="Unix timestamp (seconds) for when the job completed, if finished."
    )
    created_at: int = Field(
        default_factory=lambda: int(time.time()),
        description="Unix timestamp (seconds) for when the job was created.",
    )
    remixed_from_video_id: str | None = Field(
        default=None,
        description="Optional source video id for remix/edit flows.",
    )
    error: VideoError | None = Field(
        default=None,
        description="Error payload that explains why generation failed, if applicable.",
    )

    # vLLM specific fields
    media_type: Literal["video/mp4"] = Field(default="video/mp4", description="MIME type of the generated artifact.")

    expires_at: int | None = Field(default=None, description="Unix timestamp when this record is considered expired.")
    file_name: str | None = Field(
        default=None,
        description="Filename of the saved output video files for this job.",
    )
    inference_time_s: float | None = Field(default=None, description="End-to-end inference time in seconds.")
    stage_durations: dict[str, float] = Field(
        default_factory=dict,
        description="Profiler stage durations reported by the diffusion pipeline.",
    )
    peak_memory_mb: float = Field(
        default=0.0,
        description="Peak device memory usage in MB reported by the diffusion pipeline.",
    )

    @property
    def file_extension(self) -> str:
        return file_extension(self.media_type)


class VideoDeleteResponse(BaseModel):
    id: str = Field(description="Identifier of the deleted video.")
    deleted: bool = Field(description="Indicates that the video resource was deleted.")
    object: Literal["video.deleted"] = Field(
        default="video.deleted", description="The object type that signals the deletion response."
    )


class VideoListResponse(BaseModel):
    """Paginated-style wrapper for listing stored video jobs."""

    first_id: str | None = Field(..., description="The ID of the first item in the list.")
    last_id: str | None = Field(..., description="The ID of the last item in the list.")
    has_more: bool = Field(..., description="Whether there are more items available.")
    data: list[VideoResponse] = Field(..., description="Array of video job records.")
    object: Literal["list"] = Field(default="list", description="Object type identifier for list responses.")
