# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Literal

import torch


class VideoTensorLayout(str, Enum):
    BCTHW = "bcthw"
    BTHWC = "bthwc"


class VideoTensorEncoding(str, Enum):
    NORMALIZED_FLOAT = "normalized_float"
    UINT8_FRAMES = "uint8_frames"


class VideoValueRange(str, Enum):
    NEGATIVE_ONE_TO_ONE = "negative_one_to_one"
    ZERO_TO_ONE = "zero_to_one"
    ZERO_TO_255 = "zero_to_255"


class VideoColorModel(str, Enum):
    RGB = "rgb"


class FloatVideoConsumer(str, Enum):
    FRAME_INTERPOLATION = "frame_interpolation"
    VIDEO_GUARDRAILS = "video_guardrails"


@dataclass(frozen=True, slots=True)
class VideoTensorSpec:
    layout: VideoTensorLayout
    encoding: VideoTensorEncoding
    value_range: VideoValueRange
    color_model: VideoColorModel = VideoColorModel.RGB


@dataclass(frozen=True, slots=True)
class VideoTransportConstraints:
    pending_float_consumers: frozenset[FloatVideoConsumer] = frozenset()


@dataclass(frozen=True, slots=True)
class VideoMediaOutput:
    tensor: torch.Tensor
    spec: VideoTensorSpec
    constraints: VideoTransportConstraints = field(default_factory=VideoTransportConstraints)
    schema_version: Literal[1] = 1

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported video media schema version: {self.schema_version}")
        if not isinstance(self.spec, VideoTensorSpec):
            raise TypeError(f"Video media spec must be a VideoTensorSpec, got {type(self.spec).__name__}")
        if not isinstance(self.constraints, VideoTransportConstraints):
            raise TypeError(
                f"Video media constraints must be VideoTransportConstraints, got {type(self.constraints).__name__}"
            )
        if not isinstance(self.constraints.pending_float_consumers, frozenset):
            raise TypeError("pending_float_consumers must be a frozenset")
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError(f"Video media tensor must be a torch.Tensor, got {type(self.tensor).__name__}")
        if self.tensor.ndim != 5:
            raise ValueError(f"Video media tensor must be rank 5, got shape {tuple(self.tensor.shape)}")
        if any(size <= 0 for size in self.tensor.shape):
            raise ValueError(f"Video media dimensions must be positive, got shape {tuple(self.tensor.shape)}")
        if self.spec.color_model is not VideoColorModel.RGB:
            raise ValueError(f"Unsupported video color model: {self.spec.color_model}")
        if any(not isinstance(consumer, FloatVideoConsumer) for consumer in self.constraints.pending_float_consumers):
            raise ValueError("pending_float_consumers must contain only FloatVideoConsumer values")

        if self.spec.encoding is VideoTensorEncoding.NORMALIZED_FLOAT:
            self._validate_normalized_float()
            return
        if self.spec.encoding is VideoTensorEncoding.UINT8_FRAMES:
            self._validate_uint8_frames()
            return
        raise ValueError(f"Unsupported video tensor encoding: {self.spec.encoding}")

    def _validate_normalized_float(self) -> None:
        if self.spec.layout is not VideoTensorLayout.BCTHW:
            raise ValueError("NORMALIZED_FLOAT video must use BCTHW layout")
        if self.tensor.shape[1] != 3:
            raise ValueError(f"RGB BCTHW video must have 3 channels, got {self.tensor.shape[1]}")
        if self.tensor.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(f"NORMALIZED_FLOAT video must use fp16, bf16, or fp32, got {self.tensor.dtype}")
        if self.spec.value_range not in {
            VideoValueRange.NEGATIVE_ONE_TO_ONE,
            VideoValueRange.ZERO_TO_ONE,
        }:
            raise ValueError(f"NORMALIZED_FLOAT video has incompatible value range: {self.spec.value_range}")

    def _validate_uint8_frames(self) -> None:
        if self.spec.layout is not VideoTensorLayout.BTHWC:
            raise ValueError("UINT8_FRAMES video must use BTHWC layout")
        if self.tensor.shape[-1] != 3:
            raise ValueError(f"RGB BTHWC video must have 3 channels, got {self.tensor.shape[-1]}")
        if self.tensor.dtype is not torch.uint8:
            raise ValueError(f"UINT8_FRAMES video must use torch.uint8, got {self.tensor.dtype}")
        if self.spec.value_range is not VideoValueRange.ZERO_TO_255:
            raise ValueError(f"UINT8_FRAMES video has incompatible value range: {self.spec.value_range}")
        if self.constraints.pending_float_consumers:
            raise ValueError("UINT8_FRAMES video cannot have pending float consumers")
        if not self.tensor.is_contiguous():
            raise ValueError("UINT8_FRAMES video tensor must be contiguous")

    def with_tensor(self, tensor: torch.Tensor) -> VideoMediaOutput:
        return replace(self, tensor=tensor)

    def to_cpu(self) -> VideoMediaOutput:
        return self.with_tensor(self.tensor.detach().cpu())


@dataclass(frozen=True, slots=True)
class DiffusionMediaOutput:
    video: VideoMediaOutput
    # Only the model runner may set this before worker output packing.
    prepared_for_transport: bool = False

    def validate(self) -> None:
        if not isinstance(self.video, VideoMediaOutput):
            raise TypeError(f"Diffusion media video must be a VideoMediaOutput, got {type(self.video).__name__}")
        if not isinstance(self.prepared_for_transport, bool):
            raise TypeError("prepared_for_transport must be a bool")
        self.video.validate()
        if not self.prepared_for_transport:
            return
        if not self.video.tensor.is_contiguous():
            raise ValueError("Prepared video media tensor must be contiguous")
        if not _owns_compact_storage(self.video.tensor):
            raise ValueError("Prepared video media tensor must own request-local storage")

    def with_video(self, video: VideoMediaOutput) -> DiffusionMediaOutput:
        return replace(self, video=video)

    def to_cpu(self) -> DiffusionMediaOutput:
        self.validate()
        moved = self.with_video(self.video.to_cpu())
        moved.validate()
        return moved


def _owns_compact_storage(tensor: torch.Tensor) -> bool:
    """True if *tensor* is contiguous and backed by exactly its own elements.

    ``tensor._base is None`` is not sufficient: ``batch[i : i + 1].detach()`` is
    contiguous with ``_base=None`` yet keeps a nonzero storage offset and the
    whole batch storage, so serializing it would leak every other request's
    pixels. Require a zero offset and a storage sized to the logical elements.
    """
    if not tensor.is_contiguous():
        return False
    if tensor.storage_offset() != 0:
        return False
    try:
        storage_bytes = tensor.untyped_storage().nbytes()
    except Exception:
        return False
    return storage_bytes == tensor.numel() * tensor.element_size()


def ensure_request_owned_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if _owns_compact_storage(tensor):
        return tensor
    # A contiguous view may still retain another request's full batch storage.
    return tensor.clone(memory_format=torch.contiguous_format)


def slice_diffusion_media_output(
    media: DiffusionMediaOutput,
    start: int,
    stop: int,
) -> DiffusionMediaOutput:
    media.validate()
    source = media.video.tensor
    tensor = source if start == 0 and stop == source.shape[0] else source[start:stop]
    sliced = replace(media, video=media.video.with_tensor(tensor), prepared_for_transport=False)
    sliced.validate()
    return sliced
