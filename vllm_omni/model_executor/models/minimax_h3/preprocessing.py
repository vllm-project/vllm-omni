# SPDX-License-Identifier: Apache-2.0
"""Shared MiniMax H3 media normalization and Qwen presentation building.

Builds the positive presentation token stream:
- fl2va: '<Picture 1>: ' label + vision block (<|vision_start|> +
  N*<|image_pad|> + <|vision_end|>) + prompt text.
- t2va: prompt text only (no vision block).
Prompt text passes through verbatim (no stripping or rewriting).

All presentation variants are emitted through the shared ``_Presentation``
accumulator so ids and AdaLN token tags cannot drift apart.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image

from vllm_omni.errors import OmniClientError

VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
VIDEO_PAD = "<|video_pad|>"

_TEXT_TAG = 1
_VIDEO_TAG = 0

MINIMAX_H3_OUTPUT_SHORT_EDGE = 768
MINIMAX_H3_OUTPUT_MAX_PIXELS = 768 * 1344
MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = 2048
MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE = 32
MINIMAX_H3_SUPPORTED_ASPECT_RATIOS = {
    "21:9": 21.0 / 9.0,
    "16:9": 16.0 / 9.0,
    "4:3": 4.0 / 3.0,
    "1:1": 1.0,
    "3:4": 3.0 / 4.0,
    "9:16": 9.0 / 16.0,
}
MINIMAX_H3_MAX_REFERENCE_IMAGE_BYTES = 30 * 1024 * 1024
MINIMAX_H3_REFERENCE_IMAGE_FORMATS = frozenset({"jpeg", "png", "webp", "heic", "heif"})


def _align_multiple(value: float, multiple: int = 32) -> int:
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def load_minimax_h3_images(value: Any) -> list[Image.Image]:
    """Normalize one or more H3 image inputs to RGB PIL images."""
    if isinstance(value, (list, tuple)):
        if not value:
            raise OmniClientError("MiniMax H3 image input must not be empty")
        images: list[Image.Image] = []
        for item in value:
            loaded = load_minimax_h3_images(item)
            if len(loaded) != 1:
                raise OmniClientError(f"MiniMax H3 expected one image, got {len(loaded)}")
            images.extend(loaded)
        return images
    if isinstance(value, (str, os.PathLike)):
        if os.path.getsize(value) > MINIMAX_H3_MAX_REFERENCE_IMAGE_BYTES:
            raise OmniClientError("MiniMax H3 reference image exceeds the 30 MiB size limit")
        with Image.open(value) as image:
            image_format = str(image.format or "").lower()
            if image_format and image_format not in MINIMAX_H3_REFERENCE_IMAGE_FORMATS:
                raise OmniClientError(
                    f"MiniMax H3 reference image must use JPG, JPEG, PNG, WEBP, HEIC, or HEIF, got {image.format}"
                )
            return [image.convert("RGB")]
    if isinstance(value, Image.Image):
        return [value.convert("RGB")]
    if isinstance(value, torch.Tensor):
        tensor = value.detach().float().cpu()
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise OmniClientError(f"image tensor must be [C,H,W], got {tuple(tensor.shape)}")
        if tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        array = tensor.numpy()
        if array.max(initial=0) <= 1.0:
            array = array * 255.0
        return [Image.fromarray(array.clip(0, 255).astype(np.uint8)).convert("RGB")]
    raise OmniClientError(f"unsupported MiniMax H3 image input {type(value)!r}")


def resolve_minimax_h3_aspect_ratio(
    task: str,
    value: Any,
    image: Image.Image | None,
) -> float:
    """Resolve H3's task-specific output aspect-ratio policy."""
    if task == "fl2va":
        if image is None:
            raise OmniClientError("fl2va requires an input image to resolve its aspect ratio")
        return float(image.width) / float(image.height)

    if value is None:
        if task == "t2va":
            raise OmniClientError("t2va requires an explicit aspect_ratio")
        return MINIMAX_H3_SUPPORTED_ASPECT_RATIOS["16:9"]

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"adaptive", "auto"}:
            if task == "t2va":
                raise OmniClientError("t2va requires an explicit named aspect_ratio, not adaptive")
            return MINIMAX_H3_SUPPORTED_ASPECT_RATIOS["16:9"]
        if normalized in MINIMAX_H3_SUPPORTED_ASPECT_RATIOS:
            return MINIMAX_H3_SUPPORTED_ASPECT_RATIOS[normalized]
        try:
            numeric_value = float(normalized)
        except (TypeError, ValueError) as exc:
            supported = ", ".join(MINIMAX_H3_SUPPORTED_ASPECT_RATIOS)
            raise OmniClientError(f"MiniMax H3 aspect_ratio must be one of {supported}, got {value!r}") from exc
    elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        numeric_value = float(value)
    else:
        raise OmniClientError(f"MiniMax H3 aspect_ratio must be a string ratio, got {value!r}")

    if not math.isfinite(numeric_value) or not any(
        math.isclose(numeric_value, ratio, rel_tol=0.0, abs_tol=1e-6)
        for ratio in MINIMAX_H3_SUPPORTED_ASPECT_RATIOS.values()
    ):
        supported = ", ".join(MINIMAX_H3_SUPPORTED_ASPECT_RATIOS)
        raise OmniClientError(f"MiniMax H3 aspect_ratio must be one of {supported}, got {value!r}")
    return numeric_value


def resolve_minimax_h3_reference_image_shape(image: Image.Image) -> tuple[int, int]:
    """Resize an H3 reference image to the official 2048-short-edge canvas."""
    width, height = image.size
    ratio = width / height
    if not 0.4 <= ratio <= 2.5:
        raise OmniClientError(f"reference image aspect ratio must be in [0.4, 2.5], got {width}x{height}")
    if min(width, height) < 256 or max(width, height) > 5760:
        raise OmniClientError(f"reference image dimensions must be in [256, 5760] pixels, got {width}x{height}")
    scale = MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE / min(width, height)
    return (
        _align_multiple(width * scale, MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE),
        _align_multiple(height * scale, MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE),
    )


def resolve_minimax_h3_output_canvas(aspect_ratio: float, short_edge: int) -> tuple[int, int]:
    """Resolve the official H3 ratio/area policy to a 32-pixel canvas."""
    if not math.isfinite(float(aspect_ratio)) or float(aspect_ratio) <= 0:
        raise OmniClientError(f"MiniMax H3 canvas aspect ratio must be positive, got {aspect_ratio!r}")
    if short_edge != MINIMAX_H3_OUTPUT_SHORT_EDGE:
        raise OmniClientError(f"MiniMax H3 target.short_edge must be {MINIMAX_H3_OUTPUT_SHORT_EDGE}, got {short_edge}")
    if aspect_ratio >= 1.0:
        width = float(short_edge) * aspect_ratio
        height = float(short_edge)
    else:
        width = float(short_edge)
        height = float(short_edge) / aspect_ratio
    area = width * height
    if area > MINIMAX_H3_OUTPUT_MAX_PIXELS:
        scale = (MINIMAX_H3_OUTPUT_MAX_PIXELS / area) ** 0.5
        width *= scale
        height *= scale
    return _align_multiple(height, 32), _align_multiple(width, 32)


def _text_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _vision_block_ids(tokenizer: Any, pad_token: str, count: int) -> list[int]:
    return (
        [tokenizer.convert_tokens_to_ids(VISION_START)]
        + [tokenizer.convert_tokens_to_ids(pad_token)] * int(count)
        + [tokenizer.convert_tokens_to_ids(VISION_END)]
    )


class _Presentation:
    """Accumulates aligned (ids, token_tags) presentation segments."""

    def __init__(self) -> None:
        self.ids: list[int] = []
        self.tags: list[int] = []

    def text(self, token_ids: list[int]) -> None:
        self.ids += token_ids
        self.tags += [_TEXT_TAG] * len(token_ids)

    def vision(self, token_ids: list[int]) -> None:
        self.ids += token_ids
        self.tags += [_VIDEO_TAG] * len(token_ids)

    def build(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.ids, dtype=torch.long),
            torch.tensor(self.tags, dtype=torch.long),
        )


def _timestamped_video_blocks(
    presentation: _Presentation,
    tokenizer: Any,
    *,
    counts: Sequence[int],
    timestamps: Sequence[float],
    context: str,
) -> None:
    """Emit per-temporal-block ``<{t:.1f} seconds>`` text + VIDEO vision."""

    counts = [int(value) for value in counts]
    timestamps = [float(value) for value in timestamps]
    if not counts or len(counts) != len(timestamps):
        raise ValueError(f"{context}video block token counts and timestamps must align")
    for count, timestamp in zip(counts, timestamps):
        if count <= 0:
            raise ValueError(f"{context}video block token count must be positive")
        presentation.text(_text_ids(tokenizer, f"<{timestamp:.1f} seconds>"))
        presentation.vision(_vision_block_ids(tokenizer, VIDEO_PAD, count))


def minimax_h3_text_only_ids(tokenizer: Any, prompt: str) -> torch.Tensor:
    """t2va presentation: verbatim prompt, no special tokens."""
    if not prompt:
        raise ValueError("prompt must be non-empty")
    return torch.tensor(_text_ids(tokenizer, prompt), dtype=torch.long)


def _multi_image_presentation(
    tokenizer: Any,
    *,
    prompt: str,
    image_token_counts: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not image_token_counts:
        raise ValueError("image_token_counts must be non-empty")
    presentation = _Presentation()
    for index, count in enumerate(image_token_counts, start=1):
        if int(count) <= 0:
            raise ValueError("image_token_count must be positive")
        presentation.text(_text_ids(tokenizer, f"<Picture {index}>: "))
        presentation.vision(_vision_block_ids(tokenizer, IMAGE_PAD, count))
    presentation.text(_text_ids(tokenizer, prompt))
    return presentation.build()


def minimax_h3_multi_image_presentation(
    tokenizer: Any,
    *,
    prompt: str,
    image_token_counts: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return aligned FL2VA presentation token IDs and role IDs."""
    return _multi_image_presentation(
        tokenizer,
        prompt=prompt,
        image_token_counts=image_token_counts,
    )


def minimax_h3_ref2va_presentation(
    tokenizer: Any,
    *,
    prompt: str,
    condition_labels: list[tuple[str, int]],
    image_token_count: int | list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ref2va positive presentation:

    per condition in request order — image i: ``<Picture i>: `` label followed
    by the vision block; audio j: ``<Audio j>: `` label only (audio content
    never enters Qwen) — then the verbatim prompt. Returns (ids, token_tags)
    with the vision block tagged VIDEO(0) and everything else TEXT(1).

    condition_labels: [("image", 1), ("audio", 1), ...] with 1-based ordinals
    per type.
    """
    return minimax_h3_ref2va_video_presentation(
        tokenizer,
        prompt=prompt,
        condition_labels=condition_labels,
        image_token_count=image_token_count,
        video_block_token_counts=None,
        video_block_timestamps=None,
    )


def _as_int_list(value: int | Sequence[int] | None, *, name: str) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an int or a sequence of ints")
    return [int(item) for item in value]


def _as_nested_int_list(
    value: Sequence[int] | Sequence[Sequence[int]] | None,
    *,
    name: str,
) -> list[list[int]]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    if len(value) == 0:
        return []
    first = value[0]
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        out: list[list[int]] = []
        for group in value:
            if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
                raise ValueError(f"{name} must not mix nested and flat entries")
            out.append([int(item) for item in group])
        return out
    return [[int(item) for item in value]]


def _as_nested_float_list(
    value: Sequence[float] | Sequence[Sequence[float]] | None,
    *,
    name: str,
) -> list[list[float]]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    if len(value) == 0:
        return []
    first = value[0]
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        out: list[list[float]] = []
        for group in value:
            if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
                raise ValueError(f"{name} must not mix nested and flat entries")
            out.append([float(item) for item in group])
        return out
    return [[float(item) for item in value]]


def minimax_h3_ref2va_video_presentation(
    tokenizer: Any,
    *,
    prompt: str,
    condition_labels: list[tuple[str, int]],
    image_token_count: int | list[int] | None,
    video_block_token_counts: list[int] | list[list[int]] | None,
    video_block_timestamps: list[float] | list[list[float]] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ref2va (optionally with video refs) positive presentation:

    per condition in request order —
    - image i:  ``<Picture i>: `` label + one image vision block;
    - audio j:  ``<Audio j>: `` label only (audio content never enters Qwen);
    - video k:  ``<Video k>: `` label, then per temporal block a timestamp
      text ``<{t:.1f} seconds>`` followed by a VIDEO vision block
      (<|vision_start|> + <|video_pad|> x n + <|vision_end|>). Timestamps are
      the mean of each merged frame pair (Qwen3VL temporal merge 2; odd frame
      counts repeat the last frame), emitting the
      ``<0.2 seconds>`` ..
      ``<4.0 seconds>`` sequence — note Python bankers-rounding at .1f.
    then the verbatim prompt. Vision blocks are tagged VIDEO(0), everything
    else TEXT(1).
    """
    if not prompt:
        raise ValueError("prompt must be non-empty")
    presentation = _Presentation()
    image_token_counts = _as_int_list(image_token_count, name="image_token_count")
    video_counts_by_ref = _as_nested_int_list(
        video_block_token_counts,
        name="video_block_token_counts",
    )
    video_timestamps_by_ref = _as_nested_float_list(
        video_block_timestamps,
        name="video_block_timestamps",
    )
    if len(video_counts_by_ref) != len(video_timestamps_by_ref):
        raise ValueError("video block token counts and timestamps must align")
    image_seen = 0
    video_seen = 0
    for cond_type, ordinal in condition_labels:
        if cond_type == "image":
            image_seen += 1
            if image_seen > len(image_token_counts):
                raise ValueError("image_token_count required for an image reference")
            count = int(image_token_counts[image_seen - 1])
            if count <= 0:
                raise ValueError("image_token_count required for an image reference")
            presentation.text(_text_ids(tokenizer, f"<Picture {ordinal}>: "))
            presentation.vision(_vision_block_ids(tokenizer, IMAGE_PAD, count))
        elif cond_type == "audio":
            presentation.text(_text_ids(tokenizer, f"<Audio {ordinal}>: "))
        elif cond_type == "video":
            video_seen += 1
            if video_seen > len(video_counts_by_ref):
                raise ValueError("video reference requires block token counts and timestamps")
            counts = video_counts_by_ref[video_seen - 1]
            timestamps = video_timestamps_by_ref[video_seen - 1]
            if not counts or not timestamps:
                raise ValueError("video reference requires block token counts and timestamps")
            presentation.text(_text_ids(tokenizer, f"<Video {ordinal}>: "))
            _timestamped_video_blocks(
                presentation,
                tokenizer,
                counts=counts,
                timestamps=timestamps,
                context="",
            )
        else:
            raise ValueError(f"unsupported ref2va condition type {cond_type!r}")
    if image_seen != len(image_token_counts):
        raise ValueError("unused image_token_count entries")
    if video_seen != len(video_counts_by_ref):
        raise ValueError("unused video block token count entries")
    presentation.text(_text_ids(tokenizer, prompt))
    return presentation.build()


def build_minimax_h3_presentation(
    tokenizer: Any,
    *,
    prompt: str,
    task: str,
    condition_labels: list[tuple[str, int]],
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
    video_timestamps: Sequence[Sequence[float]] | None,
    merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the token IDs and role IDs shared by fused and split H3."""
    if task == "t2va":
        ids = minimax_h3_text_only_ids(tokenizer, prompt)
        return ids, torch.ones_like(ids)

    merge_length = int(merge_size) ** 2
    image_counts = (
        [int(grid.prod().item()) // merge_length for grid in image_grid_thw] if image_grid_thw is not None else []
    )
    if task == "fl2va":
        return minimax_h3_multi_image_presentation(
            tokenizer,
            prompt=prompt,
            image_token_counts=image_counts,
        )

    video_counts: list[list[int]] = []
    if video_grid_thw is not None:
        for grid in video_grid_thw:
            block_count = int(grid[0].item())
            tokens_per_block = int(grid[1:].prod().item()) // merge_length
            video_counts.append([tokens_per_block] * block_count)
    timestamps = (
        [[float(value) for value in group] for group in video_timestamps] if video_timestamps is not None else []
    )
    if video_counts:
        return minimax_h3_ref2va_video_presentation(
            tokenizer,
            prompt=prompt,
            condition_labels=condition_labels,
            image_token_count=image_counts or None,
            video_block_token_counts=video_counts,
            video_block_timestamps=timestamps,
        )
    return minimax_h3_ref2va_presentation(
        tokenizer,
        prompt=prompt,
        condition_labels=condition_labels,
        image_token_count=image_counts or None,
    )


__all__ = [
    "IMAGE_PAD",
    "MINIMAX_H3_OUTPUT_SHORT_EDGE",
    "VIDEO_PAD",
    "VISION_END",
    "VISION_START",
    "build_minimax_h3_presentation",
    "load_minimax_h3_images",
    "minimax_h3_multi_image_presentation",
    "minimax_h3_ref2va_presentation",
    "minimax_h3_ref2va_video_presentation",
    "minimax_h3_text_only_ids",
    "resolve_minimax_h3_aspect_ratio",
    "resolve_minimax_h3_output_canvas",
    "resolve_minimax_h3_reference_image_shape",
]
