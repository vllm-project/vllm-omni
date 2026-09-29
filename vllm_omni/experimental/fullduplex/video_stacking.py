# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Intra-unit frame stacking for the omni duplex video track.

Official MiniCPM-o duplex raises the visual refresh rate without touching the
audio cadence: ``get_video_frame_audio_segments(stack_frames=N)`` samples the
``N-1`` sub-frames captured inside each second (skipping the one that would
duplicate the second's base frame) and tiles them into a *single* composite
image. The duplex loop then feeds ``frame_list=[base, composite]`` alongside
that second's audio, so a unit still carries one second of audio and at most
two images however large ``N`` is.

Tiling therefore belongs to media preprocessing, not to the wire: a caller
turns its sub-frames into one composite here and sends it as the second frame
of the append that closes the unit.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Any

import pybase64 as base64

_BG_COLOR = (255, 255, 255)
_LINE_COLOR = (0, 0, 0)
_LINE_WIDTH = 6


def _as_image(frame: Any) -> Any:
    """Coerce a PIL image, encoded bytes, or base64/data-URL string to PIL RGB."""
    from PIL import Image

    if isinstance(frame, Image.Image):
        return frame
    if isinstance(frame, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(frame))).convert("RGB")
    if isinstance(frame, str):
        payload = frame.split(",")[-1] if ";base64," in frame else frame
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    raise TypeError(f"unsupported frame type for stacking: {type(frame)}")


def _grid_shape(images: Sequence[Any], *, cell_w: int, cell_h: int) -> tuple[int, int]:
    """Rows and columns for ``images``, matching official ``concat_images``.

    4 frames tile 2x2; 2 and 3 frames pick the row/column arrangement whose
    canvas is closest to square, so a landscape clip stacks vertically and a
    portrait one horizontally; anything else falls back to a single row.
    """
    count = len(images)
    if count == 4:
        return 2, 2
    if count == 1:
        return 1, 1
    if count not in (2, 3):
        return 1, count

    candidates = [(1, count), (count, 1)]

    def squareness(rows: int, cols: int) -> float:
        width = cols * cell_w + (cols - 1) * _LINE_WIDTH
        height = rows * cell_h + (rows - 1) * _LINE_WIDTH
        return abs(width / max(1, height) - 1.0)

    ratios = [squareness(rows, cols) for rows, cols in candidates]
    if count == 2 and ratios[0] == ratios[1]:
        average_aspect = sum(im.width / max(1, im.height) for im in images) / count
        return (1, 2) if average_aspect >= 1.0 else (2, 1)
    return candidates[min(range(len(candidates)), key=lambda index: ratios[index])]


def _letterbox(image: Any, target_w: int, target_h: int) -> Any:
    from PIL import Image

    image = image.convert("RGB")
    width, height = image.size
    scale = min(target_w / width, target_h / height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = image.resize(new_size, Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (target_w, target_h), _BG_COLOR)
    canvas.paste(resized, ((target_w - new_size[0]) // 2, (target_h - new_size[1]) // 2))
    return canvas


def concat_frames(frames: Sequence[Any]) -> Any:
    """Tile a unit's sub-frames into one PIL image, as official stacking does.

    Cells are the largest input width and height, each frame is letterboxed to
    keep its aspect ratio, and a separator band is drawn only in the interior
    seams so the model can tell the sub-frames apart.
    """
    from PIL import Image

    images = [_as_image(frame) for frame in frames]
    if not images:
        raise ValueError("cannot stack an empty frame list")

    cell_w = max(image.width for image in images)
    cell_h = max(image.height for image in images)
    rows, cols = _grid_shape(images, cell_w=cell_w, cell_h=cell_h)

    canvas = Image.new(
        "RGB",
        (cols * cell_w + (cols - 1) * _LINE_WIDTH, rows * cell_h + (rows - 1) * _LINE_WIDTH),
        _LINE_COLOR,
    )
    for index, image in enumerate(images[: rows * cols]):
        row, col = divmod(index, cols)
        canvas.paste(
            _letterbox(image, cell_w, cell_h),
            (col * (cell_w + _LINE_WIDTH), row * (cell_h + _LINE_WIDTH)),
        )
    return canvas


def concat_frames_b64(frames: Sequence[Any], *, quality: int = 95) -> str:
    """``concat_frames`` re-encoded as a base64 JPEG, ready for the wire."""
    buf = io.BytesIO()
    concat_frames(frames).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def unit_subframe_offsets(stack_frames: int, *, unit_s: float = 1.0) -> list[float]:
    """Sub-frame capture offsets inside a unit, in seconds from its start.

    ``stack_frames=5`` yields 0.2/0.4/0.6/0.8 s for a 1 s unit. Offset 0 is
    skipped because official skips it: it is the unit's base frame, already
    sent as ``frame_list[0]``.
    """
    count = max(1, int(stack_frames))
    return [index / count * unit_s for index in range(1, count)]
