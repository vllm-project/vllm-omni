# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EVS (Efficient Video Sampling) frame pre-filter for streaming video input.

Lightweight pixel-level similarity filter that runs before frames reach the
vision encoder.  For static or slow-moving scenes this can reduce the number
of frames by 2-5x, proportionally cutting encoder compute and KV-cache usage.

Usage:
    filter = FrameSimilarityFilter(threshold=0.95)
    for jpeg_bytes in incoming_frames:
        if filter.should_retain(jpeg_bytes):
            buffer.append(jpeg_bytes)
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

_DEFAULT_THRESHOLD = 0.95
_DEFAULT_THUMBNAIL_SIZE = 64


class FrameSimilarityFilter:
    """Drop near-duplicate frames based on pixel-level similarity.

    Each incoming JPEG frame is down-scaled to a small thumbnail and compared
    against the last *retained* frame.  If the normalised similarity exceeds
    ``threshold`` the frame is considered redundant and dropped.

    Args:
        threshold: Similarity score in [0, 1] above which a frame is dropped.
            Higher values keep more frames (less aggressive filtering).
            Default 0.95 works well for typical webcam / surveillance feeds.
        thumbnail_size: Edge length of the square thumbnail used for
            comparison.  Larger values are more accurate but slower.
    """

    def __init__(
        self,
        threshold: float = _DEFAULT_THRESHOLD,
        thumbnail_size: int = _DEFAULT_THUMBNAIL_SIZE,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        if thumbnail_size < 1:
            raise ValueError(f"thumbnail_size must be >= 1, got {thumbnail_size}")

        self._threshold = threshold
        self._thumbnail_size = thumbnail_size
        self._last_retained: np.ndarray | None = None
        self._retained_count = 0
        self._dropped_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_retain(self, frame_jpeg: bytes) -> bool:
        """Return ``True`` if *frame_jpeg* is sufficiently different from the
        last retained frame and should be kept in the buffer."""
        current = self._decode_and_resize(frame_jpeg)

        if self._last_retained is None:
            self._last_retained = current
            self._retained_count += 1
            return True

        similarity = self._compute_similarity(self._last_retained, current)
        if similarity >= self._threshold:
            self._dropped_count += 1
            return False

        self._last_retained = current
        self._retained_count += 1
        return True

    def reset(self) -> None:
        """Clear internal state so the next frame is always retained."""
        self._last_retained = None
        self._retained_count = 0
        self._dropped_count = 0

    @property
    def stats(self) -> dict[str, int | float]:
        """Return filtering statistics."""
        total = self._retained_count + self._dropped_count
        return {
            "retained_count": self._retained_count,
            "dropped_count": self._dropped_count,
            "total_count": total,
            "drop_rate": self._dropped_count / total if total > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Normalised pixel similarity in [0, 1].  1 = identical."""
        mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
        return 1.0 - mse / (255.0 * 255.0)

    def _decode_and_resize(self, jpeg_bytes: bytes) -> np.ndarray:
        """Decode JPEG and resize to a small square thumbnail for fast
        comparison."""
        img = Image.open(io.BytesIO(jpeg_bytes))
        img = img.resize(
            (self._thumbnail_size, self._thumbnail_size),
            Image.Resampling.BILINEAR,
        ).convert("RGB")
        return np.asarray(img, dtype=np.uint8)
