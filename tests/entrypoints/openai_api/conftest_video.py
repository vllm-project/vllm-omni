# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared test helpers for video streaming tests."""

from __future__ import annotations

import io

import pytest

np = pytest.importorskip("numpy", reason="numpy required for video stream tests")
PIL = pytest.importorskip("PIL", reason="Pillow required for video stream tests")
from PIL import Image  # noqa: E402


def make_jpeg(r: int = 128, g: int = 128, b: int = 128, size: int = 64) -> bytes:
    """Create a solid-colour JPEG image."""
    img = Image.new("RGB", (size, size), (r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def make_gradient_jpeg(seed: int, size: int = 64) -> bytes:
    """Create a random-gradient JPEG that varies based on *seed*."""
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, (size, size, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()
