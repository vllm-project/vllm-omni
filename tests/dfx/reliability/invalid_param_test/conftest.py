# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared env for ``http_invalid`` (real :func:`omni_server`)."""

from __future__ import annotations

import io
import os

import pytest
from PIL import Image

# Match ``tests/e2e/online_serving/*`` module-level env for subprocess serve.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_TEST_CLEAN_GPU_MEMORY", "0")


@pytest.fixture
def tiny_png_bytes() -> bytes:
    img = Image.new("RGB", (32, 32), color="gray")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
