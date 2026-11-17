# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen image normalization and visual-memory prompt labels."""

from io import BytesIO

import pybase64
from PIL import Image
from vllm.multimodal.image import open_image

from vllm_omni.engine.duplex.session.visual_history import SessionVisualHistory


def decode_image(url: str) -> Image.Image:
    """Match the existing Qwen input_image normalization, without network I/O."""
    header, separator, encoded = url.partition(",")
    if not separator or not header.startswith("data:image/") or not header.endswith(";base64"):
        raise ValueError("Qwen visual history requires base64 image data URLs")
    with open_image(BytesIO(pybase64.b64decode(encoded, validate=True))) as image:
        image.thumbnail((448, 448))
        return image.convert("RGB").copy()


def visual_summary_text(history: SessionVisualHistory) -> tuple[str, ...]:
    """Label resolved evidence with its current Session message positions."""
    return tuple(
        f"Visual observations from source messages "
        f"{history.inputs[int(item.first_source_id)].message_index + 1}–"
        f"{history.inputs[int(item.last_source_id)].message_index + 1}:\n{item.text}"
        for item in history.summaries()
    )
