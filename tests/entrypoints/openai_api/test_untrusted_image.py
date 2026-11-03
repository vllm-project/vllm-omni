# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Regression tests for untrusted image decoding at public ingress paths."""

import asyncio
import base64
from io import BytesIO

import pytest
from fastapi import HTTPException
from PIL import EpsImagePlugin, Image, UnidentifiedImageError

from vllm_omni.entrypoints.openai.errors import InvalidInputReferenceError
from vllm_omni.entrypoints.openai.images.helpers import _load_input_images
from vllm_omni.entrypoints.openai.video.generation.helpers import _validate_minimax_h3_image_payload
from vllm_omni.entrypoints.openai.video_api_utils import _decode_image_bytes

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_EPS_PAYLOAD = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 1 1\n%%EndComments\nshowpage\n%%EOF\n"


def _block_eps_decoder(monkeypatch):
    def fail_if_called(self):
        raise AssertionError("untrusted EPS must not reach Pillow's EPS decoder")

    monkeypatch.setattr(EpsImagePlugin.EpsImageFile, "_open", fail_if_called)


def test_decode_image_bytes_rejects_disallowed_eps(monkeypatch):
    _block_eps_decoder(monkeypatch)

    with pytest.raises(InvalidInputReferenceError, match="not a valid image") as exc_info:
        _decode_image_bytes(_EPS_PAYLOAD, source="image reference")

    assert isinstance(exc_info.value.__cause__, UnidentifiedImageError)


def test_load_input_images_rejects_disallowed_eps_data_url(monkeypatch):
    _block_eps_decoder(monkeypatch)
    data_url = f"data:image/png;base64,{base64.b64encode(_EPS_PAYLOAD).decode('ascii')}"

    with pytest.raises(ValueError, match="Invalid base64 image") as exc_info:
        asyncio.run(_load_input_images([data_url]))

    assert isinstance(exc_info.value.__context__, UnidentifiedImageError)


def test_minimax_h3_upload_validator_rejects_disallowed_eps(monkeypatch):
    _block_eps_decoder(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        _validate_minimax_h3_image_payload(_EPS_PAYLOAD, filename="untrusted.eps")

    assert exc_info.value.status_code == 400
    assert isinstance(exc_info.value.__cause__, UnidentifiedImageError)


def test_minimax_h3_upload_error_lists_supported_reference_formats():
    image_buffer = BytesIO()
    Image.new("RGB", (1, 1)).save(image_buffer, format="TIFF")

    with pytest.raises(HTTPException) as exc_info:
        _validate_minimax_h3_image_payload(image_buffer.getvalue(), filename="reference.tiff")

    assert exc_info.value.detail == "MiniMax H3 reference images must use JPG, JPEG, PNG, or WEBP; got tiff."


def test_minimax_h3_upload_decode_error_lists_supported_formats():
    with pytest.raises(HTTPException) as exc_info:
        _validate_minimax_h3_image_payload(
            b"not an image",
            filename="reference.heic",
        )

    assert exc_info.value.detail == (
        "Invalid uploaded image reference: reference.heic. Expected a JPG, JPEG, PNG, or WEBP image."
    )
