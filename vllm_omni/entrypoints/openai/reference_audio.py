# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reference-audio URI normalization shared by Ming speech endpoints."""

import binascii
from urllib.parse import urlsplit

import pybase64 as base64


def reference_audio_uri(value: str) -> str:
    """Accept a URI or a bare base64-encoded WAV."""
    if urlsplit(value).scheme:
        return value
    try:
        base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("ref_audio must be a URL, data URL, or base64-encoded WAV") from exc
    return f"data:audio/wav;base64,{value}"
