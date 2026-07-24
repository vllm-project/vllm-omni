# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for model repo interaction."""

import huggingface_hub
from huggingface_hub import HfApi

from vllm_omni.version import __version__ as VLLM_OMNI_VERSION

_hf_api: HfApi | None = None


def hf_api() -> HfApi:
    """Return a shared HfApi instance tagged with vLLM-Omni's library info."""
    global _hf_api
    if _hf_api is None:
        _hf_api = HfApi(
            library_name="vllm-omni",
            library_version=VLLM_OMNI_VERSION,
        )
    return _hf_api


def hf_fs() -> "huggingface_hub.HfFileSystem":
    """Return a fresh HfFileSystem tagged with vLLM-Omni's library info."""
    return huggingface_hub.HfFileSystem(
        library_name="vllm-omni",
        library_version=VLLM_OMNI_VERSION,
    )
