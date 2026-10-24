# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Utilities for model repo interaction."""

import os
from pathlib import PurePath

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


def repo_name_from_path(model: str) -> str:
    """Best-effort repository name for a hub repo id or local model path.

    A hub cache snapshot path (``.../models--{org}--{name}/snapshots/{rev}``)
    ends in a commit hash, so name-based checks against a cached path must
    recover the name from the ``models--`` directory instead of the basename.
    """
    for part in PurePath(model).parts:
        if part.startswith("models--"):
            return part.removeprefix("models--").replace("--", "/")
    return os.path.basename(model.rstrip("/"))
