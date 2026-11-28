# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Utilities for model repo interaction."""

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


def cached_repo_file(
    repo_id: str, filename: str, revision: str | None = None, cache_dir: str | None = None
) -> str | None:
    """Local path of ``filename`` from ``repo_id`` in the Hub cache, or None.

    A cache lookup only: it sends no Hub request. ``cache_dir`` is a cache other
    than the default one, such as vLLM's ``--download-dir``. None when
    ``repo_id`` is not a valid repo id or the file isn't cached.
    """
    from huggingface_hub import try_to_load_from_cache

    try:
        path = try_to_load_from_cache(repo_id, filename, cache_dir=cache_dir, revision=revision)
    except ValueError:
        return None
    return path if isinstance(path, str) else None
