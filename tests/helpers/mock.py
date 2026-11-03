# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared pytest mocks for unit tests.

Add reusable ``monkeypatch`` helpers here instead of a new module per target.
Callers import explicitly: ``from tests.helpers.mock import ...``.
"""

from __future__ import annotations

from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any


def patch_hf_snapshot_download(
    monkeypatch: Any,
    fake: Callable[..., Any],
    *,
    hf_home: str | PathLike[str] | None = None,
) -> None:
    """Replace ``snapshot_download`` on the cached ``hf_api()`` singleton.

    Production code calls ``hf_api().snapshot_download``. Patching
    ``huggingface_hub.HfApi.snapshot_download`` misses that instance, so
    weekly CPU jobs talk to the real Hub / ``HF_HOME`` cache.

    ``fake`` is installed on the instance, so it is called without ``self``.

    Pass ``hf_home`` (typically pytest ``tmp_path``) to isolate the Hub cache.
    ``repo_utils`` imports ``huggingface_hub`` before tests can ``setenv``, so
    ``constants.HF_HUB_CACHE`` is already frozen from the process environment
    (weekly: ``/fsx/hf_cache``). Downloads without ``cache_dir`` use that
    constant, not ``HF_HOME``. Patch both the env and the imported constants
    so a missed instance mock cannot read the builder cache.
    """
    from huggingface_hub import constants as hub_constants

    from vllm_omni.transformers_utils import repo_utils

    monkeypatch.setattr(repo_utils.hf_api(), "snapshot_download", fake)
    if hf_home is not None:
        hf_home_str = str(hf_home)
        hub_cache = str(Path(hf_home_str) / "hub")
        monkeypatch.setenv("HF_HOME", hf_home_str)
        monkeypatch.setenv("HF_HUB_CACHE", hub_cache)
        monkeypatch.setattr(hub_constants, "HF_HOME", hf_home_str)
        monkeypatch.setattr(hub_constants, "default_cache_path", hub_cache)
        monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", hub_cache)
        monkeypatch.setattr(hub_constants, "HUGGINGFACE_HUB_CACHE", hub_cache)
