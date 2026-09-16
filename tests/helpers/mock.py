# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared pytest mocks for unit tests.

Add reusable ``monkeypatch`` helpers here instead of a new module per target.
Callers import explicitly: ``from tests.helpers.mock import ...``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def patch_hf_snapshot_download(monkeypatch: Any, fake: Callable[..., Any]) -> None:
    """Replace ``snapshot_download`` on the cached ``hf_api()`` singleton.

    Production code calls ``hf_api().snapshot_download``. Patching
    ``huggingface_hub.HfApi.snapshot_download`` misses that instance, so
    weekly CPU jobs talk to the real Hub / ``HF_HOME`` cache.

    ``fake`` is installed on the instance, so it is called without ``self``.
    """
    from vllm_omni.transformers_utils import repo_utils

    monkeypatch.setattr(repo_utils.hf_api(), "snapshot_download", fake)
