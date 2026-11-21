# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Helpers for building storage backends in tests."""

from __future__ import annotations

import os
from pathlib import Path

import opendal

from vllm_omni.entrypoints.openai.storage import OpenDALBackend


def build_file_storage(
    base_path: str | Path,
    *,
    default_ttl: int | None = None,
    sweep_interval: int | None = None,
    max_concurrency: int = 4,
) -> OpenDALBackend:
    """Filesystem backend rooted at ``base_path/storage`` (real artifacts on disk)."""
    root = os.path.join(str(base_path), "storage")
    os.makedirs(root, exist_ok=True)
    return OpenDALBackend(
        opendal.AsyncOperator("fs", root=root),
        local_root=root,
        max_concurrency=max_concurrency,
        default_ttl=default_ttl,
        sweep_interval=sweep_interval,
    )


def build_memory_storage(
    *,
    default_ttl: int | None = None,
    sweep_interval: int | None = None,
) -> OpenDALBackend:
    """In-memory OpenDAL backend: listable, but without ``last_modified``."""
    return OpenDALBackend(
        opendal.AsyncOperator("memory"),
        default_ttl=default_ttl,
        sweep_interval=sweep_interval,
    )
