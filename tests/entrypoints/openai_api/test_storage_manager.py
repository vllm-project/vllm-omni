# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the OpenAI video storage managers.
"""

import asyncio
import os
import time

import pytest

from vllm_omni.entrypoints.openai import storage as storage_module
from vllm_omni.entrypoints.openai.storage import FileStorageHandle, LocalStorageManager, LocalStorageTTLManager

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_local_storage_open_returns_file_handle_for_saved_key(tmp_path):
    storage = LocalStorageManager(storage_path=str(tmp_path / "storage"))
    save_context = asyncio.run(storage.save(b"video-bytes", "video-123"))

    handle = asyncio.run(storage.open(save_context.key))

    assert isinstance(handle, FileStorageHandle)
    assert handle.path == os.path.join(storage.storage_path, "video-123")
    with open(handle.path, "rb") as file:
        assert file.read() == b"video-bytes"


def test_local_storage_open_returns_none_for_missing_key(tmp_path):
    storage = LocalStorageManager(storage_path=str(tmp_path / "storage"))

    handle = asyncio.run(storage.open("missing-video"))

    assert handle is None


def test_local_storage_ttl_save_sets_expiration_metadata(tmp_path, monkeypatch):
    storage = LocalStorageTTLManager(
        storage_path=str(tmp_path / "storage"),
        max_concurrency=1,
        ttl_seconds=60,
        sweep_interval_seconds=300,
    )
    monkeypatch.setattr(storage_module.time, "time", lambda: 1_700_000_000)

    save_context = asyncio.run(storage.save(b"video-bytes", "video-ttl"))

    assert save_context.key == "video-ttl"
    assert save_context.created_at == 1_700_000_000
    assert save_context.expires_at == 1_700_000_060


def test_local_storage_ttl_sweeper_removes_expired_file(tmp_path):
    storage = LocalStorageTTLManager(
        storage_path=str(tmp_path / "storage"),
        max_concurrency=1,
        ttl_seconds=1,
        sweep_interval_seconds=60,
    )

    async def setup_file() -> tuple[str, str]:
        save_context = await storage.save(b"video-bytes", "video-expired")
        file_path = storage.get_full_file_path(save_context.key)
        return save_context.key, file_path

    storage_key, file_path = asyncio.run(setup_file())
    expired_mtime = time.time() - 10
    os.utime(file_path, (expired_mtime, expired_mtime))
    assert os.path.exists(file_path)

    deleted = asyncio.run(storage._sweep_once(time.time() - 1))

    assert deleted == 1
    assert not os.path.exists(file_path)
    assert asyncio.run(storage.open(storage_key)) is None


def test_local_storage_ttl_sweeper_keeps_files_when_path_missing(tmp_path):
    storage = LocalStorageTTLManager(
        storage_path=str(tmp_path / "storage"),
        max_concurrency=1,
        ttl_seconds=1,
        sweep_interval_seconds=60,
    )

    missing_path = storage.get_full_file_path("video-missing")
    assert not os.path.exists(missing_path)

    deleted = asyncio.run(storage._sweep_once(time.time() - 1))

    assert deleted == 0
    assert not os.path.exists(missing_path)
