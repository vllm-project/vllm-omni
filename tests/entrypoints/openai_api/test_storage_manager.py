# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Unit tests for the OpenAI video storage managers.
"""

import asyncio
import os
import time

import pytest

from vllm_omni.entrypoints.openai import storage as storage_module
from vllm_omni.entrypoints.openai.protocol.videos import VideoResponse
from vllm_omni.entrypoints.openai.storage import FileStorageHandle, LocalStorageManager, LocalStorageTTLManager
from vllm_omni.entrypoints.openai.stores import AsyncDictStore

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


@pytest.mark.asyncio
async def test_local_storage_ttl_start_reports_only_successfully_deleted_keys(tmp_path, monkeypatch):
    storage = LocalStorageTTLManager(
        storage_path=str(tmp_path / "storage"),
        ttl_seconds=60,
        sweep_interval_seconds=60,
    )
    expired_keys = {"video-expired-1", "video-expired-2"}
    failed_key, missing_key, recent_key = "video-failed", "video-missing", "video-recent"
    for key in [*expired_keys, failed_key, missing_key, recent_key]:
        await storage.save(b"video-bytes", key)
        if key != recent_key:
            path = storage.get_full_file_path(key)
            expired_mtime = time.time() - 120
            os.utime(path, (expired_mtime, expired_mtime))
    storage_path = tmp_path / "storage"
    (storage_path / "directory").mkdir()
    target = tmp_path / "outside-video"
    target.write_bytes(b"keep-outside-video")
    (storage_path / "symlink").symlink_to(target)
    expired_mtime = time.time() - 120
    for path in [storage_path / "directory", target, storage_path / "symlink"]:
        os.utime(path, (expired_mtime, expired_mtime), follow_symlinks=False)

    remove = storage_module.os.remove

    def remove_with_failures(path):
        if path == storage.get_full_file_path(failed_key):
            raise PermissionError("Cannot delete this artifact")
        remove(path)
        if path == storage.get_full_file_path(missing_key):
            raise FileNotFoundError("Artifact disappeared before deletion")

    monkeypatch.setattr(storage_module.os, "remove", remove_with_failures)
    batches: list[list[str]] = []
    notified, release_callback, swept = asyncio.Event(), asyncio.Event(), asyncio.Event()
    sweep_results: list[int] = []
    sweep_once = storage._sweep_once

    async def observe_sweep(cutoff):
        result = await sweep_once(cutoff)
        sweep_results.append(result)
        swept.set()
        return result

    async def on_expire(keys):
        batches.append(keys)
        notified.set()
        await release_callback.wait()

    monkeypatch.setattr(storage, "_sweep_once", observe_sweep)
    await storage.start(on_expire=on_expire)
    try:
        await asyncio.wait_for(notified.wait(), timeout=5)
        assert len(batches) == 1
        assert set(batches[0]) == expired_keys
        assert not swept.is_set(), "The sweep must await its expiration callback"
        release_callback.set()
        await asyncio.wait_for(swept.wait(), timeout=5)
        assert sweep_results == [2]
        assert await sweep_once(time.time() - 60) == 0
        assert len(batches) == 1
        assert all(not storage.exists(key) for key in expired_keys)
        assert storage.exists(failed_key)
        assert not storage.exists(missing_key)
        assert storage.exists(recent_key)
        assert (storage_path / "directory").is_dir()
        assert (storage_path / "symlink").is_symlink()
        assert target.read_bytes() == b"keep-outside-video"
    finally:
        release_callback.set()
        await storage.stop()


@pytest.mark.asyncio
async def test_local_storage_ttl_start_stop_and_restart(tmp_path):
    storage = LocalStorageTTLManager(
        storage_path=str(tmp_path / "storage"),
        ttl_seconds=1,
        sweep_interval_seconds=60,
    )
    batches: list[tuple[int, list[str]]] = []
    await storage.stop()
    for generation in range(2):
        key = f"video-generation-{generation}"
        await storage.save(b"video-bytes", key)
        expired_mtime = time.time() - 10
        os.utime(storage.get_full_file_path(key), (expired_mtime, expired_mtime))
        notified = asyncio.Event()

        async def on_expire(keys):
            batches.append((generation, keys))
            notified.set()

        await storage.start(on_expire=on_expire)
        task = storage._sweeper_task
        try:
            await storage.start()
            assert storage._sweeper_task is task
            await asyncio.wait_for(notified.wait(), timeout=5)
            assert batches[-1] == (generation, [key])
        finally:
            await storage.stop()
        assert task is not None and task.cancelled()
        assert storage._sweeper_task is None
        await storage.stop()
    assert len(batches) == 2


@pytest.mark.asyncio
async def test_async_dict_store_remove_many_preserves_unrelated_items():
    store: AsyncDictStore[VideoResponse] = AsyncDictStore()
    expired = VideoResponse(model="test-model", prompt="Expired video")
    retained = VideoResponse(model="test-model", prompt="Retained video")
    await store.upsert(expired.id, expired)
    await store.upsert(retained.id, retained)

    await store.remove_many([expired.id, "missing-video", expired.id])

    assert await store.get(expired.id) is None
    assert await store.list_values() == [retained]
