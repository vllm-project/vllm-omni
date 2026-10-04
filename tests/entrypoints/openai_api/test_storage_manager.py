# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Unit tests for the OpenAI video storage managers.
"""

import asyncio
import os
import threading
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


@pytest.mark.parametrize("operation", ["save", "delete"])
@pytest.mark.parametrize("fail_first", [False, True], ids=["success", "late-error"])
def test_cancelled_storage_io_retains_concurrency_slot(tmp_path, monkeypatch, operation, fail_first):
    storage = LocalStorageManager(storage_path=str(tmp_path / "storage"), max_concurrency=1)
    original = getattr(storage, f"_{operation}_sync")
    release_first = threading.Event()

    async def scenario():
        loop = asyncio.get_running_loop()
        first_started = asyncio.Event()
        second_started = asyncio.Event()
        finished = asyncio.Event()
        unhandled_errors = []
        loop.set_exception_handler(lambda _loop, context: unhandled_errors.append(context))
        first_path = storage.get_full_file_path("first")
        second_path = storage.get_full_file_path("second")
        if operation == "delete":
            for path in (first_path, second_path):
                with open(path, "wb") as output:
                    output.write(b"data")

        def controlled_io(*args):
            name = args[-1]
            loop.call_soon_threadsafe(first_started.set if name == "first" else second_started.set)
            if name == "first":
                assert release_first.wait(timeout=5), "test did not release the first I/O operation"
            try:
                if name == "first" and fail_first:
                    raise OSError("storage operation failed after cancellation")
                return original(*args)
            finally:
                if name == "first":
                    loop.call_soon_threadsafe(finished.set)

        monkeypatch.setattr(storage, f"_{operation}_sync", controlled_io)

        async def run(name):
            if operation == "save":
                return await storage.save(b"data", name)
            return await storage.delete(name)

        first = asyncio.create_task(run("first"))
        second = None
        try:
            await asyncio.wait_for(first_started.wait(), timeout=5)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            second = asyncio.create_task(run("second"))
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(second_started.wait(), timeout=0.1)
            release_first.set()
            await asyncio.wait_for(finished.wait(), timeout=5)
            await asyncio.wait_for(second, timeout=5)
            assert second_started.is_set()
            assert os.path.exists(first_path) == ((operation == "save") != fail_first)
            assert os.path.exists(second_path) == (operation == "save")
            assert not unhandled_errors
        finally:
            release_first.set()
            await asyncio.wait_for(finished.wait(), timeout=5)
            if second is not None:
                await asyncio.wait_for(second, timeout=5)

    asyncio.run(scenario())


def test_cancelled_ttl_sweep_retains_concurrency_slot(tmp_path, monkeypatch):
    storage = LocalStorageTTLManager(
        storage_path=str(tmp_path / "storage"),
        max_concurrency=1,
        ttl_seconds=1,
        sweep_interval_seconds=60,
    )
    expired_path = storage.get_full_file_path("expired")
    with open(expired_path, "wb") as output:
        output.write(b"old")
    os.utime(expired_path, (0, 0))
    release_delete = threading.Event()
    original_remove = storage_module.os.remove

    async def scenario():
        loop = asyncio.get_running_loop()
        delete_started = asyncio.Event()
        delete_finished = asyncio.Event()
        save_started = asyncio.Event()
        original_save = storage._save_sync

        def controlled_remove(path):
            if path == expired_path:
                loop.call_soon_threadsafe(delete_started.set)
                assert release_delete.wait(timeout=5), "test did not release the TTL delete"
            try:
                return original_remove(path)
            finally:
                if path == expired_path:
                    loop.call_soon_threadsafe(delete_finished.set)

        def controlled_save(data, name):
            loop.call_soon_threadsafe(save_started.set)
            return original_save(data, name)

        monkeypatch.setattr(storage_module.os, "remove", controlled_remove)
        monkeypatch.setattr(storage, "_save_sync", controlled_save)
        sweep = asyncio.create_task(storage._sweep_once(time.time() - 1))
        save = None
        try:
            await asyncio.wait_for(delete_started.wait(), timeout=5)
            sweep.cancel()
            with pytest.raises(asyncio.CancelledError):
                await sweep
            save = asyncio.create_task(storage.save(b"new", "fresh"))
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(save_started.wait(), timeout=0.1)
            release_delete.set()
            await asyncio.wait_for(save, timeout=5)
            assert not os.path.exists(expired_path)
            assert storage.exists("fresh")
        finally:
            release_delete.set()
            await asyncio.wait_for(delete_finished.wait(), timeout=5)
            if save is not None:
                await asyncio.wait_for(save, timeout=5)

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["save", "delete"])
def test_storage_io_error_propagates_and_releases_slot(tmp_path, monkeypatch, operation):
    storage = LocalStorageManager(storage_path=str(tmp_path / "storage"), max_concurrency=1)
    original = getattr(storage, f"_{operation}_sync")

    def fail(*args):
        raise OSError("disk failure")

    async def scenario():
        monkeypatch.setattr(storage, f"_{operation}_sync", fail)
        with pytest.raises(OSError, match="disk failure"):
            if operation == "save":
                await storage.save(b"data", "video")
            else:
                await storage.delete("video")
        monkeypatch.setattr(storage, f"_{operation}_sync", original)
        await asyncio.wait_for(storage.save(b"data", "next"), timeout=5)
        assert storage.exists("next")

    asyncio.run(scenario())
