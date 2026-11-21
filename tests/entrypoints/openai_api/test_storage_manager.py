# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Unit tests for the OpenAI video storage backend.
"""

import asyncio
import json
import os
import time

import opendal
import pytest

from tests.entrypoints.openai_api.storage_test_utils import build_file_storage, build_memory_storage
from vllm_omni.entrypoints.openai.storage import (
    FileStorageHandle,
    OpenDALBackend,
    SaveContext,
    SweepStrategy,
)
from vllm_omni.entrypoints.openai.storage import opendal_backend as backend_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_open_returns_file_handle_for_saved_key(tmp_path):
    storage = build_file_storage(tmp_path)
    save_context = asyncio.run(storage.save("video-123", b"video-bytes"))

    handle = asyncio.run(storage.open(save_context.key))

    assert isinstance(handle, FileStorageHandle)
    assert handle.path == os.path.join(storage.storage_path, "artifacts", "video-123")
    with open(handle.path, "rb") as file:
        assert file.read() == b"video-bytes"


def test_open_returns_none_for_missing_key(tmp_path):
    storage = build_file_storage(tmp_path)

    assert asyncio.run(storage.open("missing-video")) is None


def test_delete_removes_artifact_and_record(tmp_path):
    storage = build_file_storage(tmp_path, default_ttl=60, sweep_interval=300)
    save_context = asyncio.run(storage.save("video-delete", b"video-bytes"))
    assert asyncio.run(storage.record_get(save_context.key)) is not None

    assert asyncio.run(storage.delete(save_context.key)) is True
    assert asyncio.run(storage.open(save_context.key)) is None
    assert asyncio.run(storage.record_get(save_context.key)) is None
    # Idempotent: the second delete reports that nothing was removed.
    assert asyncio.run(storage.delete(save_context.key)) is False


def test_delete_artifact_keeps_the_record(tmp_path):
    storage = build_file_storage(tmp_path, default_ttl=60, sweep_interval=300)
    asyncio.run(storage.save("video-delete", b"video-bytes"))

    assert asyncio.run(storage.delete_artifact("video-delete")) is True

    assert asyncio.run(storage.open("video-delete")) is None
    assert asyncio.run(storage.record_get("video-delete")) is not None
    assert asyncio.run(storage.delete_artifact("video-delete")) is False


@pytest.mark.parametrize("key", ["../escape", "a/../../b", "/absolute", "", "trailing/"])
def test_invalid_keys_are_rejected(tmp_path, key):
    storage = build_file_storage(tmp_path)

    with pytest.raises(ValueError, match="Illegal storage key"):
        asyncio.run(storage.save(key, b"video-bytes"))


def test_iter_chunks_streams_artifact(tmp_path):
    storage = build_file_storage(tmp_path)
    asyncio.run(storage.save("video-chunks", b"0123456789"))

    async def collect():
        return [chunk async for chunk in storage.iter_chunks("video-chunks", chunk_size=4)]

    assert asyncio.run(collect()) == [b"0123", b"4567", b"89"]


def test_save_with_ttl_sets_expiration_and_records_it(tmp_path, monkeypatch):
    storage = build_file_storage(tmp_path, default_ttl=60, sweep_interval=300)
    monkeypatch.setattr(backend_module.time, "time", lambda: 1_700_000_000)

    save_context = asyncio.run(storage.save("video-ttl", b"video-bytes"))

    assert save_context.key == "video-ttl"
    assert isinstance(save_context, SaveContext)
    assert save_context.created_at == 1_700_000_000
    assert save_context.expires_at == 1_700_000_060
    record = json.loads(asyncio.run(storage.record_get("video-ttl")))
    assert record == {"expires_at": 1_700_000_060}


def test_save_merges_ttl_into_existing_record(tmp_path, monkeypatch):
    storage = build_file_storage(tmp_path, default_ttl=60, sweep_interval=300)
    monkeypatch.setattr(backend_module.time, "time", lambda: 1_700_000_000)
    asyncio.run(storage.record_set("video-record", b'{"id": "video-record", "status": "in_progress"}'))

    asyncio.run(storage.save("video-record", b"video-bytes"))

    record = json.loads(asyncio.run(storage.record_get("video-record")))
    assert record == {"id": "video-record", "status": "in_progress", "expires_at": 1_700_000_060}


def test_records_roundtrip_and_pop(tmp_path):
    storage = build_file_storage(tmp_path)

    async def scenario():
        await storage.record_set("a", b"{}")
        await storage.record_set("b", b"{}")
        keys = await storage.record_keys()
        popped = await storage.record_pop("a")
        return keys, popped, await storage.record_pop("a"), await storage.record_keys()

    keys, popped, popped_again, remaining = asyncio.run(scenario())
    assert keys == ["a", "b"]
    assert popped == b"{}"
    assert popped_again is None
    assert remaining == ["b"]


def test_ttl_requires_a_sweep_interval(tmp_path):
    with pytest.raises(ValueError, match="sweep_interval"):
        build_file_storage(tmp_path, default_ttl=60)


def test_ttl_sweeper_removes_expired_artifacts_by_last_modified(tmp_path):
    storage = build_file_storage(tmp_path, default_ttl=1, sweep_interval=60)
    asyncio.run(storage.save("video-expired", b"video-bytes"))
    asyncio.run(storage.save("video-fresh", b"video-bytes"))
    stale = time.time() - 10
    os.utime(storage.get_full_file_path("video-expired"), (stale, stale))

    assert asyncio.run(storage.sweeper.strategy()) is SweepStrategy.LAST_MODIFIED
    assert asyncio.run(storage.sweeper.sweep_once()) == 1

    assert asyncio.run(storage.open("video-expired")) is None
    assert asyncio.run(storage.record_get("video-expired")) is None
    assert asyncio.run(storage.open("video-fresh")) is not None


def test_ttl_sweeper_keeps_artifacts_that_have_not_expired(tmp_path):
    storage = build_file_storage(tmp_path, default_ttl=60, sweep_interval=60)
    asyncio.run(storage.save("video-fresh", b"video-bytes"))

    assert asyncio.run(storage.sweeper.sweep_once()) == 0
    assert asyncio.run(storage.open("video-fresh")) is not None


def test_ttl_sweeper_falls_back_to_record_expiry_without_last_modified():
    storage = build_memory_storage(default_ttl=60, sweep_interval=60)
    asyncio.run(storage.save("video-expired", b"video-bytes"))
    asyncio.run(storage.save("video-fresh", b"video-bytes"))
    record = json.loads(asyncio.run(storage.record_get("video-expired")))
    record["expires_at"] = int(time.time()) - 1
    asyncio.run(storage.record_set("video-expired", json.dumps(record).encode()))

    assert asyncio.run(storage.sweeper.strategy()) is SweepStrategy.RECORD_EXPIRES_AT
    assert asyncio.run(storage.sweeper.sweep_once()) == 1

    async def remaining():
        return [key async for key, _ in storage.list_artifacts()]

    assert asyncio.run(remaining()) == ["video-fresh"]
    assert asyncio.run(storage.record_get("video-expired")) is None


def test_ttl_sweeper_skips_backends_without_listing():
    class _NoListBackend(OpenDALBackend):
        # e.g. redis: objects expire through a native default_ttl instead.
        @property
        def supports_list(self) -> bool:
            return False

    storage = _NoListBackend(opendal.AsyncOperator("memory"), default_ttl=60, sweep_interval=60)

    assert asyncio.run(storage.sweeper.strategy()) is SweepStrategy.NATIVE_TTL
    assert asyncio.run(storage.sweeper.sweep_once()) == 0


def test_ttl_sweeper_notifies_on_expire(tmp_path):
    storage = build_file_storage(tmp_path, default_ttl=1, sweep_interval=60)
    expired: list[str] = []

    async def on_expire(keys: list[str]) -> None:
        expired.extend(keys)

    asyncio.run(storage.start(on_expire=on_expire))
    try:
        asyncio.run(storage.save("video-expired", b"video-bytes"))
        asyncio.run(storage.save("video-fresh", b"video-bytes"))
        stale = time.time() - 10
        os.utime(storage.get_full_file_path("video-expired"), (stale, stale))

        assert asyncio.run(storage.sweeper.sweep_once()) == 1
    finally:
        asyncio.run(storage.stop())

    assert expired == ["video-expired"]


def test_ttl_sweeper_survives_a_failing_on_expire(tmp_path):
    storage = build_file_storage(tmp_path, default_ttl=1, sweep_interval=60)

    async def on_expire(keys: list[str]) -> None:
        raise RuntimeError("handler broke")

    asyncio.run(storage.start(on_expire=on_expire))
    try:
        asyncio.run(storage.save("video-expired", b"video-bytes"))
        stale = time.time() - 10
        os.utime(storage.get_full_file_path("video-expired"), (stale, stale))

        # The artifact is gone and the sweep keeps going even if the handler fails.
        assert asyncio.run(storage.sweeper.sweep_once()) == 1
        assert asyncio.run(storage.open("video-expired")) is None
    finally:
        asyncio.run(storage.stop())


def test_storage_path_can_be_repointed(tmp_path):
    storage = build_file_storage(tmp_path)
    asyncio.run(storage.save("video-1", b"video-bytes"))

    other_root = tmp_path / "other"
    storage.storage_path = str(other_root)

    assert asyncio.run(storage.open("video-1")) is None
    asyncio.run(storage.save("video-2", b"video-bytes"))
    assert os.path.exists(os.path.join(str(other_root), "artifacts", "video-2"))
