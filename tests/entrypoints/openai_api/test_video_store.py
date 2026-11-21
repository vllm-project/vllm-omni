# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Unit tests for the persistent video job store.
"""

import asyncio
import json

import pytest

from tests.entrypoints.openai_api.storage_test_utils import build_file_storage
from vllm_omni.entrypoints.openai.protocol.videos import VideoGenerationStatus, VideoResponse
from vllm_omni.entrypoints.openai.stores import ORPHANED_JOB_MESSAGE, AsyncDictStore, VideoStore
from vllm_omni.entrypoints.openai.video.paths import lidar_storage_key

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _job(video_id: str = "video-1", status: VideoGenerationStatus = VideoGenerationStatus.QUEUED) -> VideoResponse:
    return VideoResponse(model="test-model", prompt="a prompt", id=video_id, status=status)


def test_upsert_writes_the_job_record(tmp_path):
    storage = build_file_storage(tmp_path)
    store = VideoStore(storage)

    asyncio.run(store.upsert("video-1", _job()))

    record = json.loads(asyncio.run(storage.record_get("video-1")))
    assert record["id"] == "video-1"
    assert record["status"] == "queued"


def test_update_fields_persists_the_merged_record(tmp_path):
    storage = build_file_storage(tmp_path)
    store = VideoStore(storage)
    asyncio.run(store.upsert("video-1", _job()))

    asyncio.run(
        store.update_fields(
            "video-1",
            {"status": VideoGenerationStatus.COMPLETED, "file_name": "video-1.mp4", "progress": 100},
        )
    )

    record = json.loads(asyncio.run(storage.record_get("video-1")))
    assert record["status"] == "completed"
    assert record["file_name"] == "video-1.mp4"


def test_pop_removes_the_job_record(tmp_path):
    storage = build_file_storage(tmp_path)
    store = VideoStore(storage)
    asyncio.run(store.upsert("video-1", _job()))

    assert asyncio.run(store.pop("video-1")) is not None

    assert asyncio.run(storage.record_get("video-1")) is None
    assert asyncio.run(store.get("video-1")) is None


def test_jobs_survive_a_restart(tmp_path):
    storage = build_file_storage(tmp_path)
    first = VideoStore(storage)
    asyncio.run(first.upsert("video-1", _job(status=VideoGenerationStatus.COMPLETED)))

    restarted = VideoStore(storage)
    assert asyncio.run(restarted.list_values()) == []

    assert asyncio.run(restarted.load()) == 1
    job = asyncio.run(restarted.get("video-1"))
    assert job is not None
    assert job.status is VideoGenerationStatus.COMPLETED


def test_get_reads_a_record_that_is_not_in_memory(tmp_path):
    storage = build_file_storage(tmp_path)
    writer = VideoStore(storage)
    asyncio.run(writer.upsert("video-1", _job()))

    reader = VideoStore(storage)
    job = asyncio.run(reader.get("video-1"))

    assert job is not None and job.id == "video-1"


def test_unreadable_records_are_ignored(tmp_path):
    storage = build_file_storage(tmp_path)
    asyncio.run(storage.record_set("video-broken", b"{not json"))
    store = VideoStore(storage)

    assert asyncio.run(store.load()) == 0
    assert asyncio.run(store.get("video-broken")) is None


def test_reconcile_orphans_fails_interrupted_jobs(tmp_path):
    storage = build_file_storage(tmp_path)
    first = VideoStore(storage)
    asyncio.run(first.upsert("video-queued", _job("video-queued", VideoGenerationStatus.QUEUED)))
    asyncio.run(first.upsert("video-running", _job("video-running", VideoGenerationStatus.IN_PROGRESS)))
    asyncio.run(first.upsert("video-done", _job("video-done", VideoGenerationStatus.COMPLETED)))
    # A partially written artifact of an interrupted job must not survive.
    asyncio.run(storage.save("video-running", b"partial-bytes"))
    # A completed job keeps its artifact.
    asyncio.run(storage.save("video-done", b"video-bytes"))

    restarted = VideoStore(storage)
    orphaned = asyncio.run(restarted.reconcile_orphans())

    assert sorted(orphaned) == ["video-queued", "video-running"]
    for video_id in ("video-queued", "video-running"):
        job = asyncio.run(restarted.get(video_id))
        assert job is not None
        assert job.status is VideoGenerationStatus.FAILED
        assert job.error is not None
        assert job.error.message == ORPHANED_JOB_MESSAGE
        record = json.loads(asyncio.run(storage.record_get(video_id)))
        assert record["status"] == "failed"
    assert asyncio.run(storage.open("video-running")) is None
    completed = asyncio.run(restarted.get("video-done"))
    assert completed is not None and completed.status is VideoGenerationStatus.COMPLETED
    assert asyncio.run(storage.open("video-done")) is not None


class _FlakyRecordBackend:
    """Wrap a backend and fail the record write for one chosen status."""

    def __init__(self, inner, *, fail_status: str) -> None:
        self._inner = inner
        self._fail_status = fail_status

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def record_set(self, key, data):
        if json.loads(data).get("status") == self._fail_status:
            raise OSError("record store unavailable")
        await self._inner.record_set(key, data)


def test_completion_is_not_published_when_the_record_write_fails(tmp_path):
    storage = build_file_storage(tmp_path)
    store = VideoStore(_FlakyRecordBackend(storage, fail_status="completed"))
    asyncio.run(store.upsert("video-1", _job()))
    asyncio.run(storage.save("video-1", b"video-bytes"))

    with pytest.raises(OSError, match="record store unavailable"):
        asyncio.run(
            store.update_fields(
                "video-1",
                {"status": VideoGenerationStatus.COMPLETED, "file_name": "video-1.mp4"},
                durable=True,
            )
        )

    # The client never sees a completion that is not durable.
    job = asyncio.run(store.get("video-1"))
    assert job is not None and job.status is VideoGenerationStatus.QUEUED
    assert json.loads(asyncio.run(storage.record_get("video-1")))["status"] == "queued"

    # After a restart the stale record is treated as interrupted: no false
    # completion, and the artifact is cleaned up.
    restarted = VideoStore(storage)
    assert asyncio.run(restarted.reconcile_orphans()) == ["video-1"]
    job = asyncio.run(restarted.get("video-1"))
    assert job is not None and job.status is VideoGenerationStatus.FAILED
    assert asyncio.run(storage.open("video-1")) is None


def test_acknowledged_completion_survives_restart_with_sidecar(tmp_path):
    storage = build_file_storage(tmp_path)
    store = VideoStore(storage)
    asyncio.run(store.upsert("video-1", _job()))
    asyncio.run(storage.save("video-1", b"mp4"))
    asyncio.run(storage.save(lidar_storage_key("video-1"), b"LIDAR"))

    asyncio.run(
        store.update_fields(
            "video-1",
            {"status": VideoGenerationStatus.COMPLETED, "file_name": "video-1.mp4", "lidar": None},
            durable=True,
        )
    )

    restarted = VideoStore(storage)
    assert asyncio.run(restarted.reconcile_orphans()) == []

    job = asyncio.run(restarted.get("video-1"))
    assert job is not None and job.status is VideoGenerationStatus.COMPLETED
    assert asyncio.run(storage.open("video-1")) is not None
    assert asyncio.run(storage.open(lidar_storage_key("video-1"))) is not None


def test_reconcile_removes_the_lidar_sidecar_of_an_interrupted_job(tmp_path):
    storage = build_file_storage(tmp_path)
    first = VideoStore(storage)
    asyncio.run(first.upsert("video-1", _job("video-1", VideoGenerationStatus.IN_PROGRESS)))
    asyncio.run(storage.save("video-1", b"partial-mp4"))
    asyncio.run(storage.save(lidar_storage_key("video-1"), b"partial-lidar"))

    restarted = VideoStore(storage)
    assert asyncio.run(restarted.reconcile_orphans()) == ["video-1"]

    assert asyncio.run(storage.open("video-1")) is None
    assert asyncio.run(storage.open(lidar_storage_key("video-1"))) is None
    job = asyncio.run(restarted.get("video-1"))
    assert job is not None and job.status is VideoGenerationStatus.FAILED


def test_reconcile_without_a_backend_is_a_no_op():
    store = VideoStore(None)
    asyncio.run(store.upsert("video-1", _job()))

    assert asyncio.run(store.reconcile_orphans()) == ["video-1"]
    job = asyncio.run(store.get("video-1"))
    assert job is not None and job.status is VideoGenerationStatus.FAILED


def test_record_write_failures_are_not_fatal():
    class _BrokenBackend:
        async def record_set(self, key, data):
            raise OSError("storage down")

        async def record_get(self, key):
            return None

        async def record_pop(self, key):
            return None

        async def record_keys(self):
            return []

        async def delete_artifact(self, key):
            return False

    store = VideoStore(_BrokenBackend())
    asyncio.run(store.upsert("video-1", _job()))

    assert asyncio.run(store.get("video-1")) is not None


def test_remove_many_evicts_expired_jobs(tmp_path):
    storage = build_file_storage(tmp_path)
    store = VideoStore(storage)
    asyncio.run(store.upsert("video-1", _job()))
    asyncio.run(store.upsert("video-2", _job("video-2")))

    asyncio.run(store.remove_many(["video-1"]))

    assert [job.id for job in asyncio.run(store.list_values())] == ["video-2"]
    # Metadata persistence is untouched by an in-memory eviction.
    assert asyncio.run(storage.record_get("video-1")) is not None


def test_plain_store_reconcile_is_a_no_op():
    store: AsyncDictStore[VideoResponse] = AsyncDictStore()
    asyncio.run(store.upsert("video-1", _job()))

    assert asyncio.run(store.reconcile_orphans()) == []
