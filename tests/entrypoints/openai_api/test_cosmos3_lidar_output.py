# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Numeric output persistence and lifecycle for asynchronous video jobs."""

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from fastapi import HTTPException
from safetensors.torch import load_file

from vllm_omni.entrypoints.openai import api_server
from vllm_omni.entrypoints.openai.protocol.videos import VideoAction, VideoGenerationRequest, VideoResponse
from vllm_omni.entrypoints.openai.serving_video import (
    EncodedVideoResult,
    OmniOpenAIServingVideo,
    VideoGenerationArtifacts,
)
from vllm_omni.entrypoints.openai.storage import LocalStorageTTLManager
from vllm_omni.entrypoints.openai.stores import AsyncDictStore, TaskRegistry
from vllm_omni.entrypoints.openai.video.generation import helpers
from vllm_omni.model_extras.cosmos3_lidar import serialize_lidar_output

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def frames_and_metadata():
    frames = torch.ones(1, 3, 2, 128, 1800)
    frames[:, 0] = 52.5
    metadata = {
        "fps": 10,
        "num_frames": 2,
        "shape": list(frames.shape),
        "dtype": "float32",
        "channels": ["range", "intensity", "validity"],
        "units": ["metres", "unit", "binary"],
        "apply_validity_mask": True,
        "validity_threshold": 0.5,
        "start_time_seconds": 0.0,
        "range_projection": {"semantic_width": 1800, "model_width": 1808},
    }
    return frames, metadata


@pytest.fixture
def encoded(frames_and_metadata):
    data, metadata = serialize_lidar_output(*frames_and_metadata)
    return EncodedVideoResult(b"mp4", {"decode": 1}, 20, None, {"fps": 30}, data, metadata)


@pytest.fixture
def storage(monkeypatch, tmp_path):
    manager = LocalStorageTTLManager(60, 30, str(tmp_path))
    store = AsyncDictStore()
    for module in (api_server, helpers):
        monkeypatch.setattr(module, "STORAGE_MANAGER", manager)
        monkeypatch.setattr(module, "VIDEO_STORE", store)
    monkeypatch.setattr(api_server, "VIDEO_TASKS", TaskRegistry())
    return manager, store


async def run_job(store, encoded):
    await store.upsert("test", VideoResponse(id="test", model="cosmos3", prompt="drive"))
    handler = SimpleNamespace(generate_video_bytes=AsyncMock(return_value=encoded))
    request = VideoGenerationRequest(
        prompt="drive",
        extra_params={"lidar": {"return_output": True}} if isinstance(encoded, EncodedVideoResult) else None,
    )
    await helpers._run_video_generation_job(handler, request, "test")
    return await store.get("test")


async def drain_storage_tasks():
    # Completion callbacks may schedule a final delete after an abandoned save.
    await asyncio.sleep(0)
    while tasks := set(helpers._VIDEO_STORAGE_TASKS):
        _, pending = await asyncio.wait(tasks, timeout=1)
        assert not pending, "Deferred storage operations did not finish"
        await asyncio.sleep(0)


def test_async_job_persists_downloads_and_deletes_both_artifacts(storage, encoded, frames_and_metadata):
    manager, store = storage

    async def check():
        job = await run_job(store, encoded)
        assert job.status == "completed" and job.fps == 30
        assert job.lidar.fps == 10 and job.lidar.shape == [3, 2, 128, 1800]
        assert job.lidar.url == "/v1/videos/test/lidar"
        video = await api_server.download_video("test")
        assert Path(video.path).read_bytes() == b"mp4"
        lidar = await api_server.download_video_lidar("test")
        assert lidar.media_type == "application/octet-stream"
        assert lidar.filename == "test.lidar.safetensors"
        torch.testing.assert_close(load_file(lidar.path)["frames"], frames_and_metadata[0][0], rtol=0, atol=0)
        assert job.expires_at is not None
        await api_server.delete_video("test")
        assert await store.get("test") is None
        assert list(Path(manager.storage_path).iterdir()) == []

    asyncio.run(check())


@pytest.mark.parametrize("fail_key", ["test", "test.lidar.safetensors"])
def test_partial_write_failure_cleans_all_artifacts(storage, encoded, monkeypatch, fail_key):
    manager, store = storage
    save = manager.save

    async def fail(data, key):
        await save(data, key)
        if key == fail_key:
            raise OSError("disk failure")

    monkeypatch.setattr(manager, "save", fail)

    async def check():
        job = await run_job(store, encoded)
        assert job.status == "failed" and "disk failure" in job.error.message
        assert job.lidar is None
        assert list(Path(manager.storage_path).iterdir()) == []
        with pytest.raises(HTTPException) as error:
            await api_server.download_video_lidar("test")
        assert error.value.status_code == 422

    asyncio.run(check())


@pytest.mark.parametrize("fail_key", ["test", "test.lidar.safetensors"])
def test_failed_job_delete_attempts_both_files_despite_errors(storage, monkeypatch, caplog, fail_key):
    manager, store = storage
    delete = manager.delete
    attempted = []

    async def fail(key):
        attempted.append(key)
        if key == fail_key:
            raise OSError("disk failure")
        return await delete(key)

    monkeypatch.setattr(manager, "delete", fail)

    async def check():
        # A failed save can leave files before either descriptor is published.
        await store.upsert("test", VideoResponse(id="test", model="cosmos3", prompt="drive", status="failed"))
        for key in ("test", "test.lidar.safetensors"):
            await manager.save(b"partial", key)
        result = await api_server.delete_video("test")
        assert result.deleted and await store.get("test") is None
        assert set(attempted) == {"test", "test.lidar.safetensors"}
        assert [path.name for path in Path(manager.storage_path).iterdir()] == [fail_key]

    asyncio.run(check())
    assert "Failed to cleanup partial video artifact" in caplog.text


def test_completed_rgb_job_delete_skips_lidar_storage(storage, monkeypatch):
    manager, store = storage
    delete = manager.delete

    async def video_only(key):
        if key != "test":
            raise OSError("LiDAR deletion must not be attempted for an RGB-only job")
        return await delete(key)

    monkeypatch.setattr(manager, "delete", video_only)

    async def check():
        await store.upsert(
            "test", VideoResponse(id="test", model="cosmos3", prompt="drive", status="completed", file_name="test.mp4")
        )
        await manager.save(b"mp4", "test")
        result = await api_server.delete_video("test")
        assert result.deleted and await store.get("test") is None
        assert list(Path(manager.storage_path).iterdir()) == []

    asyncio.run(check())


@pytest.mark.parametrize("fail_key", ["test", "test.lidar.safetensors"])
def test_completed_joint_job_delete_can_retry_storage_errors(storage, encoded, monkeypatch, fail_key):
    manager, store = storage
    delete = manager.delete

    async def fail(key):
        if key == fail_key:
            raise OSError("disk failure")
        return await delete(key)

    async def check():
        await run_job(store, encoded)
        monkeypatch.setattr(manager, "delete", fail)
        with pytest.raises(OSError, match="disk failure"):
            await api_server.delete_video("test")
        assert await store.get("test") is not None
        monkeypatch.setattr(manager, "delete", delete)
        result = await api_server.delete_video("test")
        assert result.deleted and await store.get("test") is None
        assert list(Path(manager.storage_path).iterdir()) == []

    asyncio.run(check())


def test_cancellation_waits_within_grace_period_for_active_write(storage, encoded, monkeypatch):
    manager, store = storage
    save = manager.save

    async def check():
        writing, release = asyncio.Event(), asyncio.Event()

        async def delayed(data, key):
            if key.endswith("safetensors"):
                writing.set()
                await release.wait()
            return await save(data, key)

        monkeypatch.setattr(manager, "save", delayed)
        task = asyncio.create_task(run_job(store, encoded))
        await writing.wait()
        assert (await store.get("test")).status == "in_progress"
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        task.cancel()  # Another DELETE must not restart the grace period.
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await store.get("test") is None
        assert list(Path(manager.storage_path).iterdir()) == []

    asyncio.run(check())


@pytest.mark.parametrize("mode", ["rgb", "joint_video", "joint_lidar"])
@pytest.mark.parametrize("fail_after_write", [False, True])
def test_cancelled_save_has_fixed_deadline_and_cleans_late_output(
    storage, encoded, monkeypatch, mode, fail_after_write
):
    manager, store = storage
    save = manager.save
    # Ordinary video must not take the joint-output grace period.
    monkeypatch.setattr(helpers, "_VIDEO_SAVE_CANCEL_GRACE_S", 10 if mode == "rgb" else 0.03)
    blocked_key = "test.lidar.safetensors" if mode == "joint_lidar" else "test"
    result = (b"mp4", {}, 0.0, None, {}) if mode == "rgb" else encoded

    async def check():
        writing, release = asyncio.Event(), asyncio.Event()

        async def delayed(data, key):
            if key == blocked_key:
                writing.set()
                await release.wait()
            saved = await save(data, key)
            if key == blocked_key and fail_after_write:
                raise OSError("save failed after publishing")
            return saved

        monkeypatch.setattr(manager, "save", delayed)
        task = asyncio.create_task(run_job(store, result))
        await writing.wait()

        async def repeatedly_cancel():
            while not task.done():
                task.cancel()
                await asyncio.sleep(0.001)

        canceller = asyncio.create_task(repeatedly_cancel())
        try:
            done, _ = await asyncio.wait({task}, timeout=0.3)
            assert done, "Repeated cancellations extended the save deadline"
            assert task.cancelled()
            assert await store.get("test") is None
            assert helpers._VIDEO_STORAGE_TASKS, "The pending save must retain an owner"
        finally:
            canceller.cancel()
            release.set()
            await asyncio.gather(canceller, task, return_exceptions=True)
            await drain_storage_tasks()
        assert list(Path(manager.storage_path).iterdir()) == []

    asyncio.run(check())


def test_delete_returns_409_while_save_is_stalled(storage, encoded, monkeypatch):
    manager, store = storage
    save = manager.save
    monkeypatch.setattr(helpers, "_VIDEO_SAVE_CANCEL_GRACE_S", 0.5)
    monkeypatch.setattr(api_server, "VIDEO_DELETE_TIMEOUT_S", 0.02)

    async def check():
        writing, release = asyncio.Event(), asyncio.Event()

        async def delayed(data, key):
            writing.set()
            await release.wait()
            return await save(data, key)

        monkeypatch.setattr(manager, "save", delayed)
        task = asyncio.create_task(run_job(store, encoded))
        await api_server.VIDEO_TASKS.upsert("test", task)
        await writing.wait()
        deleting = asyncio.create_task(api_server.delete_video("test"))
        try:
            done, _ = await asyncio.wait({deleting}, timeout=0.2)
            assert done, "DELETE waited for the cancelled writer beyond its timeout"
            with pytest.raises(HTTPException) as error:
                deleting.result()
            assert error.value.status_code == 409
            assert not task.done() and await store.get("test") is not None
            assert task.cancelling() == 1, "The DELETE timeout must not cancel the job again"
        finally:
            release.set()
            await asyncio.gather(task, deleting, return_exceptions=True)
            await drain_storage_tasks()
        assert await store.get("test") is None
        assert list(Path(manager.storage_path).iterdir()) == []

    asyncio.run(check())


def test_cancelled_threaded_write_is_cleaned_after_publication(storage, monkeypatch):
    manager, store = storage
    save_sync = manager._save_sync

    async def check():
        writing, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()

        def blocked_save(data, key):
            loop.call_soon_threadsafe(writing.set)
            assert release.wait(timeout=2), "Test did not release the storage thread"
            return save_sync(data, key)

        monkeypatch.setattr(manager, "_save_sync", blocked_save)
        task = asyncio.create_task(run_job(store, (b"mp4", {}, 0.0, None, {})))
        await writing.wait()
        task.cancel()
        try:
            done, _ = await asyncio.wait({task}, timeout=0.3)
            assert done and task.cancelled()
            assert helpers._VIDEO_STORAGE_TASKS
            # Deferred cleanup belongs to the manager that started this write.
            monkeypatch.setattr(
                helpers, "STORAGE_MANAGER", SimpleNamespace(delete=AsyncMock(side_effect=AssertionError))
            )
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await drain_storage_tasks()
        assert await store.get("test") is None
        assert list(Path(manager.storage_path).iterdir()) == []

    asyncio.run(check())


def test_cancelling_delete_request_keeps_in_progress_job(storage, encoded, monkeypatch):
    manager, store = storage
    save = manager.save
    monkeypatch.setattr(helpers, "_VIDEO_SAVE_CANCEL_GRACE_S", 1)

    async def check():
        writing, release = asyncio.Event(), asyncio.Event()

        async def delayed(data, key):
            writing.set()
            await release.wait()
            return await save(data, key)

        monkeypatch.setattr(manager, "save", delayed)
        task = asyncio.create_task(run_job(store, encoded))
        await api_server.VIDEO_TASKS.upsert("test", task)
        await writing.wait()
        deleting = asyncio.create_task(api_server.delete_video("test"))
        try:
            await asyncio.sleep(0.01)
            deleting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await deleting
            assert not task.done() and await store.get("test") is not None
        finally:
            release.set()
            await asyncio.gather(task, deleting, return_exceptions=True)
            await drain_storage_tasks()

    asyncio.run(check())


def test_cancellation_does_not_wait_forever_for_cleanup(storage, monkeypatch):
    manager, store = storage
    delete = manager.delete
    monkeypatch.setattr(helpers, "_VIDEO_CLEANUP_TIMEOUT_S", 0.02)

    async def check():
        release, generating = asyncio.Event(), asyncio.Event()

        async def stalled_delete(key):
            await release.wait()
            return await delete(key)

        async def generate(*args, **kwargs):
            generating.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(manager, "delete", stalled_delete)
        await store.upsert("test", VideoResponse(id="test", model="cosmos3", prompt="drive"))
        for key in ("test", "test.lidar.safetensors"):
            await manager.save(b"partial", key)
        task = asyncio.create_task(
            helpers._run_video_generation_job(
                SimpleNamespace(generate_video_bytes=generate), VideoGenerationRequest(prompt="drive"), "test"
            )
        )
        await generating.wait()
        task.cancel()
        try:
            done, _ = await asyncio.wait({task}, timeout=0.2)
            assert done and task.cancelled(), "Storage cleanup blocked job cancellation"
            assert await store.get("test") is None
            assert helpers._VIDEO_STORAGE_TASKS
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await drain_storage_tasks()
        assert list(Path(manager.storage_path).iterdir()) == []

    asyncio.run(check())


def test_repeated_cancellation_does_not_skip_cleanup_of_either_file(storage, monkeypatch):
    manager, store = storage
    delete = manager.delete

    async def check():
        release, generating, deleting = asyncio.Event(), asyncio.Event(), asyncio.Event()
        attempted = set()

        async def delayed_delete(key):
            attempted.add(key)
            if len(attempted) == 2:
                deleting.set()
            await release.wait()
            return await delete(key)

        async def generate(*args, **kwargs):
            generating.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(manager, "delete", delayed_delete)
        await store.upsert("test", VideoResponse(id="test", model="cosmos3", prompt="drive"))
        for key in ("test", "test.lidar.safetensors"):
            await manager.save(b"partial", key)
        task = asyncio.create_task(
            helpers._run_video_generation_job(
                SimpleNamespace(generate_video_bytes=generate), VideoGenerationRequest(prompt="drive"), "test"
            )
        )
        await generating.wait()
        task.cancel()
        try:
            await asyncio.wait_for(deleting.wait(), timeout=0.2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert await store.get("test") is None
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await drain_storage_tasks()
        assert attempted == {"test", "test.lidar.safetensors"}
        assert list(Path(manager.storage_path).iterdir()) == []

    asyncio.run(check())


def test_both_artifacts_expire_through_storage_ttl(storage, encoded):
    manager, store = storage

    async def check():
        job = await run_job(store, encoded)
        assert await manager._sweep_once(job.completed_at + 61) == 2
        with pytest.raises(HTTPException) as error:
            await api_server.download_video_lidar("test")
        assert error.value.status_code == 404

    asyncio.run(check())


@pytest.mark.parametrize("status", [None, "queued", "in_progress", "completed"])
def test_lidar_download_rejects_missing_or_unavailable_outputs(storage, status):
    _, store = storage

    async def check():
        if status:
            await store.upsert("test", VideoResponse(id="test", model="cosmos3", prompt="drive", status=status))
        with pytest.raises(HTTPException) as error:
            await api_server.download_video_lidar("test")
        assert error.value.status_code == 404

    asyncio.run(check())


def test_sync_lidar_rejection_cleans_uploaded_files(storage, tmp_path):
    path = tmp_path / "upload.safetensors"
    path.write_bytes(b"upload")
    handler = SimpleNamespace(generate_video_bytes=AsyncMock())
    request = VideoGenerationRequest(prompt="drive", extra_params={"lidar": {"return_output": True}})
    ctx = (request, handler, "cosmos3", None, None, None, None, helpers.VideoUploadResources([str(path)]))

    async def check():
        with pytest.raises(HTTPException) as error:
            await api_server.create_video_sync(SimpleNamespace(state=SimpleNamespace()), ctx)
        assert error.value.status_code == 400 and "asynchronous" in error.value.detail
        handler.generate_video_bytes.assert_not_called()
        assert not path.exists()

    asyncio.run(check())


def test_serving_serializes_lidar_without_rgb_conversion(frames_and_metadata, monkeypatch):
    import vllm_omni.entrypoints.openai.serving_video as module

    frames, metadata = frames_and_metadata
    handler = object.__new__(OmniOpenAIServingVideo)
    handler._video_frame_converter = object()
    handler._run_and_extract = AsyncMock(
        return_value=VideoGenerationArtifacts(
            videos=["video"],
            audios=[None],
            actions=[None],
            audio_sample_rate=0,
            output_fps=30,
            stage_durations={},
            peak_memory_mb=0,
            lidar=frames,
            lidar_metadata=metadata,
        )
    )
    monkeypatch.setattr(
        module, "_encode_video_bytes", lambda video, **kw: b"mp4" if video == "video" else pytest.fail()
    )

    async def check():
        result = await handler.generate_video_bytes(VideoGenerationRequest(prompt="drive"), "test")
        assert isinstance(result, EncodedVideoResult)
        assert result.lidar_metadata["shape"] == [3, 2, 128, 1800]
        assert result.video_bytes == b"mp4"
        with pytest.raises(HTTPException, match="asynchronous"):
            await handler.generate_videos(
                VideoGenerationRequest(prompt="drive", extra_params={"lidar": {"return_output": True}}), "test"
            )

    asyncio.run(check())


@pytest.mark.parametrize("with_lidar", [False, True])
def test_serving_action_only_output_rejects_lidar(frames_and_metadata, monkeypatch, with_lidar):
    import vllm_omni.entrypoints.openai.serving_video as module

    frames, metadata = frames_and_metadata
    action = VideoAction(data=[[1.0]], shape=[1, 1])
    handler = object.__new__(OmniOpenAIServingVideo)
    handler._run_and_extract = AsyncMock(
        return_value=VideoGenerationArtifacts(
            videos=[{}],
            audios=[None],
            actions=[action],
            audio_sample_rate=0,
            output_fps=30,
            stage_durations={},
            peak_memory_mb=0,
            lidar=frames if with_lidar else None,
            lidar_metadata=metadata if with_lidar else {},
        )
    )
    monkeypatch.setattr(module, "_encode_video_bytes", lambda *a, **kw: pytest.fail("Unexpected MP4 encoding"))

    async def check():
        request = VideoGenerationRequest(prompt="drive")
        if with_lidar:
            with pytest.raises(ValueError, match="action-only output with LiDAR is unsupported"):
                await handler.generate_video_bytes(request, "test")
        else:
            result = await handler.generate_video_bytes(request, "test")
            assert result[0] == b"" and result[3] is action

    asyncio.run(check())
