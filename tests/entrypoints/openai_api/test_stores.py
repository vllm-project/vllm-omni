# SPDX-License-Identifier: Apache-2.0
"""Unit tests for stores.py — based on current implementation."""

import asyncio
import threading

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _video(video_id: str = "vid_abc", model: str = "test-model", prompt: str = "a cat"):
    from vllm_omni.entrypoints.openai.protocol.videos import VideoResponse

    return VideoResponse(id=video_id, model=model, prompt=prompt)


def _memory_store():
    from vllm_omni.entrypoints.openai.stores import InMemoryVideoMetadataStore

    return InMemoryVideoMetadataStore()


def _disk_store(tmp_path):
    from vllm_omni.entrypoints.openai.stores import DiskCacheVideoMetadataStore

    return DiskCacheVideoMetadataStore(directory=str(tmp_path / "store"))


# ---------------------------------------------------------------------------
# AsyncDictStore — exercised via InMemoryVideoMetadataStore
# ---------------------------------------------------------------------------


async def test_dict_upsert_and_get():
    store = _memory_store()
    v = _video()
    await store.upsert(v.id, v)
    result = await store.get(v.id)
    assert result is not None
    assert result.id == v.id


async def test_dict_get_missing_returns_none():
    assert await _memory_store().get("ghost") is None


async def test_dict_pop_removes_and_returns():
    store = _memory_store()
    v = _video()
    await store.upsert(v.id, v)
    popped = await store.pop(v.id)
    assert popped is not None
    assert await store.get(v.id) is None


async def test_dict_pop_missing_returns_none():
    assert await _memory_store().pop("ghost") is None


async def test_dict_update_fields():
    from vllm_omni.entrypoints.openai.protocol.videos import VideoGenerationStatus

    store = _memory_store()
    v = _video()
    await store.upsert(v.id, v)

    updated = await store.update_fields(v.id, {"status": VideoGenerationStatus.IN_PROGRESS, "progress": 42})
    assert updated is not None
    assert updated.status == VideoGenerationStatus.IN_PROGRESS
    assert updated.progress == 42

    fetched = await store.get(v.id)
    assert fetched.status == VideoGenerationStatus.IN_PROGRESS
    assert fetched.progress == 42


async def test_dict_update_fields_missing_returns_none():
    assert await _memory_store().update_fields("ghost", {"progress": 1}) is None


async def test_dict_list_values():
    store = _memory_store()
    v1, v2 = _video("id_1"), _video("id_2")
    await store.upsert(v1.id, v1)
    await store.upsert(v2.id, v2)
    assert {v.id for v in await store.list_values()} == {"id_1", "id_2"}


async def test_dict_upsert_overwrites():
    store = _memory_store()
    v = _video()
    await store.upsert(v.id, v)
    await store.upsert(v.id, v.model_copy(update={"prompt": "a dog"}))
    assert (await store.get(v.id)).prompt == "a dog"


async def test_dict_update_fields_original_unchanged():
    """update_fields must not mutate the previously stored object."""
    store = _memory_store()
    v = _video()
    await store.upsert(v.id, v)

    await store.update_fields(v.id, {"progress": 99})

    assert v.progress == 0  # original object untouched
    assert (await store.get(v.id)).progress == 99  # store reflects the change


# ---------------------------------------------------------------------------
# DiskCacheVideoMetadataStore
# ---------------------------------------------------------------------------


async def test_disk_upsert_and_get(tmp_path):
    store = _disk_store(tmp_path)
    v = _video()
    await store.upsert(v.id, v)
    result = await store.get(v.id)
    assert result is not None
    assert result.id == v.id
    store.close()


async def test_disk_get_missing_returns_none(tmp_path):
    store = _disk_store(tmp_path)
    assert await store.get("ghost") is None
    store.close()


async def test_disk_pop_removes_and_returns(tmp_path):
    store = _disk_store(tmp_path)
    v = _video()
    await store.upsert(v.id, v)
    popped = await store.pop(v.id)
    assert popped is not None
    assert popped.id == v.id
    assert await store.get(v.id) is None
    store.close()


async def test_disk_pop_missing_returns_none(tmp_path):
    store = _disk_store(tmp_path)
    assert await store.pop("ghost") is None
    store.close()


async def test_disk_update_fields(tmp_path):
    from vllm_omni.entrypoints.openai.protocol.videos import VideoGenerationStatus

    store = _disk_store(tmp_path)
    v = _video()
    await store.upsert(v.id, v)

    updated = await store.update_fields(v.id, {"status": VideoGenerationStatus.IN_PROGRESS, "progress": 77})
    assert updated is not None
    assert updated.progress == 77
    assert (await store.get(v.id)).progress == 77
    store.close()


async def test_disk_update_fields_missing_returns_none(tmp_path):
    store = _disk_store(tmp_path)
    assert await store.update_fields("ghost", {"progress": 1}) is None
    store.close()


async def test_disk_list_values(tmp_path):
    store = _disk_store(tmp_path)
    v1, v2 = _video("id_1"), _video("id_2")
    await store.upsert(v1.id, v1)
    await store.upsert(v2.id, v2)
    assert {v.id for v in await store.list_values()} == {"id_1", "id_2"}
    store.close()


async def test_disk_upsert_overwrites(tmp_path):
    store = _disk_store(tmp_path)
    v = _video()
    await store.upsert(v.id, v)
    await store.upsert(v.id, v.model_copy(update={"prompt": "a dog"}))
    assert (await store.get(v.id)).prompt == "a dog"
    store.close()


async def test_disk_persistence_across_instances(tmp_path):
    """Data survives close + reopen on the same directory."""
    from vllm_omni.entrypoints.openai.stores import DiskCacheVideoMetadataStore

    directory = str(tmp_path / "store")
    v = _video()

    s1 = DiskCacheVideoMetadataStore(directory=directory)
    await s1.upsert(v.id, v)
    s1.close()

    s2 = DiskCacheVideoMetadataStore(directory=directory)
    result = await s2.get(v.id)
    s2.close()

    assert result is not None
    assert result.id == v.id


async def test_disk_list_values_no_keyerror_on_concurrent_delete(tmp_path):
    """list_values must not raise when a key is deleted mid-iteration."""
    from vllm_omni.entrypoints.openai.stores import DiskCacheVideoMetadataStore

    store = DiskCacheVideoMetadataStore(directory=str(tmp_path / "store"))
    for i in range(10):
        await store.upsert(f"id_{i}", _video(f"id_{i}"))

    def _delete_half():
        for i in range(0, 10, 2):
            store._cache.delete(f"id_{i}")

    t = threading.Thread(target=_delete_half)
    t.start()
    values = await store.list_values()  # must not raise KeyError
    t.join()

    assert isinstance(values, list)
    store.close()


# ---------------------------------------------------------------------------
# init_video_store — parameter validation
# ---------------------------------------------------------------------------


async def test_init_video_store_diskcache_requires_directory():
    from vllm_omni.entrypoints.openai.stores import init_video_store

    with pytest.raises(ValueError, match="directory cannot be empty"):
        init_video_store("diskcache")


async def test_init_video_store_unknown_backend():
    from vllm_omni.entrypoints.openai.stores import init_video_store

    with pytest.raises(ValueError, match="Unknown video-metadata-store backend"):
        init_video_store("redis")


# ---------------------------------------------------------------------------
# TaskRegistry
# ---------------------------------------------------------------------------


async def test_task_registry_upsert_and_get():
    from vllm_omni.entrypoints.openai.stores import TaskRegistry

    reg = TaskRegistry()

    async def _noop():
        pass

    task = asyncio.create_task(_noop())
    await reg.upsert("t1", task)
    assert await reg.get("t1") is task
    await task


async def test_task_registry_get_missing_returns_none():
    from vllm_omni.entrypoints.openai.stores import TaskRegistry

    assert await TaskRegistry().get("ghost") is None


async def test_task_registry_pop_removes_and_returns():
    from vllm_omni.entrypoints.openai.stores import TaskRegistry

    reg = TaskRegistry()

    async def _long():
        await asyncio.sleep(100)

    task = asyncio.create_task(_long())
    await reg.upsert("t1", task)
    popped = await reg.pop("t1")
    assert popped is task
    assert await reg.get("t1") is None
    task.cancel()


async def test_task_registry_pop_missing_returns_none():
    from vllm_omni.entrypoints.openai.stores import TaskRegistry

    assert await TaskRegistry().pop("ghost") is None


async def test_task_registry_auto_cleanup_on_done():
    from vllm_omni.entrypoints.openai.stores import TaskRegistry

    reg = TaskRegistry()

    async def _noop():
        pass

    task = asyncio.create_task(_noop())
    await reg.upsert("t1", task)
    await task
    await asyncio.sleep(0)
    await asyncio.sleep(0)  # let done-callback coroutine flush

    assert await reg.get("t1") is None
