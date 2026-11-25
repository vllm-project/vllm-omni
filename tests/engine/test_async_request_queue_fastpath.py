# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import threading
from unittest.mock import AsyncMock

import janus
import pytest

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.messages import AbortRequestMessage, StageSubmissionMessage

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def cross_loop_engine():
    engine = object.__new__(AsyncOmniEngine)
    engine.request_queue = janus.Queue(maxsize=2)
    loop = asyncio.new_event_loop()
    engine.orchestrator_loop = loop
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    async def bind():
        await engine.request_queue.async_q.put(None)
        await engine.request_queue.async_q.get()
        engine.request_queue.async_q.task_done()

    asyncio.run_coroutine_threadsafe(bind(), loop).result(timeout=5)
    try:
        yield engine
    finally:
        engine.request_queue.close()
        asyncio.run_coroutine_threadsafe(engine.request_queue.wait_closed(), loop).result(timeout=5)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        assert not thread.is_alive()
        loop.close()


def message(artifact_dirs=None):
    return StageSubmissionMessage(
        type="add_request",
        request_id="req-1",
        prompt={"prompt_token_ids": [1]},
        original_prompt={"prompt_token_ids": [1]},
        output_prompt_text=None,
        sampling_params_list=[],
        final_stage_id=0,
        preprocess_ms=0,
        request_timestamp=0,
        enqueue_ts=0,
        request_artifact_dirs=artifact_dirs,
    )


def test_unsaturated_cross_loop_put_does_not_wait_for_owner(cross_loop_engine):
    engine = cross_loop_engine
    entered, release = threading.Event(), threading.Event()

    def freeze_owner():
        entered.set()
        release.wait(timeout=5)

    engine.orchestrator_loop.call_soon_threadsafe(freeze_owner)
    assert entered.wait(timeout=5)
    parent = message()
    outcome = {"parent_enqueued": False}

    async def submit():
        await asyncio.wait_for(engine._put_request_queue_group(parent, [], outcome), timeout=0.2)

    try:
        asyncio.run(submit())
        assert outcome["parent_enqueued"]
        abort = AbortRequestMessage(request_ids=[parent.request_id])
        engine.request_queue.sync_q.put_nowait(abort)
        assert engine.request_queue.sync_q.get_nowait() is parent
        assert engine.request_queue.sync_q.get_nowait() is abort
    finally:
        release.set()


def test_fast_put_notifies_owner_async_consumer(cross_loop_engine):
    engine = cross_loop_engine
    waiting = threading.Event()

    async def consume():
        waiting.set()
        return await engine.request_queue.async_q.get()

    received = asyncio.run_coroutine_threadsafe(consume(), engine.orchestrator_loop)
    assert waiting.wait(timeout=5)
    parent = message()
    outcome = {"parent_enqueued": False}
    asyncio.run(engine._put_request_queue_group(parent, [], outcome))
    assert received.result(timeout=5) is parent
    assert outcome["parent_enqueued"]


def test_closed_fast_put_preserves_async_error_and_artifact_cleanup(cross_loop_engine, tmp_path):
    engine = cross_loop_engine
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    parent = message([str(artifacts)])
    outcome = {"parent_enqueued": False}
    engine.request_queue.close()
    with pytest.raises(getattr(janus, "AsyncQueueShutDown", RuntimeError)):
        asyncio.run(engine._put_request_queue_group(parent, [], outcome))
    assert not outcome["parent_enqueued"]
    assert not artifacts.exists()


def test_unexpected_fast_put_error_is_not_hidden(cross_loop_engine, tmp_path, monkeypatch):
    engine = cross_loop_engine
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    parent = message([str(artifacts)])
    outcome = {"parent_enqueued": False}
    bridge = AsyncMock()
    monkeypatch.setattr(engine, "_bridge_owned_put", bridge)

    def broken_put(self, item):
        raise RuntimeError("unexpected queue failure")

    monkeypatch.setattr(type(engine.request_queue.sync_q), "put_nowait", broken_put)
    with pytest.raises(RuntimeError, match="unexpected queue failure"):
        asyncio.run(engine._put_request_queue_group(parent, [], outcome))
    assert not outcome["parent_enqueued"]
    assert not artifacts.exists()
    bridge.assert_not_awaited()
