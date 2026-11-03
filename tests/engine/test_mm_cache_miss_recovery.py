# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Recover sender-cache drift before a client retries a failed request."""

import asyncio
from types import SimpleNamespace

import janus
import pytest
from vllm.config import ModelConfig
from vllm.config.multimodal import MultiModalConfig
from vllm.exceptions import RetryableRequestError
from vllm.multimodal.cache import LruKeyReplicatedSenderCache
from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason

from vllm_omni.engine.async_engine_utils import OutputQueueSink
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.messages import ErrorMessage
from vllm_omni.entrypoints.omni_base import OmniBase

from .test_orchestrator import FakeStageClient, _build_stage_pools
from .test_orchestrator_error_handling import _build_bare_orchestrator, _register_request

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def sender_cache(mocker):
    config = mocker.Mock(spec=ModelConfig)
    config.get_multimodal_config.return_value = MultiModalConfig(mm_processor_cache_gb=0.01)
    cache = LruKeyReplicatedSenderCache(config)
    item = MultiModalKwargsItem.dummy()
    cache.get_and_update_item((item, []), "missing")
    cache.get_and_update_item((item, []), "healthy")
    assert cache.is_cached_item("missing")
    yield cache
    cache.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage_id", [0, 1])
@pytest.mark.parametrize("known_request", [False, True], ids=["cancelled", "active"])
async def test_cache_miss_preserves_metadata_and_invalidates_owning_sender(stage_id, known_request, sender_cache):
    pools = _build_stage_pools([[FakeStageClient(stage_type="llm")] for _ in range(2)])
    orchestrator, queues = _build_bare_orchestrator(pools)
    # Stage 0's sender lives on the frontend; later-stage senders live here.
    if stage_id:
        orchestrator._stage_input_processors[stage_id] = SimpleNamespace(
            renderer=SimpleNamespace(mm_processor_cache=sender_cache)
        )
    if known_request:
        _register_request(orchestrator, "cache-drift", submitted_stage=stage_id)
    raw = EngineCoreOutputs(
        outputs=[
            EngineCoreOutput(
                request_id="cache-drift",
                new_token_ids=[],
                finish_reason=FinishReason.ERROR,
                mm_cache_miss_hashes=["missing"],
            )
        ],
        timestamp=1.0,
    )
    try:
        await orchestrator._process_llm_stage_outputs(stage_id, 0, raw, set())
        if stage_id:
            assert not sender_cache.is_cached_item("missing")
        if known_request or stage_id == 0:
            msg = queues[1].sync_q.get_nowait()
            assert isinstance(msg, ErrorMessage)
            assert msg.mm_cache_miss_hashes == ["missing"]
            if known_request:
                assert (msg.status_code, msg.error_type, msg.fatal) == (503, "ServiceUnavailableError", False)
            else:
                assert msg.status_code is None  # Cache feedback for an already-cancelled request.
            assert "cache" in msg.error.lower()
        assert "cache-drift" not in orchestrator.request_states
        assert sender_cache.is_cached_item("healthy")
    finally:
        for q in queues:
            q.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("collection", ["sync", "queued", "wakeup", "timeout"])
async def test_frontend_cache_invalidated_before_error_delivery(collection, sender_cache):
    engine = object.__new__(AsyncOmniEngine)
    engine.input_processor = SimpleNamespace(renderer=SimpleNamespace(mm_processor_cache=sender_cache))
    engine.output_queue = janus.Queue()
    engine._output_sink = OutputQueueSink(engine.output_queue.sync_q)
    engine.orchestrator_thread = SimpleNamespace(is_alive=lambda: True)
    msg = ErrorMessage(
        request_id="already-cancelled",
        stage_id=0,
        error="Multi-modal processor cache miss.",
        status_code=503,
        error_type="ServiceUnavailableError",
        mm_cache_miss_hashes=["missing"],
    )
    try:
        if collection in ("wakeup", "timeout"):
            pending = asyncio.create_task(engine.get_outputs_async(timeout=0.02))
            await asyncio.sleep(0)
            if collection == "wakeup":
                engine._output_sink.put_nowait(msg)
            else:
                engine.output_queue.sync_q.put_nowait(msg)
            messages = await pending
        else:
            engine._output_sink.put_nowait(msg)
            messages = [engine.try_get_output()] if collection == "sync" else await engine.get_outputs_async()
        assert messages == [msg]
        assert not sender_cache.is_cached_item("missing")
        assert sender_cache.is_cached_item("healthy")
        # A retry sends the actual item again rather than a stale hash-only hit.
        item = MultiModalKwargsItem.dummy()
        assert sender_cache.get_and_update_item((item, []), "missing")[0] is item
    finally:
        engine.output_queue.close()


def test_cache_miss_raises_native_retryable_error():
    base = object.__new__(OmniBase)
    with pytest.raises(RetryableRequestError) as exc_info:
        base._raise_nonfatal_error_message(
            ErrorMessage(
                error="Multi-modal processor cache miss.",
                mm_cache_miss_hashes=["missing"],
                status_code=503,
                error_type="ServiceUnavailableError",
            )
        )
    assert exc_info.value.status_code == 503


def test_cancelled_cache_feedback_does_not_fail_next_offline_request():
    base = object.__new__(OmniBase)
    base.request_states = {}
    handled, *_ = base._handle_output_message(
        ErrorMessage(
            request_id="already-cancelled", stage_id=0, error="cache feedback", mm_cache_miss_hashes=["missing"]
        )
    )
    assert handled is True
