# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Keep preprocessing cache ownership ordered through stage submission."""

import asyncio
import multiprocessing
import threading
import uuid
from types import SimpleNamespace

import pytest
from vllm.config.multimodal import MultiModalConfig
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.cache import LruKeyReplicatedReceiverCache, LruKeyReplicatedSenderCache
from vllm.multimodal.cache.shm import ShmObjectStoreReceiverCache, ShmObjectStoreSenderCache
from vllm.multimodal.processing import InputProcessingContext
from vllm.utils.cache import LRUCache

from vllm_omni.engine.duplex.contracts import DuplexFence, DuplexStageRequestContext, DuplexStageSubmission
from vllm_omni.engine.duplex_orchestrator import DuplexOrchestratorRequestState
from vllm_omni.engine.messages import StageSubmissionMessage
from vllm_omni.engine.orchestrator import OrchestratorRequestState
from vllm_omni.engine.stage_pool import StageUnavailableError
from vllm_omni.inputs.preprocess import OmniRenderer

from .test_duplex_orchestrator import _build as _build_duplex
from .test_orchestrator import (
    FakeStageClient,
    _build_request_output,
    _build_stage_pools,
    _processed_request,
    _sampling_params,
)
from .test_orchestrator_error_handling import _build_bare_orchestrator

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def cache_pipeline():
    prompt = {"prompt": "describe", "multi_modal_data": {"image": ["frame"]}}
    downstream = FakeStageClient(final_output=True, next_inputs=[prompt])
    orch, queues = _build_bare_orchestrator(_build_stage_pools([[FakeStageClient()], [downstream]]))
    model = SimpleNamespace(get_multimodal_config=lambda: SimpleNamespace(mm_processor_cache_gb=0.001))
    sender = LruKeyReplicatedSenderCache(model)
    receiver = LruKeyReplicatedReceiverCache(model)
    media = {"frame-hash": {"pixel_values": [1, 2, 3]}}
    media_keys: dict[str, str] = {}
    processed = []
    received = []

    def process_inputs(*, request_id, params, **kwargs):
        key = media_keys.get(request_id, "frame-hash")
        data, _ = sender.get_and_update_item((media[key], []), key)
        processed.append((request_id, data))
        if request_id == "too-long":
            # Real processors validate length AFTER rendering/cache mutation.
            raise VLLMValidationError("The decoder prompt is longer than the maximum model length.")
        request = _processed_request(request_id, params)
        request.mm_features = [SimpleNamespace(mm_hash=key, identifier=key, data=data)]
        return request

    async def receive(request_id, state, request, **kwargs):
        feature = request.mm_features[0]
        assert receiver.get_and_update_item(feature.data, feature.mm_hash) == media[feature.mm_hash]
        received.append(request_id)
        return 0

    orch._stage_input_processors[1] = SimpleNamespace(
        process_inputs=process_inputs, renderer=SimpleNamespace(mm_processor_cache=sender)
    )
    orch.stage_pools[1].submit_initial = receive

    async def submit(request_id, path):
        params = [_sampling_params(), _sampling_params()]
        if path == "bypass":
            await orch._handle_add_request(
                StageSubmissionMessage(
                    type="add_request",
                    request_id=request_id,
                    prompt=prompt,
                    original_prompt=prompt,
                    sampling_params_list=params,
                    final_stage_id=1,
                    output_prompt_text=None,
                    preprocess_ms=0.0,
                    request_timestamp=0.0,
                    enqueue_ts=0.0,
                    entry_stage_id=1,
                )
            )
        else:
            state = OrchestratorRequestState(
                request_id=request_id,
                prompt=prompt,
                sampling_params_list=params,
                final_stage_id=1,
            )
            orch.request_states[request_id] = state
            await orch._forward_to_next_stage_unguarded(request_id, 0, _build_request_output(request_id), state)

    yield SimpleNamespace(
        orch=orch,
        submit=submit,
        processed=processed,
        received=received,
        sender=sender,
        receiver=receiver,
        media=media,
        media_keys=media_keys,
    )
    for q in queues:
        q.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["forward", "bypass"])
async def test_rejected_prompt_does_not_poison_later_media(cache_pipeline, path):
    p = cache_pipeline
    await p.submit("too-long", path)
    assert "too-long" not in p.orch.request_states
    await p.submit("valid", path)
    await p.submit("cached", path)
    assert p.received == ["valid", "cached"]
    assert p.processed[1][1] is not None
    assert p.processed[2][1] is None, "successful submission must retain IPC cache hits"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["forward", "bypass"])
async def test_failed_submission_does_not_leave_unsent_cache_entries(cache_pipeline, path):
    p = cache_pipeline
    original = p.orch.stage_pools[1].submit_initial

    async def fail_before_send(request_id, *args, **kwargs):
        if request_id == "failed":
            raise RuntimeError("transport failed before send")
        return await original(request_id, *args, **kwargs)

    p.orch.stage_pools[1].submit_initial = fail_before_send
    with pytest.raises(RuntimeError, match="transport failed before send"):
        await p.submit("failed", path)
    await p.submit("valid", path)
    assert p.received == ["valid"]
    assert p.processed[-1][1] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("path,failure", [("forward", "cancel"), ("bypass", "cancel"), ("bypass", "dispatch")])
async def test_unsent_cached_item_does_not_desynchronize_lru_eviction(cache_pipeline, monkeypatch, path, failure):
    p = cache_pipeline
    # Equal-size entries and a two-item budget expose sender/receiver eviction drift.
    p.sender._cache = LRUCache(2)
    p.receiver._cache = LRUCache(2)
    p.media.update({key: {"pixel_values": [key]} for key in ("A", "B", "C")})
    p.media_keys.update({"first": "A", "second": "B", "unsent": "B", "third": "C", "reuse": "A"})
    await p.submit("first", path)
    await p.submit("second", path)
    if failure == "cancel":
        builder_name = "_build_next_stage_request" if path == "forward" else "_build_entry_stage_request"
        original = getattr(p.orch, builder_name)

        async def cancel_after_processing(request_id, *args, **kwargs):
            request = await original(request_id, *args, **kwargs)
            if request_id == "unsent":
                p.orch.request_states.pop(request_id)
            return request

        monkeypatch.setattr(p.orch, builder_name, cancel_after_processing)
    else:
        original_submit = p.orch.stage_pools[1].submit_initial

        async def reject_before_send(request_id, *args, **kwargs):
            if request_id == "unsent":
                raise StageUnavailableError("selected replica detached before send")
            return await original_submit(request_id, *args, **kwargs)

        monkeypatch.setattr(p.orch.stage_pools[1], "submit_initial", reject_before_send)
    await p.submit("unsent", path)
    assert p.processed[-1][1] is None, "the cancelled request must have hit a previously sent item"
    await p.submit("third", path)
    await p.submit("reuse", path)
    assert p.received == ["first", "second", "third", "reuse"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [ValueError, asyncio.CancelledError])
async def test_rollback_preserves_outstanding_shm_media(cache_pipeline, monkeypatch, failure):
    monkeypatch.setenv("VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME", f"pr8031-{uuid.uuid4().hex}")
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(world_size=1),
        model_config=SimpleNamespace(
            get_multimodal_config=lambda: SimpleNamespace(
                mm_processor_cache_gb=0.001, mm_shm_cache_max_object_size_mb=1
            )
        ),
    )
    sender = ShmObjectStoreSenderCache(config)
    receiver = ShmObjectStoreReceiverCache(config, multiprocessing.Lock())
    monkeypatch.setattr(cache_pipeline.orch._stage_input_processors[1].renderer, "mm_processor_cache", sender)
    try:
        original = {"pixel_values": [1, 2, 3]}
        outstanding, _ = sender.get_and_update_item((original, []), "A")
        assert "address" in outstanding
        with pytest.raises(failure):
            async with cache_pipeline.orch._stage_input_submission(1):
                sender.get_and_update_item(({"pixel_values": [4, 5, 6]}, []), "B")
                raise failure()
        sender.get_and_update_item(({"pixel_values": [7, 8, 9]}, []), "C")
        assert receiver.get_and_update_item(outstanding, "A") == original
    finally:
        receiver._shm_cache.close()
        sender.close()


@pytest.fixture
def shm_pipeline(cache_pipeline, monkeypatch, request):
    p = cache_pipeline
    monkeypatch.setenv("VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME", f"pr8031-{uuid.uuid4().hex}")
    readers = request.param
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(world_size=readers),
        model_config=SimpleNamespace(
            get_multimodal_config=lambda: SimpleNamespace(
                mm_processor_cache_gb=0.001, mm_shm_cache_max_object_size_mb=1
            )
        ),
    )
    p.sender = ShmObjectStoreSenderCache(config)
    p.receiver = ShmObjectStoreReceiverCache(config, multiprocessing.Lock())
    processor = p.orch._stage_input_processors[1]
    monkeypatch.setattr(processor.renderer, "mm_processor_cache", p.sender)

    def process_inputs(*, request_id, params, **kwargs):
        key = p.media_keys.get(request_id, request_id)
        media = {"pixel_values": bytes(400 * 1024)}
        p.sender.touch_sender_cache_item(key)
        data, _ = p.sender.get_and_update_item((media, []), key)
        p.processed.append((request_id, data))
        if request_id == "too-long":
            raise VLLMValidationError("prompt too long")
        if request_id == "cancelled":
            p.orch.request_states.pop(request_id)
        result = _processed_request(request_id, params)
        result.mm_features = [SimpleNamespace(mm_hash=key, identifier=key, data=data)]
        return result

    async def receive(request_id, state, result, **kwargs):
        feature = result.mm_features[0]
        for _ in range(readers):
            received = SimpleNamespace(mm_hash=feature.mm_hash, identifier=feature.identifier, data=feature.data)
            p.receiver.get_and_update_features([received])
            assert received.data == {"pixel_values": bytes(400 * 1024)}
        p.received.append(request_id)
        return 0

    monkeypatch.setattr(processor, "process_inputs", process_inputs)
    monkeypatch.setattr(p.orch.stage_pools[1], "submit_initial", receive)
    try:
        yield p
    finally:
        p.receiver._shm_cache.close()
        p.sender.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("shm_pipeline", [1, 2], indirect=True)
@pytest.mark.parametrize("path", ["bypass", "forward"])
@pytest.mark.parametrize("failure", ["too-long", "cancelled"])
@pytest.mark.parametrize("retry_same_image", [False, True])
async def test_unsent_shm_references_do_not_exhaust_cache(shm_pipeline, path, failure, retry_same_image):
    p = shm_pipeline
    await p.submit(failure, path)
    if retry_same_image:
        p.media_keys["valid-0"] = failure
    for i in range(5):
        await p.submit(f"valid-{i}", path)
        assert "address" in p.processed[-1][1], "unsent references must not force full-payload fallback"
    assert p.received == [f"valid-{i}" for i in range(5)]


@pytest.mark.asyncio
@pytest.mark.parametrize("shm_pipeline", [1, 2], indirect=True)
@pytest.mark.parametrize("path", ["bypass", "forward"])
async def test_shm_failed_retries_preserve_unread_older_media(shm_pipeline, path):
    p = shm_pipeline
    original = {"pixel_values": bytes(400 * 1024)}
    p.sender.touch_sender_cache_item("older")
    outstanding, _ = p.sender.get_and_update_item((original, []), "older")
    storage = p.sender._shm_cache
    older_id = outstanding["monotonic_id"].data
    before = storage.writer_flag[older_id]

    # The unread older allocation blocks FIFO eviction of the failed one.
    await p.submit("too-long", path)
    await p.submit("too-long", path)
    assert storage.writer_flag[older_id] == before
    p.media_keys["retry"] = "too-long"
    await p.submit("retry", path)
    assert "address" in p.processed[-1][1]
    for _ in range(storage.n_readers):
        feature = SimpleNamespace(mm_hash="older", identifier="older", data=outstanding)
        p.receiver.get_and_update_features([feature])
        assert feature.data == original
    for i in range(5):
        await p.submit(f"valid-{i}", path)
        assert "address" in p.processed[-1][1]


@pytest.mark.asyncio
@pytest.mark.parametrize("shm_pipeline", [1], indirect=True)
async def test_shm_rollback_preserves_prior_refs_and_removes_hits_and_touches(shm_pipeline):
    p = shm_pipeline
    original = {"pixel_values": [1, 2, 3]}
    outstanding, _ = p.sender.get_and_update_item((original, []), "A")
    refs = p.sender._shm_cache.writer_flag
    before = refs.copy()
    with pytest.raises(ValueError):
        async with p.orch._stage_input_submission(1):
            p.sender.get_and_update_item((None, []), "A")
            p.sender.touch_sender_cache_item("A")
            raise ValueError("rejected")
    assert refs == before
    assert p.receiver.get_and_update_item(outstanding, "A") == original


@pytest.mark.asyncio
@pytest.mark.parametrize("shm_pipeline", [1], indirect=True)
@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_shm_submission_uncertainty_keeps_receiver_references(shm_pipeline, failure):
    p = shm_pipeline
    original = {"pixel_values": [1, 2, 3]}
    with pytest.raises(failure):
        async with p.orch._stage_input_submission(1) as mark_submitting:
            outstanding, _ = p.sender.get_and_update_item((original, []), "A")
            before = p.sender._shm_cache.writer_flag.copy()
            mark_submitting()
            raise failure()
    assert p.sender._shm_cache.writer_flag == before
    assert p.receiver.get_and_update_item(outstanding, "A") == original


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["bypass", "forward"])
@pytest.mark.parametrize("options", [{"size": 123}, {"fps": []}, {"num_frames": []}])
async def test_malformed_video_options_fail_only_the_request(cache_pipeline, monkeypatch, mocker, path, options):
    p = cache_pipeline
    processor = p.orch._stage_input_processors[1]
    renderer = processor.renderer
    renderer.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_vl"))
    ctx = InputProcessingContext(SimpleNamespace(get_multimodal_config=lambda: MultiModalConfig()), tokenizer=None)
    renderer.get_mm_processor = mocker.Mock(
        return_value=mocker.Mock(
            info=SimpleNamespace(
                ctx=ctx,
                _merge_and_resolve_mm_processor_kwargs=lambda kw: ctx.get_merged_mm_kwargs(
                    kw, supported_mm_processor_kwargs={"videos_kwargs": {"size", "fps", "num_frames"}}
                ),
            )
        )
    )
    original = processor.process_inputs

    def process_inputs(**kwargs):
        if kwargs["request_id"] == "bad-video":
            OmniRenderer._validate_video_processor_kwargs(
                renderer, {"multi_modal_data": {"video": ["frames"]}, "mm_processor_kwargs": options}
            )
        return original(**kwargs)

    monkeypatch.setattr(processor, "process_inputs", process_inputs)
    await p.submit("bad-video", path)
    assert "bad-video" not in p.orch.request_states
    error = p.orch.output_async_queue.get_nowait()
    assert error.request_id == "bad-video" and error.status_code == 400 and not error.fatal
    await p.submit("valid", path)
    assert p.received == ["valid"]


@pytest.mark.asyncio
async def test_unrelated_processing_type_error_still_propagates(cache_pipeline, monkeypatch, mocker):
    p = cache_pipeline
    monkeypatch.setattr(
        p.orch._stage_input_processors[1], "process_inputs", mocker.Mock(side_effect=TypeError("internal bug"))
    )
    with pytest.raises(TypeError, match="internal bug"):
        await p.submit("broken", "bypass")


@pytest.mark.asyncio
@pytest.mark.parametrize("first_path", ["forward", "bypass"])
async def test_cancel_after_processing_precedes_next_cache_hit(cache_pipeline, monkeypatch, first_path):
    p = cache_pipeline
    built = asyncio.Event()
    resume = asyncio.Event()
    builder_name = "_build_next_stage_request" if first_path == "forward" else "_build_entry_stage_request"
    original = getattr(p.orch, builder_name)

    async def build(request_id, *args, **kwargs):
        request = await original(request_id, *args, **kwargs)
        if request_id == "cancelled":
            built.set()
            await resume.wait()
        return request

    async def inline(func, *args, **kwargs):
        # Select the post-worker interleaving without relying on thread timing.
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", inline)
    monkeypatch.setattr(p.orch, builder_name, build)
    first = asyncio.create_task(p.submit("cancelled", first_path))
    await asyncio.wait_for(built.wait(), 2)
    p.orch.request_states.pop("cancelled")
    second = asyncio.create_task(p.submit("valid", "bypass" if first_path == "forward" else "forward"))
    try:
        await asyncio.sleep(0)
        assert [request_id for request_id, _ in p.processed] == ["cancelled"]
    finally:
        resume.set()
        await asyncio.gather(first, second)
    assert p.received == ["valid"]
    assert p.processed[-1][1] is not None


@pytest.mark.asyncio
async def test_submission_is_ordered_before_the_next_sender_cache_hit(cache_pipeline, monkeypatch):
    p = cache_pipeline
    entered = asyncio.Event()
    release = asyncio.Event()
    original = p.orch.stage_pools[1].submit_initial

    async def inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", inline)

    async def delayed_submit(request_id, *args, **kwargs):
        if request_id == "first":
            entered.set()
            await release.wait()
        return await original(request_id, *args, **kwargs)

    p.orch.stage_pools[1].submit_initial = delayed_submit
    first = asyncio.create_task(p.submit("first", "forward"))
    await asyncio.wait_for(entered.wait(), 2)
    second = asyncio.create_task(p.submit("second", "bypass"))
    try:
        await asyncio.sleep(0)
        assert [request_id for request_id, _ in p.processed] == ["first"]
    finally:
        release.set()
        await asyncio.gather(first, second)
    assert p.received == ["first", "second"]
    assert p.processed[-1][1] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("before_lock", [False, True])
async def test_task_cancellation_waits_for_preprocessing_worker(cache_pipeline, monkeypatch, before_lock):
    p = cache_pipeline
    started = threading.Event()
    release = threading.Event()
    processor = p.orch._stage_input_processors[1]
    original = processor.process_inputs

    def delayed_process(**kwargs):
        if kwargs["request_id"] == "cancelled":
            started.set()
            assert release.wait(5)
        return original(**kwargs)

    if before_lock:
        original_lock = p.orch._prompt_processing_lock

        class DelayedLock:
            delay_next_acquire = True

            def __enter__(self):
                if self.delay_next_acquire:
                    self.delay_next_acquire = False
                    started.set()
                    assert release.wait(5)
                return original_lock.__enter__()

            def __exit__(self, *args):
                return original_lock.__exit__(*args)

        monkeypatch.setattr(p.orch, "_prompt_processing_lock", DelayedLock())
    else:
        processor.process_inputs = delayed_process
    first = asyncio.create_task(p.submit("cancelled", "forward"))
    assert await asyncio.to_thread(started.wait, 2)
    first.cancel()
    second = asyncio.create_task(p.submit("valid", "bypass"))
    try:
        await asyncio.sleep(0)
        first.cancel()  # Repeated cancellation must not release cache ownership.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(first), 0.05)
        assert not p.processed
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
    assert p.received == ["valid"]
    assert p.processed[-1][1] is not None


@pytest.mark.asyncio
async def test_duplex_cancel_after_processing_does_not_poison_next_turn(cache_pipeline, monkeypatch):
    p = cache_pipeline
    orch, *_ = _build_duplex(stages=2)
    orch.stage_pools = p.orch.stage_pools
    orch._stage_input_processors = p.orch._stage_input_processors
    built = asyncio.Event()
    release = asyncio.Event()
    original = orch._build_next_stage_request

    async def build(request_id, *args, **kwargs):
        request = await original(request_id, *args, **kwargs)
        if request_id == "cancelled":
            built.set()
            await release.wait()
        return request

    async def inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", inline)
    monkeypatch.setattr(orch, "_build_next_stage_request", build)

    async def submit(request_id):
        context = DuplexStageRequestContext(
            request_id=request_id,
            session_id="session",
            fence=DuplexFence("session"),
            stage_id=1,
            final_stage_id=1,
            config_generation=0,
            sampling_params=(_sampling_params(), _sampling_params()),
        )
        orch.request_states[request_id] = DuplexOrchestratorRequestState(
            request_id=request_id,
            sampling_params_list=list(context.sampling_params),
            final_stage_id=1,
        )
        return await orch.submit(
            DuplexStageSubmission(
                context=context,
                prompt={"prompt": "describe", "multi_modal_data": {"image": ["frame"]}},
                already_submitted=False,
                resumable=False,
            )
        )

    first = asyncio.create_task(submit("cancelled"))
    await asyncio.wait_for(built.wait(), 2)
    orch.request_states.pop("cancelled")
    second = asyncio.create_task(submit("valid"))
    try:
        await asyncio.sleep(0)
        assert [request_id for request_id, _ in p.processed] == ["cancelled"]
    finally:
        release.set()
        with pytest.raises(RuntimeError, match="cancelled during input preprocessing"):
            await first
        await second
    assert p.received == ["valid"]
    assert p.processed[-1][1] is not None
