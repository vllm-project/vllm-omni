# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Qwen selection is checked after real preparation, locks and local routing."""

import asyncio
import threading
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.engine.duplex.test_memory_card_ledger import _admit, _operation
from tests.engine.duplex.test_qwen_context_memory_runtime import _Processor
from tests.engine.test_duplex_orchestrator import SESSION_ID, _build, _open, _settle, _submit
from tests.engine.test_orchestrator import FakeRunningCounter, _processed_request
from tests.engine.test_orchestrator_input_cache import cache_pipeline as _cache_fixture
from tests.engine.test_orchestrator_input_cache import shm_pipeline as _shm_fixture
from tests.model_executor.models.qwen3_omni.test_context_budget_processor import processor as _processor_fixture
from vllm_omni.config.context_memory import DuplexAuxiliaryBudgetConfig, DuplexContextMemoryConfig
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.commands import CreateItem, CreateResponse
from vllm_omni.engine.duplex.contracts import DuplexContextSelection
from vllm_omni.engine.duplex.messages import DuplexSessionCommandMessage
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin
from vllm_omni.protocol.realtime.items import normalize_conversation_item

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.asyncio]
KEY = "user/self/preferred_tea"
qwen_processor = _processor_fixture
cache_pipeline = _cache_fixture
shm_pipeline = _shm_fixture


class _Tokenizer:
    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(character) for character in text]

    def __call__(self, *, text, **kwargs):
        return {"input_ids": [self.encode(value) for value in text]}


class _CountedProcessor(_Processor):
    tokenizer = _Tokenizer()


@asynccontextmanager
async def _case(monkeypatch, *, cap=256, processor=None, summary=False, heard=False):
    runtime = DuplexSessionRuntimeConfig(
        qwen_context_policy="budget_summary" if summary else "budget",
        qwen_context_max_tokens=650 if summary else 4096,
        qwen_summary_max_tokens=128,
        context_auxiliary_budget=DuplexAuxiliaryBudgetConfig(
            max_input_tokens=8192, max_output_tokens=1024, window_s=60, cooldown_s=0, idle_s=100
        )
        if summary
        else None,
        context_memory=DuplexContextMemoryConfig(idle_s=100, cooldown_s=0),
        context_memory_tokens=cap,
        collect_stats=True,
    )
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin.processor = processor or _CountedProcessor()
    counter = FakeRunningCounter()
    orch, clients, rpc, output = _build(plugin=plugin, runtime_config=runtime, running_counter=counter)
    orch.session_manager.model_config = SimpleNamespace(max_model_len=4096, hf_config=SimpleNamespace())
    env = SimpleNamespace(
        orch=orch,
        clients=clients,
        rpc=rpc,
        output=output,
        plugin=plugin,
        counter=counter,
        preparations=[],
        processed=[],
        selections=[],
        accepts=[],
        submitted_callbacks=[],
        actual_limit=None,
    )

    def process_inputs(*, request_id, prompt, params, **kwargs):
        env.processed.append(prompt["prompt"])
        request = _processed_request(request_id, params)
        request.prompt_token_ids = plugin.processor.tokenizer.encode(prompt["prompt"])
        if env.actual_limit is not None:
            request.prompt_token_ids = [1] * (
                env.actual_limit(params) if callable(env.actual_limit) else env.actual_limit
            )
        return request

    env.input_processor = SimpleNamespace(
        process_inputs=process_inputs, renderer=SimpleNamespace(mm_processor_cache=None)
    )
    orch._stage_input_processors[0] = env.input_processor
    original_prepare = plugin.prepare_append_plan

    async def prepare(**kwargs):
        # A formatter gets detached candidates, never an owner scope or guard.
        candidates = kwargs["session_config"].get("context_memory_cards", ())
        assert all(not hasattr(card, "scope") and not callable(card) for card in candidates)
        env.preparations.append(kwargs)
        plan = await original_prepare(**kwargs)
        env.selections.append(plan.context_selection)
        return plan

    monkeypatch.setattr(plugin, "prepare_append_plan", prepare)
    original_submit = orch.submit

    async def submit(submission, *, before_accept=None):
        env.submitted_callbacks.append(before_accept)
        return await original_submit(submission, before_accept=before_accept)

    monkeypatch.setattr(orch, "submit", submit)
    try:
        opened = await _open(orch, rpc, extra_body={"auto_response": False})
        assert opened.ok, opened.error
        env.session = orch.session_manager.get(SESSION_ID)
        env.runner = orch.session_manager.runners[SESSION_ID]
        env.events = orch.session_manager._outputs[SESSION_ID]
        env.session.register_history_item(
            "fact", {"role": "user", "content": "I prefer OOLONG tea." + (" detail" * 200 if summary else "")}
        )
        env.session.register_history_item("answer", {"role": "assistant", "content": "Recorded."})
        env.session.register_history_item(
            "later-fact", {"role": "user", "content": "GREEN tea is the later preference."}
        )
        env.session.register_history_item("later-answer", {"role": "assistant", "content": "Recorded."})
        result = _admit(env.session, _operation(env.session, "fact", "The user prefers OOLONG tea.", key=KEY))
        assert result.status == "applied"
        env.node_id = result.node_ids[0]
        if summary:
            from vllm_omni.engine.duplex.session.context_ledger import DuplexContextLevel, source_refs
            from vllm_omni.model_executor.models.qwen3_omni.duplex.context_budget import ContextSummary

            snapshot = env.session.history_snapshot()
            refs = source_refs(snapshot, *snapshot.source_ids[:2])
            node = env.session.context_ledger.add_node(
                snapshot,
                level=DuplexContextLevel.SUMMARY,
                sources=refs,
                message={"role": "system", "content": "Earlier tea preference."},
            )
            env.summary_id = node.node_id
            env.runner.model_state.context_memory.summary = ContextSummary(
                "Earlier tea preference.", refs, node.node_id
            )
        if heard:
            from tests.engine.duplex.test_memory_card_auxiliary import _spoken

            config = env.session.config
            env.spoken_id = _spoken(env.session)
            assert env.session.apply_heard_text(env.spoken_id, text="hello world", audio_end_ms=900, char_end=4)
            env.session.replace_config(
                replace(config, playback_commit_policy=env.session.config.playback_commit_policy)
            )
            # The helper leaves half the old audio queued. Free its current
            # playback gate while retaining the per-response calibration record.
            env.session.clear_playback_cursor()
        await _submit(
            orch,
            CreateItem(
                item=normalize_conversation_item(
                    {
                        "id": "query",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Which tea do I prefer?"}],
                    }
                )
            ),
        )
        env.before = env.session.history_snapshot()
        yield env
    finally:
        await orch.session_manager.shutdown()


def _replace_card(env):
    before = env.session.history_snapshot()
    result = _admit(
        env.session,
        _operation(
            env.session,
            "later-fact",
            "The user prefers GREEN tea.",
            key=KEY,
            generation=1,
        ),
    )
    assert result.status == "applied"
    assert env.session.history_snapshot() == before, "card replacement must not edit prompt history"


async def _start(env):
    await env.orch._dispatch_message(DuplexSessionCommandMessage(session_id=SESSION_ID, command=CreateResponse()))


def _accepted(env):
    return [request for client in env.clients for request, *_ in client.add_request_calls]


def _rendered(request):
    return "".join(chr(token) for token in request.prompt_token_ids)


async def _error_codes(env):
    codes = set()
    while env.events.pending_events:
        event = await env.events.get()
        if event is not None and event.type == "error":
            codes.add(event.code)
    return codes


async def test_selected_card_reaches_actual_stage_submission(monkeypatch):
    async with _case(monkeypatch) as env:
        await _start(env)
        await _settle(env.orch)
        assert len(_accepted(env)) == 1
        assert "[Conversation cards:" in _rendered(_accepted(env)[0])
        assert env.selections == [DuplexContextSelection((env.node_id,), None)]
        assert len(env.submitted_callbacks) == 1 and callable(env.submitted_callbacks[0])
        assert not hasattr(_accepted(env)[0], "before_accept")
        assert env.session.input_seq == env.session.input_turn_seq == 1
        assert env.counter.value == 1


@pytest.mark.parametrize("wait_point", ["formatter", "processor", "input_lock", "routing"])
async def test_card_only_change_after_wait_rebuilds_once(monkeypatch, wait_point):
    async with _case(monkeypatch) as env:
        entered, release = asyncio.Event(), asyncio.Event()
        worker_release = threading.Event()
        lock = None
        if wait_point == "formatter":
            original = env.plugin.prepare_append_plan
            calls = 0

            async def prepare(**kwargs):
                nonlocal calls
                calls += 1
                plan = await original(**kwargs)
                if calls == 1:
                    entered.set()
                    await release.wait()
                return plan

            monkeypatch.setattr(env.plugin, "prepare_append_plan", prepare)
        elif wait_point == "processor":
            original = env.input_processor.process_inputs
            loop = asyncio.get_running_loop()
            calls = 0

            def process_inputs(**kwargs):
                nonlocal calls
                calls += 1
                result = original(**kwargs)
                if calls == 1:
                    loop.call_soon_threadsafe(entered.set)
                    assert worker_release.wait(5)
                return result

            env.input_processor.process_inputs = process_inputs
        elif wait_point == "input_lock":
            lock = env.orch._stage_input_submission_locks.setdefault(0, asyncio.Lock())
            await lock.acquire()
            original = env.orch._stage_input_submission

            @asynccontextmanager
            async def submission(stage_id):
                entered.set()
                async with original(stage_id) as mark:
                    yield mark

            monkeypatch.setattr(env.orch, "_stage_input_submission", submission)
        else:
            pool = env.orch.stage_pools[0]
            original = pool._pick_or_select
            calls = 0

            async def select(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    entered.set()
                    await release.wait()
                return await original(*args, **kwargs)

            monkeypatch.setattr(pool, "_pick_or_select", select)
        commits = []
        original_commit = env.session.commit_append

        def commit(reservation):
            commits.append(reservation)
            return original_commit(reservation)

        monkeypatch.setattr(env.session, "commit_append", commit)
        try:
            await _start(env)
            await asyncio.wait_for(entered.wait(), 5)
            assert not _accepted(env)
            _replace_card(env)
            if lock is not None:
                lock.release()
                lock = None
            release.set()
            worker_release.set()
            await _settle(env.orch)
            assert len(env.preparations) == 2
            assert env.selections[0].card_node_ids == (env.node_id,)
            assert env.selections[1].card_node_ids == ()
            assert len(_accepted(env)) == 1
            assert "[Conversation cards:" not in _rendered(_accepted(env)[0])
            assert len(commits) == 1 and env.session.input_seq == env.session.input_turn_seq == 1
            assert env.counter.value == 1
            assert not env.orch._stage_input_submission_locks[0].locked()
            chunks = [chunk for turn in env.session.perf.snapshot()["turns"] for chunk in turn["chunks"]]
            assert len(chunks) == 1
        finally:
            if lock is not None:
                lock.release()
            release.set()
            worker_release.set()


@pytest.mark.parametrize("change", ["edit", "delete", "history", "epoch", "config", "close", "request"])
async def test_base_context_change_is_never_downgraded_to_a_card_retry(monkeypatch, change):
    async with _case(monkeypatch) as env:
        pool = env.orch.stage_pools[0]
        entered, release = asyncio.Event(), asyncio.Event()
        original = pool._pick_or_select

        async def select(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)

        monkeypatch.setattr(pool, "_pick_or_select", select)
        try:
            await _start(env)
            await asyncio.wait_for(entered.wait(), 5)
            if change == "edit":
                env.session.register_history_item("fact", {"role": "user", "content": "Changed tea."})
            elif change == "delete":
                assert env.session.delete_history_item("fact")
            elif change == "history":
                env.session.register_history_item("unrelated", {"role": "system", "content": "New instruction."})
            elif change == "epoch":
                env.session.epoch += 1
            elif change == "config":
                env.session.replace_config(replace(env.session.config, instructions="Changed instructions."))
            elif change == "close":
                env.session.begin_close(reason="test")
            else:
                request_id = next(iter(env.orch.request_states))
                context = env.session.request_resources[(0, request_id)]
                env.session.request_resources[(0, request_id)] = replace(context)
            release.set()
            await _settle(env.orch)
            assert not _accepted(env)
            assert len(env.preparations) == 1
            assert env.session.input_seq == env.session.input_turn_seq == 0
            assert env.counter.value == 0
            if change in ("edit", "delete", "history", "config", "request"):
                assert "context_changed" in await _error_codes(env)
        finally:
            release.set()


async def test_actual_materialized_tokens_are_checked_with_output_reserve(monkeypatch):
    async with _case(monkeypatch) as env:
        env.actual_limit = 4096
        await _start(env)
        await _settle(env.orch)
        assert not _accepted(env)
        assert len(env.preparations) == 1
        assert env.session.input_seq == 0 and env.counter.value == 0
        assert "context_budget_exceeded" in await _error_codes(env)


async def test_disabled_card_consumer_keeps_the_original_submission_path(monkeypatch):
    async with _case(monkeypatch, cap=0) as env:
        await _start(env)
        await _settle(env.orch)
        assert len(_accepted(env)) == 1
        assert env.selections == [None]
        assert env.submitted_callbacks == [None]


async def test_pinned_processor_card_reaches_the_actual_stage_port(monkeypatch, qwen_processor):
    async with _case(monkeypatch, processor=qwen_processor) as env:
        await _start(env)
        await _settle(env.orch)
        assert len(_accepted(env)) == 1
        actual = _accepted(env)[0]
        expected = qwen_processor(text=env.processed[-1], return_tensors="np")["input_ids"][0].tolist()
        assert actual.prompt_token_ids == expected
        assert "[Conversation cards:" in env.processed[-1]
        assert env.selections[0].card_node_ids == (env.node_id,)
        assert callable(env.submitted_callbacks[0])


@pytest.mark.parametrize("change", ["summary_discard", "heard_prefix"])
async def test_changed_summary_or_heard_prefix_is_not_a_card_only_retry(monkeypatch, change):
    async with _case(monkeypatch, summary=change == "summary_discard", heard=change == "heard_prefix") as env:
        if change == "summary_discard":
            from vllm_omni.engine.duplex.memory_types import DuplexMemoryProposalBatch

            # This case isolates late summary invalidation. Its seeded card
            # needs completed coverage before testing the stage-accept fence;
            # otherwise the earlier freshness guard correctly rebuilds first.
            for _ in range(8):
                context = env.session.capture_memory_context()
                if context is None:
                    break
                result = env.session.admit_memory_result(context=context, proposal=DuplexMemoryProposalBatch(()))
                assert result.status == "no_change"
            else:
                pytest.fail("summary fixture coverage did not complete")
        entered, release = asyncio.Event(), asyncio.Event()
        pool = env.orch.stage_pools[0]
        original = pool._pick_or_select

        async def select(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)

        monkeypatch.setattr(pool, "_pick_or_select", select)
        try:
            await _start(env)
            try:
                await asyncio.wait_for(entered.wait(), 5)
            except asyncio.TimeoutError as exc:
                raise AssertionError(
                    {
                        "error_codes": sorted(await _error_codes(env)),
                        "preparations": len(env.preparations),
                        "processed": len(env.processed),
                        "active_response": env.session.active_response_id is not None,
                        "turn_state": str(env.session.turn_state),
                        "unanswered": env.session.unanswered_user_items(),
                    }
                ) from exc
            if change == "summary_discard":
                assert env.selections[0].summary_node_id == env.summary_id
                env.session.context_ledger.discard(env.summary_id)
            else:
                assert env.session.apply_heard_text(env.spoken_id, text="hello world", audio_end_ms=1000, char_end=4)
            release.set()
            await _settle(env.orch)
            assert not _accepted(env)
            assert len(env.preparations) == 1 and env.session.input_seq == 0
        finally:
            release.set()


@pytest.mark.parametrize("after_capture", [False, True])
async def test_reconciliation_allows_valid_no_card_foreground(monkeypatch, after_capture):
    async with _case(monkeypatch) as env:
        original = env.plugin.prepare_append_plan
        calls = 0

        async def prepare(**kwargs):
            nonlocal calls
            calls += 1
            plan = await original(**kwargs)
            if after_capture and calls == 1:
                env.session.restart_memory_reconciliation()
            return plan

        monkeypatch.setattr(env.plugin, "prepare_append_plan", prepare)
        if not after_capture:
            env.session.restart_memory_reconciliation()
        await _start(env)
        await _settle(env.orch)
        assert len(_accepted(env)) == 1
        assert "[Conversation cards:" not in _rendered(_accepted(env)[0])
        assert len(env.preparations) == (2 if after_capture else 1)
        assert env.selections[-1].card_node_ids == ()


async def test_a_second_context_change_during_rebuild_is_rejected(monkeypatch):
    async with _case(monkeypatch) as env:
        original = env.plugin.prepare_append_plan
        calls = 0

        async def prepare(**kwargs):
            nonlocal calls
            calls += 1
            plan = await original(**kwargs)
            if calls == 1:
                _replace_card(env)
            else:
                env.session.register_history_item("change", {"role": "system", "content": "New base context."})
            return plan

        monkeypatch.setattr(env.plugin, "prepare_append_plan", prepare)
        await _start(env)
        await _settle(env.orch)
        assert not _accepted(env)
        assert len(env.preparations) == 2
        assert env.session.input_seq == 0 and env.counter.value == 0


async def test_cleanup_failure_prevents_a_no_card_retry(monkeypatch):
    async with _case(monkeypatch) as env:
        original_prepare = env.plugin.prepare_append_plan
        original_cleanup = env.orch.cleanup
        cleanup_calls = []

        async def prepare(**kwargs):
            plan = await original_prepare(**kwargs)
            _replace_card(env)
            return plan

        async def cleanup(*args, **kwargs):
            cleanup_calls.append(args)
            if len(cleanup_calls) == 1:
                raise RuntimeError("controlled cleanup failure")
            return await original_cleanup(*args, **kwargs)

        monkeypatch.setattr(env.plugin, "prepare_append_plan", prepare)
        monkeypatch.setattr(env.orch, "cleanup", cleanup)
        await _start(env)
        await _settle(env.orch)
        assert cleanup_calls and not _accepted(env)
        assert len(env.preparations) == 1 and env.session.input_seq == 0


@pytest.mark.parametrize("error_kind", ["transport", "card_shaped"])
async def test_uncertain_transport_delivery_is_not_retried(monkeypatch, error_kind):
    async with _case(monkeypatch) as env:
        client = env.clients[0]
        original = client.add_request_async

        async def send(*args, **kwargs):
            await original(*args, **kwargs)
            if error_kind == "card_shaped":
                from vllm_omni.engine.duplex.session.model_channel import _CardSelectionChangedError

                raise _CardSelectionChangedError()
            raise RuntimeError("controlled uncertain transport outcome")

        monkeypatch.setattr(client, "add_request_async", send)
        await _start(env)
        await _settle(env.orch)
        assert len(_accepted(env)) == 1
        assert len(env.preparations) == 1 and env.session.input_seq == 0


@pytest.mark.parametrize("over", [0, 1])
async def test_actual_sampling_reserve_has_an_exact_acceptance_boundary(monkeypatch, over):
    async with _case(monkeypatch) as env:
        env.actual_limit = lambda params: 4096 - params.max_tokens + over
        await _start(env)
        await _settle(env.orch)
        assert len(_accepted(env)) == (0 if over else 1)
        assert env.session.input_seq == (0 if over else 1)


async def test_fallback_applies_only_the_accepted_tool_and_runtime_patch(monkeypatch):
    async with _case(monkeypatch) as env:
        pair = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "tea-call", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "tea-call", "content": "Tea lookup complete."},
        ]
        env.session.replace_runtime_config(
            {
                **env.session.runtime_config,
                "qwen_tool_turns": [{"history_len": 4, "messages": pair}],
            }
        )
        pool = env.orch.stage_pools[0]
        original_select = pool._pick_or_select
        selections = 0
        original_patch = env.plugin.runtime_patch_for_submitted_prompt
        original_replace = env.session.replace_runtime_config
        patches: list[dict[str, object]] = []
        writes: list[dict[str, object]] = []

        async def select(*args, **kwargs):
            nonlocal selections
            selections += 1
            if selections == 1:
                _replace_card(env)
            return await original_select(*args, **kwargs)

        def patch(prompt):
            value = {**original_patch(prompt), "_test_accepted_patch": len(patches) + 1}
            patches.append(value)
            return value

        def write(runtime):
            if "_test_accepted_patch" in runtime:
                writes.append(runtime)
            return original_replace(runtime)

        monkeypatch.setattr(pool, "_pick_or_select", select)
        monkeypatch.setattr(env.plugin, "runtime_patch_for_submitted_prompt", patch)
        monkeypatch.setattr(env.session, "replace_runtime_config", write)
        await _start(env)
        await _settle(env.orch)
        assert len(env.preparations) == 2 and len(_accepted(env)) == 1
        assert env.session.input_seq == 1
        # bind_inflight_response may reconcile the accepted patch; every write
        # must come from the final plan, never the discarded card-bearing input.
        assert len(patches) == 2
        assert writes and all(value["_test_accepted_patch"] == patches[-1]["_test_accepted_patch"] for value in writes)
        assert len(env.session.runtime_config["qwen_tool_turns_inflight"]) == 1
        assert env.session.runtime_config["qwen_tool_turns"] == []
        assert "[Conversation cards:" not in _rendered(_accepted(env)[0])


async def _assert_cache_retry(monkeypatch, pipeline, *, shm):
    async with _case(monkeypatch) as env:
        env.input_processor.renderer.mm_processor_cache = pipeline.sender
        count = env.input_processor.process_inputs
        cache_process = pipeline.orch._stage_input_processors[1].process_inputs
        received = []

        def process(**kwargs):
            request = count(**kwargs)
            cache_request = cache_process(**kwargs)
            request.mm_features = cache_request.mm_features
            return request

        env.input_processor.process_inputs = process
        client = env.clients[0]
        original_send = client.add_request_async

        async def send(request, *args, **kwargs):
            feature = request.mm_features[0]
            if shm:
                value = SimpleNamespace(mm_hash=feature.mm_hash, identifier=feature.identifier, data=feature.data)
                pipeline.receiver.get_and_update_features([value])
                assert value.data == {"pixel_values": bytes(400 * 1024)}
            else:
                value = pipeline.receiver.get_and_update_item(feature.data, feature.mm_hash)
                assert value == pipeline.media[feature.mm_hash]
            received.append(request.request_id)
            return await original_send(request, *args, **kwargs)

        monkeypatch.setattr(client, "add_request_async", send)
        pool = env.orch.stage_pools[0]
        original_select = pool._pick_or_select
        calls = 0

        async def select(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                _replace_card(env)
            return await original_select(*args, **kwargs)

        monkeypatch.setattr(pool, "_pick_or_select", select)
        await _start(env)
        await _settle(env.orch)
        assert len(env.preparations) == 2
        assert len(received) == len(_accepted(env)) == 1
        assert "[Conversation cards:" not in _rendered(_accepted(env)[0])
        assert not env.orch._stage_input_submission_locks[0].locked()
        assert env.session.input_seq == 1 and env.counter.value == 1


async def test_lru_sender_and_receiver_stay_aligned_after_automatic_retry(monkeypatch, cache_pipeline):
    await _assert_cache_retry(monkeypatch, cache_pipeline, shm=False)


@pytest.mark.parametrize("shm_pipeline", [1], indirect=True)
async def test_shm_sender_and_receiver_stay_aligned_after_automatic_retry(monkeypatch, shm_pipeline):
    await _assert_cache_retry(monkeypatch, shm_pipeline, shm=True)


async def test_no_card_rebuild_cannot_drop_the_admission_metadata(monkeypatch):
    async with _case(monkeypatch) as env:
        original = env.plugin.prepare_append_plan
        calls = 0

        async def prepare(**kwargs):
            nonlocal calls
            calls += 1
            plan = await original(**kwargs)
            if calls == 1:
                _replace_card(env)
                return plan
            return replace(plan, context_selection=None)

        monkeypatch.setattr(env.plugin, "prepare_append_plan", prepare)
        await _start(env)
        await _settle(env.orch)
        assert not _accepted(env)
        assert len(env.preparations) == 2
        assert env.session.input_seq == 0 and env.counter.value == 0


@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_rejected_base_context_releases_materialized_request_state(monkeypatch, cleanup_fails):
    async with _case(monkeypatch) as env:
        original_process = env.input_processor.process_inputs
        original_select = env.orch.stage_pools[0]._pick_or_select
        cleanup = env.orch.cleanup
        cleanup_calls = []

        async def cleanup_attempt(*args, **kwargs):
            cleanup_calls.append(args)
            if cleanup_fails and len(cleanup_calls) == 1:
                raise RuntimeError("controlled rejected-context cleanup failure")
            return await cleanup(*args, **kwargs)

        monkeypatch.setattr(env.orch, "cleanup", cleanup_attempt)

        def process(**kwargs):
            request = original_process(**kwargs)
            request.mm_features = [SimpleNamespace(mm_hash="rejected", identifier="rejected", data={"pixels": [1, 2]})]
            return request

        async def select(*args, **kwargs):
            env.session.register_history_item("fact", {"role": "user", "content": "Changed source."})
            return await original_select(*args, **kwargs)

        env.input_processor.process_inputs = process
        monkeypatch.setattr(env.orch.stage_pools[0], "_pick_or_select", select)
        await _start(env)
        await _settle(env.orch)
        assert not _accepted(env) and env.counter.value == env.session.input_seq == 0
        assert len(cleanup_calls) == 1 and len(env.preparations) == 1
        if not cleanup_fails:
            for state in env.orch.request_states.values():
                assert state.prompt is None and state.mm_features is None, (
                    "rejected materialization must not survive admission"
                )
        assert "context_changed" in await _error_codes(env)


async def test_old_submission_does_not_clean_a_replacement_engine_request(monkeypatch):
    async with _case(monkeypatch) as env:
        contexts = []
        ensure = env.orch.ensure_request
        entered, release = asyncio.Event(), asyncio.Event()
        pool = env.orch.stage_pools[0]
        original_select = pool._pick_or_select

        def record_context(context):
            contexts.append(context)
            return ensure(context)

        async def select(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original_select(*args, **kwargs)

        monkeypatch.setattr(env.orch, "ensure_request", record_context)
        monkeypatch.setattr(pool, "_pick_or_select", select)
        try:
            await _start(env)
            await asyncio.wait_for(entered.wait(), 5)
            context = contexts[-1]
            old = env.orch.request_states[context.request_id]
            await env.orch.cleanup([context.request_id])
            ensure(context)
            replacement = env.orch.request_states[context.request_id]
            assert replacement is not old
            release.set()
            await _settle(env.orch)
            assert not _accepted(env) and len(env.preparations) == 1
            assert env.orch.request_states[context.request_id] is replacement
            assert env.counter.value == env.session.input_seq == 0
        finally:
            release.set()
