# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Memory state follows the real manager's live-session lifetime."""

import asyncio

import pytest

from tests.engine.duplex.test_context_auxiliary_resources import (
    account,
    active,
    finish,
    install,
    limits,
    second_session,
    start,
    submissions,
)
from tests.engine.duplex.test_memory_card_auxiliary import _proposal
from tests.engine.duplex.test_memory_card_ledger import _admit, _operation
from tests.engine.duplex.test_memory_card_retrieval import CITY, DRINK, _card
from tests.engine.duplex.test_session_auxiliary import wait_until
from tests.engine.duplex.test_session_manager import FakeClock
from tests.engine.duplex.test_session_runner import close_harness, open_harness
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.commands import Heartbeat
from vllm_omni.engine.duplex.config import DuplexSessionState
from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
from vllm_omni.engine.duplex.messages import (
    CloseDuplexSessionMessage,
    OpenDuplexSessionMessage,
    ResumeDuplexSessionMessage,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.asyncio]


async def close_session(h, control_id):
    await h.manager.close(CloseDuplexSessionMessage(control_id=control_id, session_id=h.session.session_id))
    result = await asyncio.wait_for(h.results.get(), 2)
    assert result.ok


async def reopen(h, config):
    settings = h.manager.runtime_config
    buffer = DuplexOutputBuffer(
        max_bytes=settings.max_pending_output_bytes_per_session,
        max_events=settings.max_pending_output_events_per_session,
    )
    await h.manager.handle(
        OpenDuplexSessionMessage(
            control_id="open-replacement",
            session_id=h.session.session_id,
            session_config=config,
            output_buffer=buffer,
        )
    )
    assert (await asyncio.wait_for(h.results.get(), 2)).ok
    h.runner = h.manager.runners[h.session.session_id]
    h.output_buffer = buffer
    return h.session


async def test_live_resume_preserves_card_heads_tombstones_and_spent_budget(mocker):
    # Resetting L3 state on resume would lose facts, revive retired keys or
    # grant another full auxiliary allowance to the same live session.
    clock, policy = FakeClock(), limits()
    h = await open_harness(auto_response=False, clock=clock)
    try:
        work = install(h, mocker, policy)
        city = _card(h.session, "profile", "The user lives in Hangzhou.")
        _card(h.session, "drink", "The user drinks tea.", key=DRINK)
        assert _admit(h.session, _operation(h.session, "drink", None, key=DRINK, generation=1)).status == "applied"
        heads = h.session.context_ledger.card_catalog()
        assert [(head.generation, head.node_id) for head in heads] == [(1, city), (2, None)]
        await start(h)
        budget = account(h, policy)
        assert budget.usage(now=clock()) == (100, 10)
        finish(h)
        await wait_until(lambda: len(work.decodes) == 1)
        await h.settle()

        original = h.session
        original.detach_lease()
        await h.manager.handle(
            ResumeDuplexSessionMessage(
                control_id="resume-memory",
                session_id=original.session_id,
                expected_lease_generation=original.lease_generation,
            )
        )
        assert (await asyncio.wait_for(h.results.get(), 2)).ok
        resumed = h.manager.runners[original.session_id].session
        assert resumed is original
        assert resumed.context_ledger.card_catalog() == heads
        assert account(h, policy) is budget
        assert budget.usage(now=clock()) == (100, 10)
        assert budget.try_reserve(1, 1, now=clock()) is None
        receipt = resumed.capture_memory_prompt(query="Hangzhou")
        assert [card.node_id for card in receipt.cards] == [city]
        assert resumed.memory_prompt_matches(receipt, (city,))
        assert resumed.context_ledger.card_catalog()[1].node_id is None
    finally:
        await close_harness(h)


async def test_reopened_session_rejects_old_receipts_and_auxiliary_results():
    # Reusing a public session/item ID must not reuse the old owner's scope.
    h = await open_harness(auto_response=False, clock=FakeClock())
    try:
        original, config = h.session, h.session.config
        city = _card(original, "profile", "The user lives in Hangzhou.")
        receipt = original.capture_memory_prompt(query="Hangzhou")
        context = original.capture_memory_context()
        assert context is not None
        proposal = _proposal(context)
        await close_session(h, "close-old")
        assert original.state is DuplexSessionState.CLOSED

        replacement = await reopen(h, config)
        assert replacement is not original
        assert replacement.context_ledger.card_catalog() == ()
        assert _card(replacement, "profile", "The user lives in Hangzhou.") == city
        assert replacement.capture_memory_context() is not None
        before = replacement.context_ledger.nodes()
        assert not replacement.memory_prompt_matches(receipt, (city,))
        assert replacement.admit_memory_result(context=context, proposal=proposal).status == "stale"
        assert replacement.context_ledger.nodes() == before
        assert not original.memory_prompt_matches(receipt, (city,))
    finally:
        await close_harness(h)


async def test_same_public_item_ids_cannot_share_or_invalidate_card_state():
    # A key replacement or source edit belongs only to its owning session.
    h = await open_harness(
        auto_response=False, clock=FakeClock(), runtime_config=DuplexSessionRuntimeConfig(max_sessions=2)
    )
    try:
        first = _card(h.session, "profile", "The user lives in Hangzhou.")
        other = await second_session(h)
        second = _card(other.session, "profile", "The user lives in Shanghai.")
        left = h.session.capture_memory_prompt(query="Hangzhou")
        right = other.session.capture_memory_prompt(query="Shanghai")
        assert not other.session.memory_prompt_matches(left, (first,))
        assert not h.session.memory_prompt_matches(right, (second,))
        other.session.delete_history_item("profile")
        assert other.session.capture_memory_prompt(query="Shanghai").cards == ()
        assert h.session.memory_prompt_matches(left, (first,))
        assert [(card.fact_key, card.text) for card in h.session.capture_memory_prompt(query="Hangzhou").cards] == [
            (CITY, "The user lives in Hangzhou.")
        ]
    finally:
        await close_harness(h)


@pytest.mark.parametrize("phase", ["preparing", "running", "waiting_budget"])
async def test_repeated_close_releases_memory_runtime_resources(mocker, phase):
    # Closing/reopening at each auxiliary wait must release the same manager's
    # request/permit capacity and cancel wakeups without reviving old work.
    h = await open_harness(auto_response=False, clock=FakeClock())
    try:
        config = h.session.config
        for cycle in range(3):
            work = install(h, mocker, limits(idle_s=1 if phase == "waiting_budget" else 0))
            if phase == "preparing":
                work.gate = asyncio.Event()
                h.submit(Heartbeat())
                await asyncio.wait_for(work.entered.wait(), 2)
            elif phase == "running":
                before = len(submissions(h))
                h.submit(Heartbeat())
                await wait_until(lambda: len(submissions(h)) == before + 1)
                assert active(h) == 1
            else:
                h.submit(Heartbeat())
                await wait_until(lambda: h.runner.memory._resource_timer is not None)
            old_runner = h.runner
            timer = old_runner.memory._resource_timer
            work.enabled = False
            await close_session(h, f"close-cycle-{cycle}")
            assert h.manager.active_count() == 0
            assert active(h) == 0 and not h.manager._auxiliary_request_permits
            assert old_runner.session.request_resources == {}
            assert old_runner.memory._resource_timer is None
            assert old_runner.memory._retry_timer is None
            assert old_runner.memory._task is None or old_runner.memory._task.done()
            assert timer is None or timer.cancelled()
            if work.gate is not None:
                work.gate.set()
            replacement = await reopen(h, config)
            await h.settle()
            assert replacement.context_ledger.card_catalog() == ()
            assert replacement.capture_memory_prompt(query="Hangzhou").cards == ()
            assert h.manager.active_count() == 1 and active(h) == 0
    finally:
        await close_harness(h)
