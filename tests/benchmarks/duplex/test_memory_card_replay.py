# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fixture replay drives real owner state, independently of model answers."""

import importlib
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from tests.engine.duplex.test_memory_card_ledger import _admit, _operation
from tests.engine.duplex.test_session_manager import FakeClock
from tests.engine.duplex.test_session_runner import close_harness, open_harness
from vllm_omni.engine.duplex.messages import CloseDuplexSessionMessage, ResumeDuplexSessionMessage

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
SEEDS = Path(__file__).parent / "fixtures/memory_card_seeds.jsonl"


def api():
    return importlib.import_module("vllm_omni.benchmarks.duplex.memory_card_replay")


def message(item="fact", text="Old value", label="old"):
    return {"type": "message", "item_id": item, "role": "user", "text": text, "label": label}


def case(events, *, timings=("fixed_idle",), queries=1):
    gold = {
        "fact_key": "user.fact",
        "expected_status": "known",
        "required_facts": [{"key": "user.fact", "value": "gold-only-canary"}],
        "must_not_assert_as_current": [],
        "allow_explicit_historical_mentions": True,
        "evidence_expectation": "current_valid_source",
    }
    return {
        "case_id": "fixture",
        "synthetic": True,
        "events": events,
        "queries": [{"text": f"Query {i}", "gold": gold} for i in range(queries)],
        "timing_variants": list(timings),
    }


def load(tmp_path, data):
    path = tmp_path / "fixture.jsonl"
    path.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
    return api().load_memory_replay_cases(path)


class Driver:
    def __init__(self, inspect=None):
        self.harnesses = []
        self.idles = []
        self.resumes = []
        self.queries = []
        self.closed = 0
        self.inspect = inspect
        self.on_idle = None

    @asynccontextmanager
    async def factory(self, variant):
        # The factory/query interface must not supply evaluation answers.
        assert not hasattr(variant, "gold") and not hasattr(variant, "queries")
        h = await open_harness(auto_response=False, clock=FakeClock())
        h.session.replace_config(replace(h.session.config, playback_commit_policy="ack_only"))
        self.harnesses.append(h)
        try:
            yield h.session
        finally:
            await close_harness(h)
            self.closed += 1

    def harness(self, session):
        return next(h for h in self.harnesses if h.session is session)

    async def idle(self, session, duration_ms):
        self.idles.append((session, duration_ms))
        self.harness(session).manager._clock.advance(duration_ms / 1000)
        if self.on_idle is not None:
            self.on_idle(session, duration_ms)

    async def resume(self, session):
        h = self.harness(session)
        session.detach_lease()
        await h.manager.handle(
            ResumeDuplexSessionMessage(
                control_id="replay-resume",
                session_id=session.session_id,
                expected_lease_generation=session.lease_generation,
            )
        )
        assert (await h.results.get()).ok
        self.resumes.append(session)
        return h.manager.runners[session.session_id].session

    async def query(self, session, text):
        self.queries.append((session, text))
        if self.inspect is not None:
            self.inspect(session, text)
        return {"fixture_query_completed": True}

    async def run(self, cases, **kwargs):
        return await api().replay_memory_cases(
            cases,
            session_factory=self.factory,
            idle=self.idle,
            resume=self.resume,
            query=self.query,
            **kwargs,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["edit", "delete", "recreate"])
async def test_mutation_updates_authority_and_invalidates_old_card(tmp_path, mutation):
    events = [message(), {"type": "idle", "duration_ms": 10}]
    if mutation == "edit":
        events.append({"type": "edit", "item_id": "fact", "replacement_text": "New value", "label": "new"})
    else:
        events.append({"type": "delete", "item_id": "fact"})
        if mutation == "recreate":
            events.append(message(text="New value", label="new"))
    cases = load(tmp_path, case(events))

    def inspect(session, _):
        assert [m["content"] for m in session.history_snapshot().messages] == (
            [] if mutation == "delete" else ["New value"]
        )
        assert session.capture_memory_prompt(query="Old value").cards == ()

    driver = Driver(inspect)
    driver.on_idle = lambda session, _: _admit(session, _operation(session, "fact", "Old value", key="user/self/fact"))
    (result,) = await driver.run(cases)
    old = next(row for row in result.source_trace if row.label == "old")
    assert old.source_id is not None and old.revision == 0
    if mutation != "delete":
        new = next(row for row in result.source_trace if row.label == "new")
        if mutation == "edit":
            assert new.source_id == old.source_id and new.revision == 1
        else:
            assert new.source_id != old.source_id and new.revision == 0
    assert driver.closed == 1 and driver.harnesses[0].manager.active_count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("heard,played", [("The day is Tuesday.", None), ("", 0), ("", 1000)])
async def test_calibrated_fixture_never_admits_unheard_suffix(tmp_path, heard, played):
    full = "The day is Tuesday. The secret is Blue Lantern."
    event = {
        "type": "assistant_playback",
        "item_id": "answer",
        "label": "spoken",
        "generated_text": full,
        "heard_text": heard,
        "heard_char_end": len(heard),
        "calibration": "accepted_prefix_fixture_only" if heard else "accepted_empty_prefix_fixture_only",
    }
    if played is not None:
        event["played_audio_ms"] = played
    if not heard:
        event["barge_in"] = True

    def inspect(session, _):
        snapshot = session.history_snapshot()
        assert all("Blue Lantern" not in str(m.get("content", "")) for m in snapshot.messages)
        context = session.capture_memory_context()
        texts = [] if context is None else [e.text for e in context.evidence]
        assert texts == ([heard] if heard else [])

    driver = Driver(inspect)
    (result,) = await driver.run(load(tmp_path, case([event])), allow_calibrated_text_fixtures=True)
    assert result.calibrated_text_fixture_only
    if heard:
        assert result.source_trace[0].source_id is not None
    assert driver.closed == 1


@pytest.mark.asyncio
async def test_immediate_variant_omits_only_last_idle_before_live_resume(tmp_path):
    events = [
        message(),
        {"type": "idle", "duration_ms": 10},
        {"type": "edit", "item_id": "fact", "replacement_text": "New value", "label": "new"},
        {"type": "idle", "duration_ms": 20},
        {"type": "resume_live_session", "same_session_object": True},
    ]
    driver = Driver()
    results = await driver.run(load(tmp_path, case(events, timings=("fixed_idle", "immediate_after_mutation"))))
    assert [duration for _, duration in driver.idles] == [10, 20, 10]
    assert len(driver.resumes) == 2 and driver.resumes[0] is not driver.resumes[1]
    assert [r.timing for r in results] == ["fixed_idle", "immediate_after_mutation"]


@pytest.mark.asyncio
async def test_each_query_has_fresh_owner_and_no_previous_answer(tmp_path):
    def inspect(session, _):
        assert "answer" not in session.history_snapshot().item_ids
        session.register_history_item("answer", {"role": "assistant", "content": "previous answer"})

    driver = Driver(inspect)
    results = await driver.run(load(tmp_path, case([message()], queries=2)))
    assert len(results) == 2 and driver.queries[0][0] is not driver.queries[1][0]
    assert driver.closed == 2


@pytest.mark.asyncio
async def test_query_error_still_closes_real_manager(tmp_path):
    def inspect(session, text):
        raise RuntimeError("query_failed")

    driver = Driver(inspect)
    with pytest.raises(RuntimeError, match="query_failed"):
        await driver.run(load(tmp_path, case([message()])))
    assert driver.closed == 1 and driver.harnesses[0].manager.active_count() == 0


@pytest.mark.asyncio
async def test_seed_suite_replays_thirteen_variants_and_records_native_sources():
    cases = api().load_memory_replay_cases(SEEDS)
    assert len(cases) == 8
    variants = api().memory_replay_variants(cases)
    assert len(variants) == 13
    assert len({(v.case_id, v.query_index) for v in variants}) == 9
    assert {v.source_sha256 for v in variants} == {"c0c6241a616e86700be395b8eb10a5b314decfd3d05226e4500e36d71d86b7b9"}
    driver = Driver()
    results = await driver.run(cases, allow_calibrated_text_fixtures=True)
    assert len(results) == driver.closed == 13
    assert all(result.source_trace for result in results)
    assert all(row.source_id is None or type(row.source_id) is int for result in results for row in result.source_trace)
    assert len(driver.resumes) == 2


@pytest.mark.parametrize(
    "bad_event",
    [
        {"type": "idle", "duration_ms": True},
        {"type": "idle", "duration_ms": -1},
        {"type": "idle", "duration_ms": 600001},
        {"type": "delete", "item_id": "missing"},
        {"type": "edit", "item_id": "missing", "replacement_text": "value", "label": "edit"},
        {"type": "message", "item_id": "fact", "role": "tool", "text": "value", "label": "other"},
        {"type": "idle", "duration_ms": 0, "execute": "forbidden"},
        {"type": "unknown"},
        {"type": "resume_live_session", "same_session_object": False},
        {
            "type": "assistant_playback",
            "item_id": "speech",
            "label": "speech",
            "generated_text": "abc",
            "heard_text": "x",
            "heard_char_end": 1,
            "calibration": "accepted_prefix_fixture_only",
        },
    ],
)
def test_invalid_events_are_rejected_before_replay(tmp_path, bad_event):
    with pytest.raises(ValueError, match="invalid_memory_replay"):
        load(tmp_path, case([message(), bad_event]))


def test_duplicate_json_keys_are_rejected(tmp_path):
    path = tmp_path / "duplicate.jsonl"
    path.write_text('{"case_id":"first","case_id":"second"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_memory_replay"):
        api().load_memory_replay_cases(path)


@pytest.mark.asyncio
async def test_calibrated_text_requires_opt_in_before_opening_any_session():
    driver = Driver()
    with pytest.raises(ValueError, match="calibrated_text_opt_in"):
        await driver.run(api().load_memory_replay_cases(SEEDS))
    assert driver.harnesses == []


@pytest.mark.asyncio
async def test_reusing_owner_for_second_query_is_rejected(tmp_path):
    h = await open_harness(auto_response=False)
    driver = Driver()

    @asynccontextmanager
    async def reuse(variant):
        yield h.session

    try:
        with pytest.raises(ValueError, match="fresh_session"):
            await api().replay_memory_cases(
                load(tmp_path, case([message()], queries=2)),
                session_factory=reuse,
                idle=driver.idle,
                resume=driver.resume,
                query=driver.query,
            )
        assert len(driver.queries) == 1
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_live_resume_cannot_replace_owner(tmp_path):
    driver = Driver()
    other = await open_harness(auto_response=False)

    async def wrong_resume(session):
        return other.session

    try:
        with pytest.raises(ValueError, match="replaced_live_session"):
            await api().replay_memory_cases(
                load(tmp_path, case([message(), {"type": "resume_live_session", "same_session_object": True}])),
                session_factory=driver.factory,
                idle=driver.idle,
                resume=wrong_resume,
                query=driver.query,
            )
        assert driver.queries == [] and driver.closed == 1
    finally:
        await close_harness(other)


def test_valid_fixture_with_excessive_file_size_is_rejected(tmp_path):
    path = tmp_path / "large.jsonl"
    path.write_bytes(json.dumps(case([message()])).encode() + b" " * (8 * 1024 * 1024))
    with pytest.raises(ValueError, match="invalid_memory_replay"):
        api().load_memory_replay_cases(path)


@pytest.mark.parametrize("payload", [b"", b'{"case_id":"\\ud800"}\n', b'{"case_id":NaN}\n'])
def test_invalid_file_encoding_or_json_is_rejected(tmp_path, payload):
    path = tmp_path / "invalid.jsonl"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="invalid_memory_replay"):
        api().load_memory_replay_cases(path)


@pytest.mark.asyncio
async def test_owner_closed_during_idle_never_reaches_query(tmp_path):
    driver = Driver()

    async def close_during_idle(session, duration_ms):
        h = driver.harness(session)
        await h.manager.close(CloseDuplexSessionMessage(control_id="idle-close", session_id=session.session_id))
        assert (await h.results.get()).ok

    with pytest.raises(ValueError, match="closed_session"):
        await api().replay_memory_cases(
            load(tmp_path, case([message(), {"type": "idle", "duration_ms": 10}])),
            session_factory=driver.factory,
            idle=close_during_idle,
            resume=driver.resume,
            query=driver.query,
        )
    assert driver.queries == [] and driver.closed == 1


@pytest.mark.asyncio
async def test_prepopulated_fresh_owner_is_rejected_before_query(tmp_path):
    driver = Driver()

    @asynccontextmanager
    async def polluted(variant):
        async with driver.factory(variant) as session:
            session.register_history_item("previous-answer", {"role": "assistant", "content": "Previous answer"})
            yield session

    with pytest.raises(ValueError, match="fresh_session"):
        await api().replay_memory_cases(
            load(tmp_path, case([message()])),
            session_factory=polluted,
            idle=driver.idle,
            resume=driver.resume,
            query=driver.query,
        )
    assert driver.queries == [] and driver.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("earlier_idle", [True, False])
async def test_already_immediate_query_preserves_all_earlier_idles(tmp_path, earlier_idle):
    events = [message()]
    if earlier_idle:
        events.append({"type": "idle", "duration_ms": 10})
    events.append(message(item="new", label="new", text="New value"))
    driver = Driver()
    results = await driver.run(load(tmp_path, case(events, timings=("fixed_idle", "immediate_after_mutation"))))
    assert len(results) == 2
    assert [duration for _, duration in driver.idles] == ([10, 10] if earlier_idle else [])


@pytest.mark.asyncio
async def test_interrupted_fixture_cannot_be_expanded_by_late_playback_ack(tmp_path):
    def inspect(session, _):
        response_id = session.last_response_id
        session.acknowledge_playback(played_ms=2000, response_id=response_id)
        session.truncate_history_item(f"item_{response_id}", audio_end_ms=2000)
        assert all("Unheard secret" not in str(m.get("content")) for m in session.history_snapshot().messages)

    event = {
        "type": "assistant_playback",
        "item_id": "spoken",
        "label": "spoken",
        "generated_text": "Unheard secret",
        "heard_text": "",
        "heard_char_end": 0,
        "played_audio_ms": 0,
        "barge_in": True,
        "calibration": "accepted_empty_prefix_fixture_only",
    }
    driver = Driver(inspect)
    await driver.run(load(tmp_path, case([event])), allow_calibrated_text_fixtures=True)
