# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen budget/summary memory through the real session runner mailbox."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import pytest

from tests.engine.duplex.test_session_runner import Harness, close_harness, open_harness, tts_output
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.commands import (
    AckPlayback,
    AppendAudio,
    CancelResponse,
    CloseSession,
    Commit,
    CreateItem,
    CreateResponse,
    DeleteItem,
    TruncateItem,
)
from vllm_omni.engine.duplex.contracts import DuplexStageSubmission, duplex_is_auxiliary_request
from vllm_omni.engine.duplex.events import DuplexEvent
from vllm_omni.engine.duplex.session.context_ledger import DuplexContextLevel
from vllm_omni.model_executor.models.qwen3_omni.duplex.context_budget import CONTEXT_OMITTED
from vllm_omni.model_executor.models.qwen3_omni.duplex.context_memory import ContextSummary
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass(frozen=True)
class _FeatureExtractor:
    hop_length: int = 160


class _Processor:
    """Deterministic prompt expansion; model output still comes from the runner."""

    feature_extractor = _FeatureExtractor()

    def apply_chat_template(self, conversation: list[dict[str, object]], **kwargs: object) -> str:
        del kwargs
        return "\n".join(self._message(message) for message in conversation)

    @staticmethod
    def _message(message: dict[str, object]) -> str:
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                else:
                    parts.append(f"[{part.get('type')}]")
            content = " ".join(parts)
        return f"{message.get('role', '')}:{content}"

    @staticmethod
    def _tokens(text: str, audio: list[np.ndarray] | None = None, images: list[object] | None = None) -> int:
        return len(text) + 64 * len(audio or ()) + 128 * len(images or ())

    def tokenizer(self, *, text: list[str], **kwargs: object) -> dict[str, list[list[int]]]:
        del kwargs
        return {"input_ids": [list(range(self._tokens(value))) for value in text]}

    def __call__(
        self,
        *,
        text: str,
        audio: list[np.ndarray] | None = None,
        images: list[object] | None = None,
        sampling_rate: int,
        return_tensors: str,
    ) -> dict[str, np.ndarray]:
        del sampling_rate, return_tensors
        return {"input_ids": np.zeros((1, self._tokens(text, audio, images)), dtype=np.int64)}


@dataclass(frozen=True)
class _Completion:
    text: str
    token_ids: tuple[int, ...]
    finish_reason: str = "stop"


@dataclass(frozen=True)
class _Output:
    request_id: str
    outputs: tuple[_Completion, ...]
    accepted: bool = True
    finished: bool = True


_FACTS = (
    (
        "MARKER_ALPHA the access code is 4271 and it must remain available for a later request.",
        "MARKER_ALPHA_ACK I will retain the access code for the rest of this conversation.",
    ),
    (
        "MARKER_BRAVO the preferred meeting room is Cedar and the booking starts at nine.",
        "MARKER_BRAVO_ACK The Cedar room at nine is recorded as the preferred booking.",
    ),
    (
        "MARKER_CHARLIE the vegetarian meal must not contain peanuts or shellfish.",
        "MARKER_CHARLIE_ACK The dietary restriction is recorded for future meal suggestions.",
    ),
    (
        "MARKER_DELTA the delivery address is 18 River Street and the gate code is 5510.",
        "MARKER_DELTA_ACK The River Street address and gate code are retained.",
    ),
) + tuple(
    (
        f"CHECKPOINT_{index} the package for office {index} must arrive before noon and use the rear entrance.",
        f"CHECKPOINT_{index}_ACK The delivery deadline and rear entrance for office {index} are recorded.",
    )
    for index in range(8)
)


def _runtime() -> DuplexSessionRuntimeConfig:
    # This processor counts characters as tokens. Allow the full summary task
    # plus an exchange, while the longer seeded history still requires eviction.
    return DuplexSessionRuntimeConfig(
        qwen_context_policy="budget_summary",
        qwen_context_max_tokens=1600,
        qwen_summary_max_tokens=128,
        qwen_summary_input_max_tokens=1472,
        qwen_summary_timeout_s=5.0,
    )


def _plugin(runtime: DuplexSessionRuntimeConfig) -> Qwen3OmniDuplexPlugin:
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin.processor = _Processor()
    plugin.max_model_len = 4096
    plugin.context_view_policy(stage_id=0, runtime_config=runtime)
    return plugin


async def _open() -> Harness:
    runtime = _runtime()
    return await open_harness(
        plugin=_plugin(runtime),
        auto_response=False,
        stage_count=3,
        modalities=("text",),
        runtime_config=runtime,
        model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    )


def _seed_history(h: Harness) -> None:
    for index, (question, answer) in enumerate(_FACTS):
        h.session.register_history_item(f"fact-{index}-user", {"role": "user", "content": question})
        h.session.register_history_item(f"fact-{index}-assistant", {"role": "assistant", "content": answer})


async def _commit_audio(h: Harness, transcript: str) -> None:
    audio = np.linspace(0.05, 0.1, 16000, dtype=np.float32)
    await h.run(AppendAudio(audio=audio.tobytes(), format="pcm_f32le", sample_rate_hz=16000, is_speech=True))
    events = await h.run(Commit(final=True, create_response=False))
    assert not [event for event in events if event.type == "error"]
    item_id = h.session.history_snapshot().item_ids[-1]
    assert isinstance(item_id, str)
    assert h.session.update_input_transcript(item_id, transcript, audio_part=h.session.input_audio_part(item_id))


def _submissions(h: Harness, *, auxiliary: bool) -> list[DuplexStageSubmission]:
    return [
        submission
        for submission in h.port.submissions
        if duplex_is_auxiliary_request(submission.context.request_id) == auxiliary
    ]


async def _wait_auxiliary(h: Harness, *, count: int = 1):
    async def wait() -> None:
        while len(_submissions(h, auxiliary=True)) < count:
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait(), timeout=2)
    return _submissions(h, auxiliary=True)[count - 1]


async def _respond(h: Harness) -> str:
    before = len(_submissions(h, auxiliary=False))
    events = await h.run(CreateResponse())
    assert len(_submissions(h, auxiliary=False)) == before + 1, [event.type for event in events]
    request_id = _submissions(h, auxiliary=False)[-1].context.request_id
    await h.deliver_and_settle(
        tts_output(request_id, samples=0, text="short private answer", finished=True), stage_id=0
    )
    await h.deliver_and_settle(tts_output(request_id, samples=0, finished=True), stage_id=2)
    return request_id


async def _finish_summary(h: Harness, request_id: str, text: str) -> list[DuplexEvent]:
    assert h.deliver(
        _Output(request_id, (_Completion(text, tuple(range(min(len(text), 32)))),)),
        stage_id=0,
        segment_finished=True,
    )
    return await h.settle(idle_s=0.01, timeout_s=0.2)


@pytest.mark.asyncio
async def test_budget_summary_is_private_and_reused_by_the_next_real_prompt() -> None:
    h = await _open()
    try:
        _seed_history(h)
        before = h.session.history_snapshot()
        await _commit_audio(h, "What should I remember from the earlier conversation?")
        await _respond(h)

        omitted = h.session.runtime_config.get(CONTEXT_OMITTED)
        assert isinstance(omitted, tuple) and omitted
        assert "MARKER_ALPHA" not in _submissions(h, auxiliary=False)[-1].prompt["prompt"]
        summary_request = await _wait_auxiliary(h)
        assert summary_request.context.stage_id == 0
        assert "MARKER_ALPHA the access code" in summary_request.prompt["prompt"]

        events = await _finish_summary(h, summary_request.context.request_id, "Earlier facts are retained.")
        assert not any(event.to_realtime()["type"].startswith("response.") for event in events)
        memory = h.runner.model_state.context_memory
        assert isinstance(memory.summary, ContextSummary)
        assert memory.summary.text == "Earlier facts are retained."
        assert memory.ledger is not None and memory.ledger.valid(memory.summary.node_id)
        summaries = [node for node in memory.ledger.nodes() if node.level == DuplexContextLevel.SUMMARY]
        assert [node.node_id for node in summaries] == [memory.summary.node_id]
        assert h.session.history_snapshot() == before or len(h.session.history_snapshot().messages) > len(
            before.messages
        )

        await _commit_audio(h, "Please continue using the earlier facts.")
        before_next = len(_submissions(h, auxiliary=False))
        await h.run(CreateResponse())
        assert len(_submissions(h, auxiliary=False)) == before_next + 1
        next_prompt = _submissions(h, auxiliary=False)[-1].prompt["prompt"]
        assert "Earlier facts are retained." in next_prompt
        assert "MARKER_ALPHA the access code" not in next_prompt
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_source_deletion_rejects_an_inflight_summary_result() -> None:
    h = await _open()
    try:
        _seed_history(h)
        await _commit_audio(h, "Use the earlier facts for this answer.")
        await _respond(h)
        summary_request = await _wait_auxiliary(h)
        assert h.session.delete_history_item("fact-0-user")

        await _finish_summary(h, summary_request.context.request_id, "Stale summary must not publish.")
        memory = h.runner.model_state.context_memory
        assert memory.summary is None
        assert memory.ledger is not None
        assert not [node for node in memory.ledger.nodes() if node.level == DuplexContextLevel.SUMMARY]
        assert "fact-0-user" not in h.session.history_snapshot().item_ids
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_new_audio_preempts_summary_and_rejects_its_late_output() -> None:
    h = await _open()
    try:
        _seed_history(h)
        await _commit_audio(h, "First question that leaves older turns behind.")
        await _respond(h)
        summary_request = await _wait_auxiliary(h)
        summary_id = summary_request.context.request_id

        await _commit_audio(h, "A new live question must take priority.")
        await _respond(h)
        assert ([summary_id], True) in h.port.cleanups
        await _finish_summary(h, summary_id, "Late summary must be ignored.")
        memory = h.runner.model_state.context_memory
        assert memory.summary is None
        assert memory.ledger is not None
        assert not [node for node in memory.ledger.nodes() if node.level == DuplexContextLevel.SUMMARY]
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_failed_source_does_not_loop_but_new_omitted_evidence_retries() -> None:
    h = await _open()
    try:
        _seed_history(h)
        await _commit_audio(h, "First question that leaves older turns behind.")
        await _respond(h)
        first = await _wait_auxiliary(h)

        await _finish_summary(h, first.context.request_id, "")
        await asyncio.sleep(0.05)
        assert len(_submissions(h, auxiliary=True)) == 1
        memory = h.runner.model_state.context_memory
        assert memory.summary is None and memory.attempted is not None
        first_omitted = set(memory.omitted)

        for index, (question, answer) in enumerate(_FACTS, start=len(_FACTS)):
            h.session.register_history_item(f"fact-{index}-user", {"role": "user", "content": question})
            h.session.register_history_item(f"fact-{index}-assistant", {"role": "assistant", "content": answer})

        await _commit_audio(h, "A later question that omits additional history.")
        await _respond(h)
        second = await _wait_auxiliary(h, count=2)
        later_omitted = h.session.runtime_config.get(CONTEXT_OMITTED)
        assert isinstance(later_omitted, tuple)
        assert first_omitted < set(later_omitted)
        assert second.context.request_id != first.context.request_id
        assert "MARKER_ALPHA the access code" in second.prompt["prompt"]
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_live_lease_resume_preserves_summary_and_source_revisions() -> None:
    h = await _open()
    try:
        _seed_history(h)
        await _commit_audio(h, "Keep the earlier facts available.")
        await _respond(h)
        request = await _wait_auxiliary(h)
        await _finish_summary(h, request.context.request_id, "Access code 4271.")
        state = h.session.model_state
        summary = h.runner.model_state.context_memory.summary
        assert summary is not None
        snapshot = h.session.history_snapshot()
        generation = h.session.lease.generation
        h.session.detach_lease()
        assert h.session.resume_lease(expected_lease_generation=generation) == generation + 1
        assert h.session.model_state is state
        assert h.session.history_snapshot() == snapshot
        h.runner.plugin.observe_history_context(
            snapshot=h.session.history_snapshot(),
            ledger=h.session.context_ledger,
            state=h.runner.model_state,
            runtime_config=h.session.runtime_config,
        )
        assert h.runner.model_state.context_memory.summary == summary
        assert h.session.context_ledger.valid(summary.node_id)
    finally:
        await close_harness(h)


async def _wait_until(predicate) -> None:
    async def wait() -> None:
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), timeout=2)


async def _seed_public_history(h: Harness) -> None:
    for index, (question, answer) in enumerate(_FACTS):
        for role, content in (("user", question), ("assistant", answer)):
            h.submit(
                CreateItem(
                    item={
                        "id": f"fact-{index}-{role}",
                        "type": "message",
                        "role": role,
                        "content": [{"type": "input_text" if role == "user" else "output_text", "text": content}],
                    }
                )
            )
    events = await h.settle()
    assert not [event for event in events if event.type == "error"]
    h.session.reset_unanswered_user_items()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["unrelated_append", "selected_delete", "selected_correction"])
async def test_foreground_history_fence_rejects_stale_plan_and_retry_recovers(monkeypatch, mutation) -> None:
    h = await _open()
    entered, release = asyncio.Event(), asyncio.Event()
    original = h.runner.plugin.prepare_append_plan
    prepared = []

    async def blocked_prepare(**kwargs):
        plan = await original(**kwargs)
        prepared.append(plan)
        entered.set()
        await release.wait()
        return plan

    try:
        await _seed_public_history(h)
        await _commit_audio(h, "MANDATORY_CURRENT_QUESTION: retain this input across retries.")
        current_item = h.session.history_snapshot().item_ids[-1]
        assert isinstance(current_item, str)
        monkeypatch.setattr(h.runner.plugin, "prepare_append_plan", blocked_prepare)
        h.submit(CreateResponse())
        await asyncio.wait_for(entered.wait(), timeout=2)
        stale_prompt = prepared[0].prompt["prompt"]
        assert "user:[audio]" in stale_prompt
        assert "CHECKPOINT_7 the package" in stale_prompt
        revision = h.session.history_revision
        if mutation == "unrelated_append":
            h.submit(
                CreateItem(
                    item={
                        "id": "unrelated-observation",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Unrelated new observation."}],
                    }
                )
            )
        elif mutation == "selected_delete":
            h.submit(DeleteItem(item_id="fact-11-user"))
        else:
            h.submit(DeleteItem(item_id="fact-11-user"))
            h.submit(
                CreateItem(
                    item={
                        "id": "fact-11-user",
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "CORRECTED_SOURCE: office 7 uses the front entrance."}
                        ],
                    }
                )
            )
            await _wait_until(lambda: h.session.history_revision >= revision + 2)
        await _wait_until(lambda: h.session.history_revision > revision)
        release.set()
        events = await h.settle()
        assert [event.code for event in events if event.type == "error"] == ["context_changed"]
        assert not _submissions(h, auxiliary=False)
        assert h.session.active_response_id is None
        assert not h.session.lease.active_operations
        assert not h.session.request_resources
        assert not h.manager._request_index
        assert len(prepared) == 1  # No hidden unbounded preparation loop.
        assert set(h.session.user_source_ids_for_items((current_item,))) <= set(h.session.pending_user_source_ids())

        monkeypatch.setattr(h.runner.plugin, "prepare_append_plan", original)
        await _respond(h)
        assert len(_submissions(h, auxiliary=False)) == 1
        prompt = _submissions(h, auxiliary=False)[0].prompt["prompt"]
        assert "user:[audio]" in prompt
        assert current_item in h.session.history_snapshot().item_ids
        audio = _submissions(h, auxiliary=False)[0].prompt["multi_modal_data"]["audio"]
        assert any(np.array_equal(value[0], np.linspace(0.05, 0.1, 16000, dtype=np.float32)) for value in audio)
        if mutation == "selected_correction":
            assert "CORRECTED_SOURCE" in prompt
            assert "CHECKPOINT_7 the package" not in prompt
        elif mutation == "selected_delete":
            assert "CHECKPOINT_7 the package" not in prompt
            assert "fact-11-user" not in h.session.history_snapshot().item_ids
        assert not h.session.lease.active_operations
    finally:
        release.set()
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["cancel", "close"])
async def test_foreground_prepare_cancel_or_close_releases_work(monkeypatch, action) -> None:
    h = await _open()
    entered, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = h.runner.plugin.prepare_append_plan

    async def blocked_prepare(**kwargs):
        plan = await original(**kwargs)
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return plan

    try:
        _seed_history(h)
        await _commit_audio(h, "Pending input before cancellation.")
        monkeypatch.setattr(h.runner.plugin, "prepare_append_plan", blocked_prepare)
        old_epoch = h.session.epoch
        h.submit(CreateResponse())
        await asyncio.wait_for(entered.wait(), timeout=2)
        h.submit(CancelResponse() if action == "cancel" else CloseSession(reason="client_close"))
        await asyncio.wait_for(cancelled.wait(), timeout=2)
        await h.settle()
        assert not _submissions(h, auxiliary=False)
        assert h.session.active_response_id is None
        assert not h.session.lease.active_operations
        assert not h.session.request_resources
        assert not [task for task in h.runner.tasks.append_tasks if not task.done()]
        assert h.runner.memory._request_id is None
        assert h.runner.memory._output is None
        if action == "cancel":
            assert h.session.epoch > old_epoch
            monkeypatch.setattr(h.runner.plugin, "prepare_append_plan", original)
            events = await h.run(CreateResponse())
            assert [event.code for event in events if event.type == "error"] == ["response_create_without_input"]
            await _commit_audio(h, "NEW_EPOCH_INPUT must be answered after cancellation.")
            await _respond(h)
            assert "user:[audio]" in _submissions(h, auxiliary=False)[0].prompt["prompt"]
            assert _submissions(h, auxiliary=False)[0].context.fence.epoch == h.session.epoch
    finally:
        release.set()
        await close_harness(h)


@pytest.mark.asyncio
async def test_context_changed_does_not_rearm_deleted_current_input(monkeypatch) -> None:
    h = await _open()
    entered, release = asyncio.Event(), asyncio.Event()
    original = h.runner.plugin.prepare_append_plan

    async def blocked_prepare(**kwargs):
        plan = await original(**kwargs)
        entered.set()
        await release.wait()
        return plan

    try:
        _seed_history(h)
        await _commit_audio(h, "Delete this pending question before it is submitted.")
        current_item = h.session.history_snapshot().item_ids[-1]
        assert isinstance(current_item, str)
        monkeypatch.setattr(h.runner.plugin, "prepare_append_plan", blocked_prepare)
        h.submit(CreateResponse())
        await asyncio.wait_for(entered.wait(), timeout=2)
        h.submit(DeleteItem(item_id=current_item))
        await _wait_until(lambda: current_item not in h.session.history_snapshot().item_ids)
        release.set()
        events = await h.settle()
        assert [event.code for event in events if event.type == "error"] == ["context_changed"]
        assert not h.session.pending_user_source_ids()
        assert not h.session.request_resources
        assert not h.session.lease.active_operations
        assert h.runner.model_state.committed_audio_payload is None
        events = await h.run(CreateResponse())
        assert [event.code for event in events if event.type == "error"] == ["response_create_without_input"]
        assert not _submissions(h, auxiliary=False)
    finally:
        release.set()
        await close_harness(h)


@pytest.mark.asyncio
async def test_repeated_history_changes_make_bounded_attempts_and_eventually_progress(monkeypatch) -> None:
    h = await _open()
    entered, release = asyncio.Event(), asyncio.Event()
    original = h.runner.plugin.prepare_append_plan
    attempts = 0

    async def blocked_prepare(**kwargs):
        nonlocal attempts
        attempts += 1
        plan = await original(**kwargs)
        entered.set()
        await release.wait()
        return plan

    try:
        await _seed_public_history(h)
        await _commit_audio(h, "The same committed input must survive repeated preparation races.")
        current_item = h.session.history_snapshot().item_ids[-1]
        monkeypatch.setattr(h.runner.plugin, "prepare_append_plan", blocked_prepare)
        for index in range(5):
            entered.clear()
            release.clear()
            h.submit(CreateResponse())
            await asyncio.wait_for(entered.wait(), timeout=2)
            revision = h.session.history_revision
            h.submit(
                CreateItem(
                    item={
                        "id": f"observation-{index}",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": f"Unrelated observation {index}."}],
                    }
                )
            )
            await _wait_until(lambda: h.session.history_revision > revision)
            release.set()
            events = await h.settle()
            assert [event.code for event in events if event.type == "error"] == ["context_changed"]
            assert attempts == index + 1
            assert not _submissions(h, auxiliary=False)
            assert not h.session.request_resources
            assert not h.session.lease.active_operations
            assert current_item in h.session.history_snapshot().item_ids
        await _respond(h)
        assert attempts == 6
        assert len(_submissions(h, auxiliary=False)) == 1
        assert "user:[audio]" in _submissions(h, auxiliary=False)[0].prompt["prompt"]
    finally:
        release.set()
        await close_harness(h)


@pytest.mark.asyncio
async def test_source_deletion_during_auxiliary_preparation_prevents_submit_and_recovers(monkeypatch) -> None:
    h = await _open()
    entered, release = asyncio.Event(), asyncio.Event()
    original = h.runner.plugin.prepare_auxiliary_request
    prepared = []

    async def blocked_prepare(**kwargs):
        request = await original(**kwargs)
        prepared.append(request)
        entered.set()
        await release.wait()
        return request

    try:
        await _seed_public_history(h)
        monkeypatch.setattr(h.runner.plugin, "prepare_auxiliary_request", blocked_prepare)
        await _commit_audio(h, "Summarize the earlier facts after answering.")
        await _respond(h)
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert prepared[0] is not None
        assert "MARKER_ALPHA the access code" in prepared[0].prompt["prompt"]
        h.submit(DeleteItem(item_id="fact-0-user"))
        await _wait_until(lambda: "fact-0-user" not in h.session.history_snapshot().item_ids)
        release.set()
        await h.settle(timeout_s=0.2)
        assert all(
            "MARKER_ALPHA the access code" not in submission.prompt["prompt"]
            for submission in _submissions(h, auxiliary=True)
        )
        assert h.runner.model_state.context_memory.summary is None

        monkeypatch.setattr(h.runner.plugin, "prepare_auxiliary_request", original)
        request = await _wait_auxiliary(h)
        assert "MARKER_ALPHA the access code" not in request.prompt["prompt"]
        await _finish_summary(h, request.context.request_id, "Meeting room Cedar remains available.")
        summary = h.runner.model_state.context_memory.summary
        assert summary is not None and "Cedar" in summary.text
    finally:
        release.set()
        await close_harness(h)


@pytest.mark.asyncio
async def test_source_deletion_after_summary_output_before_publication_rejects_result(monkeypatch) -> None:
    h = await _open()
    entered, release = asyncio.Event(), asyncio.Event()
    original = h.port.cleanup

    async def blocked_cleanup(request_ids, *, abort):
        if any(duplex_is_auxiliary_request(request_id) for request_id in request_ids):
            entered.set()
            await release.wait()
        return await original(request_ids, abort=abort)

    try:
        await _seed_public_history(h)
        await _commit_audio(h, "Use the earlier facts.")
        await _respond(h)
        request = await _wait_auxiliary(h)
        monkeypatch.setattr(h.port, "cleanup", blocked_cleanup)
        assert h.deliver(
            _Output(request.context.request_id, (_Completion("Access code 4271.", (1, 2)),)),
            stage_id=0,
            segment_finished=True,
        )
        await asyncio.wait_for(entered.wait(), timeout=2)
        h.submit(DeleteItem(item_id="fact-0-user"))
        await _wait_until(lambda: "fact-0-user" not in h.session.history_snapshot().item_ids)
        release.set()
        await h.settle(timeout_s=0.2)
        memory = h.runner.model_state.context_memory
        assert memory.summary is None
        assert not [node for node in memory.ledger.nodes() if node.level == DuplexContextLevel.SUMMARY]
        assert (0, request.context.request_id) not in h.session.request_resources
        replacement = await _wait_auxiliary(h, count=2)
        assert "MARKER_ALPHA the access code" not in replacement.prompt["prompt"]
        await _finish_summary(h, replacement.context.request_id, "Meeting room Cedar remains available.")
        assert memory.summary is not None and "Cedar" in memory.summary.text
    finally:
        release.set()
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("played_ms", [0, 1000, 2000], ids=["zero", "partial", "full"])
@pytest.mark.parametrize("ack_before_cancel", [True, False], ids=["ack-before-cancel", "ack-after-cancel"])
async def test_calibrated_playback_cutoff_controls_summary_source(played_ms, ack_before_cancel) -> None:
    """Exact synthetic alignment marks test memory admission, not acoustic alignment."""
    h = await _open()
    first, second = "HEARD_FACT: the badge is violet.", "UNHEARD_FACT: the locker is 9127."
    try:
        h.session.config.playback_commit_policy = "ack_only"
        await h.run(
            CreateItem(
                item={
                    "id": "playback-question",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Tell me the badge color and locker number."}],
                }
            )
        )
        response_id = h.session.begin_response()
        h.runner.emit(h.runner.model.response_created_payload(response_id, epoch=h.session.epoch))
        h.session.append_assistant_text(first + " " + second)
        h.session.mark_audio_sent(1000, text_chars=len(first))
        h.session.mark_audio_sent(2000, text_chars=len(first + " " + second), audio_complete=True)
        h.runner.emit(
            {
                "type": "response.output_audio.delta",
                "response_id": response_id,
                "audio": "AAA=",
                "text": first + " " + second,
                "audio_duration_ms": 2000,
                "audio_text_marks": [
                    {"text_chars": len(first), "audio_end_ms": 1000},
                    {"text_chars": len(first + " " + second), "audio_end_ms": 2000},
                ],
            }
        )
        await h.settle()
        if ack_before_cancel:
            await h.run(AckPlayback(response_id=response_id, played_ms=played_ms))
        await h.run(CancelResponse())
        if not ack_before_cancel:
            await h.run(AckPlayback(response_id=response_id, played_ms=played_ms))
        events = await h.run(TruncateItem(item_id=f"item_{response_id}", audio_end_ms=played_ms))
        assert not [event for event in events if event.type == "error"]
        # A late full ACK cannot undo the authoritative hard cutoff.
        await h.run(AckPlayback(response_id=response_id, played_ms=2000))
        history_text = str(h.session.history_snapshot().messages)
        assert (first in history_text) is (played_ms >= 1000)
        assert (second in history_text) is (played_ms == 2000)

        await _seed_public_history(h)
        await _commit_audio(h, "Use the earlier conversation.")
        await _respond(h)
        request = await _wait_auxiliary(h)
        prompt = request.prompt["prompt"]
        assert "MARKER_ALPHA the access code" in prompt  # Independent positive control.
        assert (first in prompt) is (played_ms >= 1000)
        assert (second in prompt) is (played_ms == 2000)
    finally:
        await close_harness(h)
