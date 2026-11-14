# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Omitted, unprocessed user evidence must not be replaced by older cards."""

from types import SimpleNamespace

import pytest
from vllm import SamplingParams

from tests.engine.duplex.test_qwen_context_memory_runtime import _submissions
from tests.engine.duplex.test_session_runner import RecordingStagePort, close_harness, open_harness
from tests.model_executor.models.qwen3_omni.test_context_budget_processor import _expanded
from tests.model_executor.models.qwen3_omni.test_context_budget_processor import processor as _processor_fixture
from tests.model_executor.models.qwen3_omni.test_memory_card_prompt import _plugin
from vllm_omni.engine.duplex.commands import CreateItem, CreateResponse
from vllm_omni.engine.duplex.memory_types import (
    DuplexMemoryEvidenceSelection,
    DuplexMemoryProposal,
    DuplexMemoryProposalBatch,
)
from vllm_omni.engine.duplex.session.auxiliary_budget import DuplexAuxiliaryBudgetLimits
from vllm_omni.model_executor.models.qwen3_omni.duplex.context_budget import CONTEXT_OMITTED
from vllm_omni.protocol.realtime.items import normalize_conversation_item

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
processor = _processor_fixture


def _complete_coverage(session):
    for _ in range(16):
        context = session.capture_memory_context()
        if context is None:
            return
        assert (
            session.admit_memory_result(context=context, proposal=DuplexMemoryProposalBatch(())).status == "no_change"
        )
    pytest.fail("bounded fixture did not complete coverage")


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["pending", "covered", "retained", "dirty", "partial"])
async def test_actual_owner_submission_fences_omitted_unprocessed_evidence(processor, monkeypatch, state):
    # Only the model transport is replaced. Owner, ledger, quota, formatter,
    # processor tokenization and both admission checks are the real code.
    monkeypatch.setattr(
        RecordingStagePort, "sampling_defaults", lambda self: tuple(SamplingParams(max_tokens=64) for _ in range(3))
    )
    clock = {"now": 0.0}
    plugin = _plugin(processor, limit=2048)
    h = await open_harness(
        plugin=plugin,
        auto_response=False,
        stage_count=3,
        modalities=("text",),
        runtime_config=plugin._context_config,
        model="Qwen/test",
        clock=lambda: clock["now"],
        model_config=SimpleNamespace(max_model_len=32768, hf_config=SimpleNamespace()),
    )
    original_submit = h.port.submit
    accepted_ids = []

    async def materializing_submit(submission, *, before_accept=None):
        if before_accept is None:
            return await original_submit(submission)

        def validate(prompt):
            ids = _expanded(processor, prompt)["input_ids"][0].tolist()
            before_accept(
                SimpleNamespace(prompt_token_ids=ids, sampling_params=submission.context.stage_sampling_params)
            )
            accepted_ids.append(ids)

        return await original_submit(submission, before_accept=validate)

    monkeypatch.setattr(h.port, "submit", materializing_submit)
    try:
        session = h.session
        session.register_history_item("fact", {"role": "user", "content": "I prefer jasmine tea."})
        session.register_history_item("ack", {"role": "assistant", "content": "Recorded."})
        account = session.context_compute_budget(DuplexAuxiliaryBudgetLimits())
        for now, inputs in ((1.0, 560), (6.0, 564)):
            clock["now"] = now
            context = session.capture_memory_context(max_sources=1)
            proposal = DuplexMemoryProposalBatch(
                (
                    DuplexMemoryProposal(
                        "user_preference_tea",
                        "jasmine tea",
                        (DuplexMemoryEvidenceSelection(0, 0, len(context.evidence[0].text)),),
                    ),
                )
            )
            reservation = account.try_reserve(inputs, 512, now=now)
            assert reservation is not None and account.settle(reservation, submitted=True, now=now)
            assert session.admit_memory_result(context=context, proposal=proposal).status == "applied"
        assert account.usage(now=6) == (1124, 1024)
        assert account.next_ready_at(1, 1, now=6) == 61
        # The old assistant window may already be pending when new input arrives.
        assert session.capture_memory_context().evidence[0].role == "assistant"
        correction = (
            "The notebook is blue."
            if state in {"covered", "dirty"}
            else "Correction: I now prefer mint tea, not jasmine tea."
        )
        session.register_history_item("correction", {"role": "user", "content": correction})
        session.register_history_item("correction-ack", {"role": "assistant", "content": "Recorded."})
        if state != "retained":
            for i in range(6):
                session.register_history_item(
                    f"filler-{i}",
                    {
                        "role": "user",
                        "content": "This is disposable background filler for the current synthetic test. " * 40,
                    },
                )
                session.register_history_item(f"filler-ack-{i}", {"role": "assistant", "content": "Acknowledged."})
        if state in {"covered", "dirty"}:
            _complete_coverage(session)
        if state == "dirty":
            session.register_history_item("correction", {"role": "user", "content": "Actually, I now prefer mint tea."})
        if state == "partial":
            # Advance across the old assistant, then cover only a prefix of
            # the correction. A partial source is not a freshness certificate.
            context = session.capture_memory_context()
            assert (
                session.admit_memory_result(context=context, proposal=DuplexMemoryProposalBatch(())).status
                == "no_change"
            )
            context = session.capture_memory_context(max_chars=3)
            assert context.evidence[0].text == "Cor"
            assert (
                session.admit_memory_result(context=context, proposal=DuplexMemoryProposalBatch(())).status
                == "no_change"
            )
        clock["now"] = 50.0
        await h.run(
            CreateItem(
                item=normalize_conversation_item(
                    {
                        "id": "query",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "The user prefers which tea?"}],
                    }
                )
            )
        )
        events = await h.run(CreateResponse())
        assert not [event for event in events if event.type == "error"]
        submitted = _submissions(h, auxiliary=False)
        assert len(submitted) == len(accepted_ids) == 1
        prompt = submitted[0].prompt
        # The plugin moves internal selection metadata into runtime state
        # before dispatch; it is intentionally absent from the public prompt.
        omitted = {ref.source_id for ref in session.runtime_config[CONTEXT_OMITTED]}
        assert (2 in omitted) == (state != "retained")
        assert len(accepted_ids[0]) + 64 <= 2048
        assert account.usage(now=50) == (1124, 1024)
        assert _submissions(h, auxiliary=True) == []
        if state in {"pending", "dirty", "partial"}:
            assert "[Conversation cards:" not in prompt["prompt"]
            assert "jasmine" not in prompt["prompt"]
        else:
            assert "[Conversation cards:" in prompt["prompt"] and "jasmine tea" in prompt["prompt"]
        if state == "retained":
            assert "mint tea" in prompt["prompt"]
    finally:
        await close_harness(h)
