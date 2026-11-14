# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Qwen summary publication must use the shared ledger retirement path."""

import pytest

from tests.engine.duplex.test_memory_card_ledger import _admit, _head, _operation
from tests.engine.duplex.test_qwen_context_memory import Processor, output
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.session.context_ledger import DuplexContextNodeRejectedError, source_refs
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession
from vllm_omni.model_executor.models.qwen3_omni.duplex.context_budget import CONTEXT_OMITTED
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("reject_admission", [False, True])
async def test_summary_replacement_preserves_new_node_and_retires_old_descendants(monkeypatch, reject_admission):
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin.processor = Processor()
    plugin.max_model_len = 16384
    plugin.configure_session_runtime(DuplexSessionRuntimeConfig(qwen_context_policy="budget_summary"))
    state = plugin.create_session_state()
    session = DuplexEngineSession(session_id="qwen-l3-summary", config=DuplexSessionConfig(model="test"))
    ledger = session.context_ledger
    for item_id, role, text in (
        ("fact", "user", "Tea"),
        ("ack", "assistant", "Noted"),
        ("later", "user", "A later fact"),
        ("later-ack", "assistant", "Recorded"),
    ):
        session.register_history_item(item_id, {"role": role, "content": text})
    snapshot = session.history_snapshot()
    refs = source_refs(snapshot, *snapshot.source_ids)

    def observe(omitted):
        plugin.observe_history_context(
            snapshot=snapshot, ledger=ledger, state=state, runtime_config={CONTEXT_OMITTED: omitted}
        )

    observe(refs[:2])
    request = await plugin.prepare_auxiliary_request(history=(), runtime_config={}, state=state)
    assert request is not None
    assert plugin.apply_auxiliary_result(request=request, output=output("The user likes tea."), state=state)
    previous = state.context_memory.summary
    assert previous is not None
    card = _admit(session, _operation(session, "fact", "Tea", parents=(previous.node_id,)))
    assert card.status == "applied"
    unrelated = _admit(session, _operation(session, "later", "A later fact", key="user/self/detail"))
    assert unrelated.status == "applied"
    ledger.card_lexemes(card.node_ids[0])
    observe(refs)
    request = await plugin.prepare_auxiliary_request(history=(), runtime_config={}, state=state)
    assert request is not None
    before_nodes = ledger.nodes()

    if reject_admission:

        def reject(*args, **kwargs):
            raise DuplexContextNodeRejectedError("injected_summary_admission")

        monkeypatch.setattr(ledger, "add_node", reject)
        with pytest.raises(DuplexContextNodeRejectedError, match="injected_summary_admission"):
            plugin.apply_auxiliary_result(request=request, output=output("Updated facts."), state=state)
        assert state.context_memory.summary is previous
        assert ledger.nodes() == before_nodes
        assert _head(session).node_id == card.node_ids[0]
    else:
        assert plugin.apply_auxiliary_result(request=request, output=output("Updated facts."), state=state)
        summary = state.context_memory.summary
        assert summary is not None and summary.node_id != previous.node_id
        assert ledger.valid(summary.node_id)
        assert ledger.nodes()[-1].parents == ()
        assert not ledger.valid(previous.node_id)
        assert not ledger.valid(card.node_ids[0])
        assert ledger.valid(unrelated.node_ids[0])
        assert _head(session).node_id is None
        assert _head(session).generation == 2
        assert ledger._card_lexemes is None
    assert session.history_snapshot() == snapshot
