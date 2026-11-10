# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.session.context_ledger import DuplexContextLedger, DuplexContextLevel, source_refs
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_discard_invalidates_descendants_without_touching_history_or_other_nodes():
    session = DuplexEngineSession(session_id="discard", config=DuplexSessionConfig(model="test"))
    session.register_history_item("one", {"role": "user", "content": "one"})
    session.register_history_item("two", {"role": "user", "content": "two"})
    snapshot = session.history_snapshot()
    ledger = DuplexContextLedger()
    old = ledger.add_node(
        snapshot,
        level=DuplexContextLevel.MEDIA_DROPPED,
        sources=source_refs(snapshot, snapshot.source_ids[0]),
        message=snapshot.messages[0],
    )
    child = ledger.add_node(
        snapshot, level=DuplexContextLevel.SUMMARY, sources=(), parents=(old.node_id,), message={"content": "child"}
    )
    grandchild = ledger.add_node(
        snapshot,
        level=DuplexContextLevel.SUMMARY,
        sources=(),
        parents=(child.node_id,),
        message={"content": "grandchild"},
    )
    independent = ledger.add_node(
        snapshot,
        level=DuplexContextLevel.SUMMARY,
        sources=source_refs(snapshot, *snapshot.source_ids),
        message={"content": "replacement"},
    )
    assert ledger.discard(old.node_id) == (old.node_id, child.node_id, grandchild.node_id)
    assert ledger.nodes() == (independent,)
    assert ledger.l1_node_for(snapshot.source_ids[0]) is None
    assert ledger.discard(old.node_id) == ()
    assert ledger.discard(99999) == ()
    assert session.history_snapshot() == snapshot
    replacement = ledger.add_node(
        snapshot,
        level=DuplexContextLevel.MEDIA_DROPPED,
        sources=source_refs(snapshot, snapshot.source_ids[0]),
        message=snapshot.messages[0],
    )
    assert replacement.node_id > independent.node_id
    assert ledger.l1_node_for(snapshot.source_ids[0]) == replacement
    assert ledger.refresh(snapshot) == ()
