# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A playback ACK must not revoke unrelated camera-source memory."""

import pytest

from tests.model_executor.models.qwen3_omni.test_history_multimodal import image_message
from tests.model_executor.models.qwen3_omni.test_history_policy import completion, runtime, sync_history
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("merged", [False, True])
async def test_materializing_assistant_placeholder_keeps_valid_visual_memory(merged):
    owner = runtime(recent_messages=2, visual_chunk_images=1, visual_mid_group=2)
    state = owner.create_state()
    session = DuplexEngineSession(session_id="camera-playback", config=DuplexSessionConfig(model="test"))
    session.register_history_item("early-camera", image_message())
    session.reserve_history_item("pending-reply")
    session.register_history_item("later-camera", image_message("blue"))
    for i in range(4):
        role = "user" if i % 2 == 0 else "assistant"
        session.register_history_item(f"tail-{i}", {"role": role, "content": f"later {i}"})
    sync_history(state, session.history, session=session)
    for text in ("first a red object", "later a blue object"):
        request = await owner.prepare(state)
        assert owner.apply(state, request, completion(text))
    if merged:
        request = await owner.prepare(state)
        assert request.model_source.source.kind == "long"
        assert owner.apply(state, request, completion("first red, then blue"))
    nodes = dict(state.visual.nodes)
    source_ids = {item.source for item in state.visual.inputs}
    covered = state.visual.covered
    session.register_history_item("pending-reply", {"role": "assistant", "content": "played reply"})
    snapshot = session.history_snapshot()
    session.context_ledger.refresh(snapshot)
    assert all(session.context_ledger.valid(node) for node in nodes.values())
    sync_history(state, session.history, session=session)
    assert {item.source for item in state.visual.inputs} == source_ids
    assert state.visual.nodes == nodes
    assert state.visual.covered == covered
    assert "red" in " ".join(item.text for item in state.visual.summaries())


@pytest.mark.asyncio
async def test_placeholder_shift_rejects_inflight_labels_without_losing_published_memory():
    owner = runtime(recent_messages=2, visual_chunk_images=1)
    state = owner.create_state()
    session = DuplexEngineSession(session_id="inflight-camera", config=DuplexSessionConfig(model="test"))
    session.register_history_item("early", image_message())
    session.reserve_history_item("reply")
    session.register_history_item("later", image_message("blue"))
    for i in range(4):
        role = "user" if i % 2 == 0 else "assistant"
        session.register_history_item(f"tail-{i}", {"role": role, "content": f"later {i}"})
    sync_history(state, session.history, session=session)
    first = await owner.prepare(state)
    assert owner.apply(state, first, completion("first red"))
    inflight = await owner.prepare(state)
    nodes = dict(state.visual.nodes)
    session.register_history_item("reply", {"role": "assistant", "content": "played reply"})
    sync_history(state, session.history, session=session)
    assert state.visual.nodes == nodes
    assert not owner.source_matches(state, inflight.model_source)
    retry = await owner.prepare(state)
    assert retry.prompt["multi_modal_data"]["image"][0].getpixel((0, 0)) == (0, 0, 255)
    assert owner.apply(state, retry, completion("later blue"))
