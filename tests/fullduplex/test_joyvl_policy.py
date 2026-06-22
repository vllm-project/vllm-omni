# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import pytest

from vllm_omni.experimental.fullduplex.joyvl.bridges.delegation import (
    DelegationResult,
    ImageEditDelegationBridge,
    ImageGenDelegationBridge,
    OpenAIDelegationBridge,
)
from vllm_omni.experimental.fullduplex.joyvl.decision.policy import JoyVLPolicy, sample_frames

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_sample_frames_keeps_recent():
    assert sample_frames(["a", "b"], 4) == ["a", "b"]
    out = sample_frames([str(i) for i in range(10)], 4)
    assert out[-1] == "9" and len(out) == 4


def test_build_messages_stable_head_and_query():
    p = JoyVLPolicy(num_frames=4)
    p.tick(2)
    p.set_query("count the bottles")
    msgs, user = p.build_messages([{"type": "image_url", "image_url": {"url": "x"}}])
    assert msgs[0]["role"] == "system" and "</silence>" in msgs[0]["content"]

    assert any("count the bottles" in part.get("text", "") for part in user["content"])


def test_commit_records_only_spoken():
    p = JoyVLPolicy()
    p.set_query("q")
    assert p.commit("</silence>").action.value == "silence"
    assert p.brain.response_records == []
    assert p.commit("</response> hello").action.value == "response"
    assert p.brain.response_records == [("0.0s", "hello")]


def test_commit_dedup_default_is_exact_match_only():
    p = JoyVLPolicy()  # threshold 1.0 == reference behavior
    assert p.commit("</response> a red ball appears").action.value == "response"
    assert p.commit("</response> a red ball appears").action.value == "silence"  # exact repeat dropped
    assert p.commit("</response> a red ball is appearing").action.value == "response"  # near-dup kept


def test_commit_dedup_threshold_drops_near_duplicates():
    p = JoyVLPolicy(response_dedup_threshold=0.8)
    assert p.commit("</response> a red ball appears").action.value == "response"
    assert p.commit("</response> a red ball appears!").action.value == "silence"  # near-dup dropped
    assert p.commit("</response> the sky is now blue").action.value == "response"  # distinct kept


def test_commit_dedup_rearms_after_silence():
    p = JoyVLPolicy(response_dedup_threshold=0.85)
    assert p.commit("</response> someone raises a hand").action.value == "response"
    # no gap: an immediate repeat is still suppressed
    assert p.commit("</response> someone raises a hand").action.value == "silence"
    # a genuine silent tick (hand lowered) re-arms; the recurrence speaks again
    assert p.commit("</silence>").action.value == "silence"
    assert p.commit("</response> someone raises a hand").action.value == "response"


class _FakeBrain(OpenAIDelegationBridge):
    """Override only the network call so submit/poll plumbing is exercised for real."""

    def __init__(self, answer, gate=None):
        super().__init__("http://unused/v1", "background-model")
        self._answer = answer
        self._gate = gate

    async def _complete(self, messages):
        if self._gate is not None:
            await self._gate.wait()
        return self._answer


@pytest.mark.asyncio
async def test_openai_delegation_bridge_is_nonblocking_and_resolves():
    gate = asyncio.Event()
    bridge = _FakeBrain("the bottle is on the left shelf", gate=gate)
    frames = [("0.0s", "data:image/jpeg;base64,AAA")]
    task_id = await bridge.submit("where is the bottle?", "let me check", frames)

    # poll must not block on the slow background brain
    assert (await bridge.poll(task_id)).status == "pending"

    gate.set()
    await asyncio.sleep(0)
    for _ in range(10):
        result = await bridge.poll(task_id)
        if result.status != "pending":
            break
        await asyncio.sleep(0)
    assert result.is_ready
    assert "bottle is on the left shelf" in result.digest
    assert (await bridge.poll("missing")).status == "error"


class _FakeImageBrain(ImageGenDelegationBridge):
    async def _call_images(self, payload):
        # echo the prompt back so the test can assert it was passed through
        return {"data": [{"b64_json": "QUJD"}]}  # "ABC"


@pytest.mark.asyncio
async def test_image_delegation_bridge_returns_media_not_text():
    bridge = _FakeImageBrain("http://unused/v1")
    task_id = await bridge.submit("draw a red dragon", "", [])
    for _ in range(10):
        result = await bridge.poll(task_id)
        if result.status != "pending":
            break
        await asyncio.sleep(0)
    assert result.is_ready
    # the image rides in media as a data URL; digest is only a short text placeholder
    assert result.media == "data:image/png;base64,QUJD"
    assert "QUJD" not in result.digest and "red dragon" in result.digest


class _FakeEditBrain(ImageEditDelegationBridge):
    def __init__(self, base_url, model):
        super().__init__(base_url, model)
        self.seen_frame = None

    async def _call_chat(self, frame, instruction):
        self.seen_frame = frame
        return {"choices": [{"message": {"content": [{"image_url": {"url": "data:image/png;base64,Q0FSVA=="}}]}}]}


@pytest.mark.asyncio
async def test_image_edit_delegation_conditions_on_frame_and_returns_media():
    bridge = _FakeEditBrain("http://unused/v1", "Qwen-Image-Edit")
    frames = [("0.0s", "data:image/jpeg;base64,OLD"), ("1.0s", "data:image/jpeg;base64,NOW")]
    task_id = await bridge.submit("把画面变成卡通风格", "", frames)
    for _ in range(10):
        result = await bridge.poll(task_id)
        if result.status != "pending":
            break
        await asyncio.sleep(0)
    assert result.is_ready
    assert bridge.seen_frame == "data:image/jpeg;base64,NOW"  # latest frame, not the old one
    assert result.media == "data:image/png;base64,Q0FSVA=="
    assert result.media not in result.digest  # image rides in media, not context


@pytest.mark.asyncio
async def test_image_edit_delegation_errors_without_frame():
    bridge = _FakeEditBrain("http://unused/v1", "Qwen-Image-Edit")
    task_id = await bridge.submit("cartoonify", "", [])
    for _ in range(10):
        result = await bridge.poll(task_id)
        if result.status != "pending":
            break
        await asyncio.sleep(0)
    assert result.status == "error"


class _NamedBridge:
    def __init__(self, name):
        self.name = name

    async def submit(self, question, note, frames):
        return f"{self.name}-1"

    async def poll(self, task_id):
        return DelegationResult(task_id, "ready", digest=self.name)


@pytest.mark.asyncio
async def test_routing_delegation_dispatches_by_request():
    from vllm_omni.experimental.fullduplex.joyvl.bridges.delegation import RoutingDelegationBridge

    chat, image, edit = _NamedBridge("chat"), _NamedBridge("image"), _NamedBridge("edit")
    r = RoutingDelegationBridge(chat=chat, image=image, edit=edit)

    async def route(q):
        tid = await r.submit(q, "", [])
        return (await r.poll(tid)).digest

    assert await route("把你看到的画面变成卡通风格") == "edit"  # restyle current view
    assert await route("画一只红色的龙") == "image"  # generate a new image
    assert await route("这道数学题的答案是什么") == "chat"  # hard question -> chat brain


class _Delegation:
    async def submit(self, question, note, frames):
        return "t1"

    async def poll(self, task_id):
        return DelegationResult(task_id, "ready", digest="background answer")


def test_build_delegation_off_unless_backend_configured():
    from vllm_omni.experimental.fullduplex.joyvl.serving.config import InteractionConfig
    from vllm_omni.experimental.fullduplex.joyvl.serving.server import SessionManager

    # No backend URL -> delegation disabled (None), NOT a silent stub that folds fake answers.
    assert SessionManager._build_delegation(InteractionConfig()) is None
    assert SessionManager._build_delegation(InteractionConfig(delegation_kind="router")) is None
    # Stub only on explicit opt-in.
    assert SessionManager._build_delegation(InteractionConfig(delegation_kind="stub")) is not None
    # A real chat backend -> a live bridge.
    cfg = InteractionConfig(delegation_backend_url="http://x/v1", delegation_model="m")
    assert SessionManager._build_delegation(cfg) is not None


@pytest.mark.asyncio
async def test_delegation_cancel_drops_task():
    from vllm_omni.experimental.fullduplex.joyvl.bridges.delegation import StubDelegationBridge

    b = StubDelegationBridge(ready_after_ticks=1)
    task_id = await b.submit("q", "", [])
    b.cancel(task_id)
    # after cancel the task is gone -> poll reports it as unknown rather than leaking
    assert (await b.poll(task_id)).status == "error"


@pytest.mark.asyncio
async def test_delegation_aclose_clears_pending():
    from vllm_omni.experimental.fullduplex.joyvl.bridges.delegation import StubDelegationBridge

    b = StubDelegationBridge(ready_after_ticks=5)
    task_id = await b.submit("q", "", [])
    await b.aclose()
    assert (await b.poll(task_id)).status == "error"


@pytest.mark.asyncio
async def test_session_reset_cancels_and_awaits_consolidation():
    from vllm_omni.experimental.fullduplex.joyvl.serving.config import InteractionConfig
    from vllm_omni.experimental.fullduplex.joyvl.serving.session import InteractionSession

    class _Backend:
        async def generate(self, *a, **k):
            return "</silence>", None

        async def aclose(self):
            pass

    class _Summarizer:
        async def summarize_chunk(self, *a, **k):
            await asyncio.sleep(100)  # hang so the consolidation task is in-flight
            return "x"

        async def compress_to_long_term(self, *a, **k):
            return ""

        async def aclose(self):
            pass

    sess = InteractionSession("s", InteractionConfig(), _Backend(), summarizer=_Summarizer())
    sess._spawn_consolidation(1, [("0.0s", "u")])
    assert len(sess._consolidating) == 1
    await sess.reset()  # must cancel AND await the in-flight task, not just clear the set
    assert len(sess._consolidating) == 0


@pytest.mark.asyncio
async def test_manager_aclose_closes_without_error():
    from vllm_omni.experimental.fullduplex.joyvl.serving.config import InteractionConfig
    from vllm_omni.experimental.fullduplex.joyvl.serving.server import SessionManager

    mgr = SessionManager(InteractionConfig(enable_memory=False, enable_delegation=False))
    await mgr.aclose()
    await mgr.aclose()  # idempotent


@pytest.mark.asyncio
async def test_delegation_submit_and_fold():
    p = JoyVLPolicy(delegation=_Delegation())
    action = p.commit("</response> hold on <delegation> hard question")
    assert action.action.value == "delegate"
    info = await p.submit_if_delegate(action)
    assert info["status"] == "submitted"
    folded = await p.fold_delegations()
    assert folded["status"] == "ready"
    assert p.brain.memory.qa_history[-1].responses[0][1] == "background answer"
