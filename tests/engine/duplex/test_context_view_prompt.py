# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from typing import Any

import pytest

from tests.engine.duplex.test_session_runner import Harness, append_audio, close_harness, open_harness
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.config import DuplexContextViewPolicy
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import MiniCPMO45DuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

EARLIER_USER: dict[str, object] = {
    "role": "user",
    "content": [
        {"type": "audio_url", "audio_url": {"url": "native-duplex:input-audio"}, "transcript": "earlier"},
    ],
    "transcript": "earlier",
}
EARLIER_ASSISTANT: dict[str, object] = {"role": "assistant", "content": "reply"}


class RecordingPlugin(MiniCPMO45DuplexPlugin):
    """Records the prompt config each append hands to the model plugin."""

    def __init__(self, policies: dict[int, DuplexContextViewPolicy] | None = None) -> None:
        super().__init__(lambda *args: None)
        self.policies = dict(policies or {})
        self.prompt_configs: list[dict[str, object]] = []

    def context_view_policy(
        self, *, stage_id: int, runtime_config: DuplexSessionRuntimeConfig
    ) -> DuplexContextViewPolicy | None:
        del runtime_config
        return self.policies.get(stage_id)

    def prepare_prompt_config(self, config: dict[str, object], *, state: Any, payload: dict[str, object]):
        self.prompt_configs.append(dict(config))
        return super().prepare_prompt_config(config, state=state, payload=payload)


def _messages(value: object) -> list[dict[str, Any]]:
    assert isinstance(value, list)
    return value


def _seed_history(session: DuplexEngineSession) -> tuple[dict[str, object], ...]:
    session.register_history_item("u-earlier", dict(EARLIER_USER))
    session.register_history_item("a-earlier", dict(EARLIER_ASSISTANT))
    return session.history


async def _first_prompt_config(plugin: RecordingPlugin) -> tuple[Harness, tuple[dict[str, object], ...]]:
    h = await open_harness(plugin=plugin)
    seeded = _seed_history(h.session)
    await h.run(append_audio())
    assert plugin.prompt_configs, "the append never reached prepare_prompt_config"
    return h, seeded


def test_default_policy_is_none():
    plugin = MiniCPMO45DuplexPlugin(lambda *args: None)

    for stage_id in range(3):
        assert plugin.context_view_policy(stage_id=stage_id, runtime_config=DuplexSessionRuntimeConfig()) is None


@pytest.mark.asyncio
async def test_prompt_session_identical_without_policy():
    plugin = RecordingPlugin()
    h, seeded = await _first_prompt_config(plugin)
    try:
        config = plugin.prompt_configs[0]

        conversation = _messages(config["conversation"])

        assert "conversation_view" not in config
        assert conversation[:2] == list(seeded)
        for sent, held in zip(conversation[:2], seeded, strict=True):
            assert sent["content"] is held["content"]
        assert h.session.context_ledger.nodes() == ()
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_render_skipped_without_policy(monkeypatch):
    plugin = RecordingPlugin()
    h = await open_harness(plugin=plugin)
    try:
        _seed_history(h.session)

        def _fail(policy: DuplexContextViewPolicy) -> None:
            raise AssertionError("render_history_view() must not run without a policy")

        monkeypatch.setattr(h.session, "render_history_view", _fail)
        await h.run(append_audio())

        assert len(plugin.prompt_configs) == 1
        assert len(h.port.submissions) == 1
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_opted_in_append_captures_history_once(monkeypatch):
    plugin = RecordingPlugin({0: DuplexContextViewPolicy(keep_media_user_turns=0)})
    h = await open_harness(plugin=plugin)
    try:
        _seed_history(h.session)
        original = h.session.history_snapshot
        snapshots = []

        def capture():
            snapshot = original()
            snapshots.append(snapshot)
            return snapshot

        monkeypatch.setattr(h.session, "history_snapshot", capture)
        await h.run(append_audio())

        assert len(h.port.submissions) == 1
        assert len(snapshots) == 1
        config = plugin.prompt_configs[0]
        assert config["conversation_item_ids"] == snapshots[0].item_ids
        assert _messages(config["conversation_view"])[0]["content"] == [{"type": "text", "text": "earlier"}]
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_conversation_authoritative_with_policy():
    baseline_plugin = RecordingPlugin()
    baseline, _ = await _first_prompt_config(baseline_plugin)
    await close_harness(baseline)
    plugin = RecordingPlugin({0: DuplexContextViewPolicy(keep_media_user_turns=0)})
    h, seeded = await _first_prompt_config(plugin)
    try:
        config = plugin.prompt_configs[0]
        conversation = _messages(config["conversation"])
        view = _messages(config["conversation_view"])

        assert set(config) == set(baseline_plugin.prompt_configs[0]) | {"conversation_view"}
        assert conversation == baseline_plugin.prompt_configs[0]["conversation"]
        assert conversation[:2] == list(seeded)
        assert conversation[0]["content"] is seeded[0]["content"]
        assert len(view) == len(conversation)
        assert view[0]["content"] == [{"type": "text", "text": "earlier"}]
        assert view[1] == conversation[1]
        assert view[1] is not conversation[1]
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_policy_is_per_stage():
    plugin = RecordingPlugin({1: DuplexContextViewPolicy(keep_media_user_turns=0)})
    h, _ = await _first_prompt_config(plugin)
    try:
        assert "conversation_view" not in plugin.prompt_configs[0]
        assert h.session.context_ledger.nodes() == ()
    finally:
        await close_harness(h)
