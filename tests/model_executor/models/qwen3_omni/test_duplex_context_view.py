# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Any

import pytest

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.config import DuplexContextViewPolicy, DuplexSessionConfig
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


class Conversation:
    """Feeds a Qwen3-Omni plugin the prompt config model_channel would build."""

    def __init__(self, keep_media_user_turns: int | None = None) -> None:
        self.session = DuplexEngineSession(session_id="qwen-view", config=DuplexSessionConfig(model="qwen"))
        self.plugin = Qwen3OmniDuplexPlugin(lambda *args: None)
        self.state = self.plugin.create_session_state()
        self.policy = (
            None
            if keep_media_user_turns is None
            else DuplexContextViewPolicy(keep_media_user_turns=keep_media_user_turns)
        )

    def image(self, item_id: str) -> None:
        self.session.register_history_item(item_id, {"role": "user", "content": [dict(IMAGE)]})

    def reply(self, item_id: str, text: str) -> None:
        self.session.register_history_item(item_id, {"role": "assistant", "content": text})

    def audio(self, item_id: str, audio: str, *, transcript: str | None = None) -> list[dict[str, Any]]:
        committed = self.session.commit_audio_input(transcript=transcript)
        self.session.register_history_item(item_id, committed.message)
        config: dict[str, Any] = {"conversation": list(self.session.history)}
        if self.policy is not None:
            config["conversation_view"] = list(self.session.render_history_view(self.policy).messages)
        return self.plugin.prepare_prompt_config(config, state=self.state, payload={"audio": audio})["qwen_messages"]


def _plain(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace each audio payload object with its audio string."""
    plain = []
    for message in messages:
        payload = message.get("audio_payload")
        if isinstance(payload, dict):
            message = {"role": message["role"], "audio": payload["audio"], "_src": message["_src"]}
        plain.append(message)
    return plain


def _image_parts(messages: list[dict[str, Any]]) -> int:
    parts = [part for message in messages if isinstance(message.get("content"), list) for part in message["content"]]
    return sum(part.get("type") == "image_url" for part in parts)


def _camera_session(conversation: Conversation) -> list[dict[str, Any]]:
    conversation.image("u1_image")
    conversation.audio("u1", "A1", transcript="what is this")
    conversation.reply("a1", "a lamp")
    conversation.session.append_history_message({"role": "user", "content": "typed question"})
    conversation.reply("a2", "yes")
    conversation.image("u3_image")
    return conversation.audio("u3", "A3", transcript="and this")


#: ``qwen_messages`` for ``_camera_session`` at #9311 head a463ba2, before L1 existed.
CAMERA_SESSION_PROMPT = [
    {"role": "user", "content": [IMAGE], "_src": 0},
    {"role": "user", "audio": "A1", "_src": 1},
    {"role": "assistant", "content": "a lamp", "_src": 2},
    {"role": "user", "content": "typed question", "_src": 3},
    {"role": "assistant", "content": "yes", "_src": 4},
    {"role": "user", "content": [IMAGE], "_src": 5},
    {"role": "user", "audio": "A3", "_src": 6},
]


def test_qwen3_policy_follows_knob():
    plugin = Qwen3OmniDuplexPlugin(lambda *args: None)
    knob = DuplexSessionRuntimeConfig(context_keep_media_turns=2)

    assert plugin.context_view_policy(stage_id=0, runtime_config=DuplexSessionRuntimeConfig()) is None
    assert plugin.context_view_policy(stage_id=0, runtime_config=knob) == DuplexContextViewPolicy(
        keep_media_user_turns=2
    )
    assert plugin.context_view_policy(stage_id=1, runtime_config=knob) is None
    assert plugin.context_view_policy(stage_id=2, runtime_config=knob) is None


def test_qwen3_prompt_identical_with_knob_off():
    assert _plain(_camera_session(Conversation())) == CAMERA_SESSION_PROMPT


def test_keep_zero_tool_followup_keeps_retained_current_audio():
    conversation = Conversation(keep_media_user_turns=0)
    conversation.audio("u1", "A1", transcript="current question")
    history = list(conversation.session.history)
    config = {
        "conversation": history,
        "conversation_view": list(conversation.session.render_history_view(conversation.policy).messages),
        "qwen_tool_turns": [
            {
                "history_len": len(history),
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "tool result"},
                ],
            }
        ],
    }

    prepared = conversation.plugin.prepare_prompt_config(
        config, state=conversation.state, payload={"type": "conversation"}
    )

    assert _plain(prepared["qwen_messages"]) == [{"role": "user", "audio": "A1", "_src": 0}]


def test_qwen3_ignores_misaligned_view():
    conversation = Conversation()
    _camera_session(conversation)
    conversation.state = conversation.plugin.create_session_state()
    history = list(conversation.session.history)
    view = [{"role": "user", "content": []}]

    for prefix_len, payload in ((2, "A1"), (len(history), "A3")):
        config = {"conversation": history[:prefix_len], "conversation_view": view}
        prepared = conversation.plugin.prepare_prompt_config(
            config, state=conversation.state, payload={"audio": payload}
        )

    assert _plain(prepared["qwen_messages"]) == CAMERA_SESSION_PROMPT


def test_current_retained_audio_survives_system_item_before_response():
    conversation = Conversation(keep_media_user_turns=0)
    conversation.audio("u1", "A1", transcript="current question")
    conversation.session.append_history_message({"role": "system", "content": "answer briefly"})
    config = {
        "conversation": list(conversation.session.history),
        "conversation_view": list(conversation.session.render_history_view(conversation.policy).messages),
    }

    prepared = conversation.plugin.prepare_prompt_config(
        config, state=conversation.state, payload={"type": "conversation"}
    )

    assert _plain(prepared["qwen_messages"]) == [
        {"role": "user", "audio": "A1", "_src": 0},
        {"role": "system", "content": "answer briefly", "_src": 1},
    ]


def test_qwen3_skips_old_image_items_with_knob_on():
    messages = _camera_session(Conversation(keep_media_user_turns=1))

    assert _plain(messages) == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "what is this"}],
            "transcript": "what is this",
            "_src": 1,
        },
        *CAMERA_SESSION_PROMPT[2:],
    ]
    assert _image_parts(messages) == 1


@pytest.mark.parametrize(
    ("keep", "transcript", "first"),
    [
        (None, "q1", {"role": "user", "audio": "A2", "_src": 2}),
        (2, "q1", {"role": "user", "content": [{"type": "text", "text": "q1"}], "transcript": "q1", "_src": 0}),
        (2, None, {"role": "user", "audio": "A2", "_src": 2}),
    ],
    ids=["knob_off_drops_turn", "knob_on_keeps_transcript", "knob_on_without_transcript_drops_turn"],
)
def test_qwen3_keeps_transcript_for_evicted_audio_turn(keep, transcript, first):
    conversation = Conversation(keep)
    conversation.audio("u1", "A1", transcript=transcript)
    for turn in range(2, 6):
        conversation.reply(f"a{turn - 1}", f"r{turn - 1}")
        messages = conversation.audio(f"u{turn}", f"A{turn}")

    plain = _plain(messages)
    assert plain[0] == first
    if first["_src"] == 0:
        assert plain[1] == {"role": "assistant", "content": "r1", "_src": 1}
    assert [message["audio"] for message in plain if "audio" in message] == ["A2", "A3", "A4", "A5"]


@pytest.mark.parametrize(
    ("keep", "audios", "texts"),
    [
        (2, ["A9", "A10"], ["q2", "q3", "q4", "q5", "q6", "q7", "q8"]),
        (20, ["A7", "A8", "A9", "A10"], ["q2", "q3", "q4", "q5", "q6"]),
    ],
    ids=["l1_trims_retained_audio", "l1_preserves_existing_audio_cap"],
)
def test_qwen3_l1_respects_existing_windows(keep, audios, texts):
    conversation = Conversation(keep_media_user_turns=keep)
    conversation.audio("u1", "A1", transcript="q1")
    for turn in range(2, 11):
        conversation.reply(f"a{turn - 1}", f"r{turn - 1}")
        messages = conversation.audio(f"u{turn}", f"A{turn}", transcript=f"q{turn}")

    plain = _plain(messages)
    # 19 history messages: the 16-message window, rounded to whole turns, starts at u2.
    assert min(message["_src"] for message in plain) == 2
    assert [message["audio"] for message in plain if "audio" in message] == audios
    assert [message["content"][0]["text"] for message in plain if isinstance(message.get("content"), list)] == texts
