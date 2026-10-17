# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Session voice must survive Qwen duplex prompt construction."""

import base64
import struct

import pytest

from vllm_omni.engine.duplex.config import DuplexConfigError, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexAppendPlan, DuplexFence
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin
from vllm_omni.model_executor.stage_input_processors.tts_utils import extract_speaker_from_prompt

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def plugin(mocker) -> Qwen3OmniDuplexPlugin:
    plugin = Qwen3OmniDuplexPlugin(mocker.Mock())
    plugin.processor = mocker.Mock()
    plugin.processor.apply_chat_template.return_value = "rendered prompt"
    return plugin


def _append_plan(
    plugin: Qwen3OmniDuplexPlugin,
    config: DuplexSessionConfig,
    *,
    audio: bool = False,
) -> DuplexAppendPlan:
    payload = {"audio": base64.b64encode(struct.pack("<8f", *([0.25] * 8))).decode()} if audio else {}
    session_config = config.as_dict()
    session_config["conversation"] = [{"role": "user", "content": "Hello"}]
    session_config = plugin.prepare_prompt_config(session_config, state=plugin.create_session_state(), payload=payload)
    return plugin.plan_append(
        request_id="voice-test",
        fence=DuplexFence("voice-session"),
        session_config=session_config,
        runtime_config=plugin.runtime_config_for_update(config, {}),
        seq=1,
        turn_seq=1,
        payload=payload,
        final=True,
        sampling_params=(),
    )


@pytest.mark.parametrize("voice", ["aiden", "chelsie", "ethan"])
@pytest.mark.parametrize("audio", [False, True], ids=["text", "committed-audio"])
def test_selected_voice_is_forwarded(plugin, voice, audio):
    plan = _append_plan(plugin, DuplexSessionConfig(voice=voice), audio=audio)

    assert plan.prompt["additional_information"] == {"speaker": [voice]}
    assert extract_speaker_from_prompt(plan.prompt) == [voice]
    assert plan.prompt["prompt"] == "rendered prompt"
    assert bool(plan.prompt["multi_modal_data"].get("audio")) is audio


@pytest.mark.parametrize("voice,expected", [(" Chelsie ", "chelsie"), ("ETHAN\n", "ethan"), ("Default", "default")])
def test_voice_is_normalized_for_talker_lookup(plugin, voice, expected):
    plan = _append_plan(plugin, DuplexSessionConfig(voice=voice))

    assert plan.prompt["additional_information"] == {"speaker": [expected]}


@pytest.mark.parametrize("voice", [None, "", "   "])
def test_no_selected_voice_preserves_checkpoint_default(plugin, voice):
    plan = _append_plan(plugin, DuplexSessionConfig(voice=voice))

    assert "additional_information" not in plan.prompt


@pytest.mark.parametrize("patch", [{"voice": "Chelsie"}, {"audio": {"output": {"voice": "Chelsie"}}}])
def test_session_update_changes_the_next_prompt_voice(plugin, patch):
    config = DuplexSessionConfig(voice="aiden")
    previous = _append_plan(plugin, config)
    config.apply_realtime_update(patch, audio_started=False)

    current = _append_plan(plugin, config)

    assert current.prompt["additional_information"] == {"speaker": ["chelsie"]}
    assert previous.prompt["additional_information"] == {"speaker": ["aiden"]}


def test_clearing_voice_does_not_reuse_the_previous_speaker(plugin):
    config = DuplexSessionConfig(voice="chelsie")
    previous = _append_plan(plugin, config)
    config.apply_realtime_update({"voice": None}, audio_started=False)

    current = _append_plan(plugin, config)

    assert "additional_information" not in current.prompt
    assert previous.prompt["additional_information"] == {"speaker": ["chelsie"]}


def test_voice_update_after_audio_remains_rejected():
    config = DuplexSessionConfig(voice="aiden")

    with pytest.raises(DuplexConfigError) as exc:
        config.apply_realtime_update({"voice": "chelsie"}, audio_started=True)

    assert exc.value.code == "voice_update_after_audio_unsupported"
    assert config.voice == "aiden"


def test_shared_plugin_does_not_leak_voice_between_sessions(plugin):
    aiden = _append_plan(plugin, DuplexSessionConfig(voice="aiden"))
    chelsie = _append_plan(plugin, DuplexSessionConfig(voice="chelsie"))
    default = _append_plan(plugin, DuplexSessionConfig())

    assert aiden.prompt["additional_information"] == {"speaker": ["aiden"]}
    assert chelsie.prompt["additional_information"] == {"speaker": ["chelsie"]}
    assert "additional_information" not in default.prompt
