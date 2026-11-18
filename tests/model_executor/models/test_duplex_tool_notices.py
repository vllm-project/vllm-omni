# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Each tool-capable duplex plugin shows an abandoned call in its own format.

These cases cover the runtime config a plugin produces. They do not show how a
model behaves after reading a notice; that needs a real checkpoint.
"""

from __future__ import annotations

import json

import pytest

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import ToolCallAbandonReason
from vllm_omni.engine.duplex.session.tool_calls import ToolCallAbandonment, ToolCallOrigin
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.delegation import DELEGATION_TOOL_NAME
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.plugin import JoyAIDuplexPlugin
from vllm_omni.model_executor.models.nemotron_voicechat.duplex.plugin import NemotronVoiceChatDuplexPlugin
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CONFIG = DuplexSessionConfig(model="m")
REASONS_WITH_NOTICE = [
    ToolCallAbandonReason.TIMEOUT,
    ToolCallAbandonReason.CLIENT_CANCELLED,
    ToolCallAbandonReason.CLIENT_FAILED,
]


def abandonment(reason, *, name="get_weather", arguments='{"city":"Taipei"}'):
    origin = ToolCallOrigin(
        call_id="call_a",
        epoch=1,
        turn_id=2,
        response_id="resp_a",
        item_id="item_a",
        name=name,
        arguments=arguments,
    )
    return ToolCallAbandonment(origin, reason)


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]


# --------------------------------------------------------------------------- #
# Qwen3-Omni                                                                  #
# --------------------------------------------------------------------------- #


def qwen() -> Qwen3OmniDuplexPlugin:
    return Qwen3OmniDuplexPlugin(lambda *args: None)


@pytest.mark.parametrize(
    "reason",
    [*REASONS_WITH_NOTICE, ToolCallAbandonReason.EPOCH_CANCELLED],
)
def test_qwen_keeps_a_notice_at_the_calls_anchor_and_drops_the_anchor(reason):
    current = {"qwen_call_anchors": {"call_a": 3, "call_b": 4}}

    updated = qwen().runtime_config_for_abandoned_tool_call(CONFIG, current, abandonment(reason), history_len=9)

    turn = updated["qwen_tool_turns"][0]
    assert turn["history_len"] == 3
    assert turn["notice"] is True
    assert updated["qwen_call_anchors"] == {"call_b": 4}
    assistant, tool = turn["messages"]
    assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
    assert tool["tool_call_id"] == "call_a"
    assert json.loads(tool["content"])["reason"] == reason.value


def test_qwen_notice_alone_opens_no_follow_up():
    plugin = qwen()
    updated = plugin.runtime_config_for_abandoned_tool_call(
        CONFIG, {}, abandonment(ToolCallAbandonReason.TIMEOUT), history_len=2
    )
    assert plugin.has_pending_tool_followup(updated) is False
    updated["qwen_tool_turns"].append({"history_len": 2, "messages": []})
    assert plugin.has_pending_tool_followup(updated) is True


def test_qwen_notice_survives_the_prompt_splice_bookkeeping():
    plugin = qwen()
    updated = plugin.runtime_config_for_abandoned_tool_call(
        CONFIG, {}, abandonment(ToolCallAbandonReason.TIMEOUT), history_len=0
    )
    collected = plugin._collect_tool_turns(updated["qwen_tool_turns"])
    assert collected[0]["notice"] is True


def test_qwen_notice_is_dropped_once_a_response_stores_its_text():
    plugin = qwen()
    updated = plugin.runtime_config_for_abandoned_tool_call(
        CONFIG, {}, abandonment(ToolCallAbandonReason.TIMEOUT), history_len=2
    )
    turn = {**updated["qwen_tool_turns"][0], "response_id": "resp_a"}
    inflight = {"qwen_tool_turns_inflight": [turn]}

    cancelled = plugin.runtime_config_after_response(inflight, committed=False, response_id="resp_a")
    assert [item.get("notice") for item in cancelled["qwen_tool_turns"]] == [True]

    stored = plugin.runtime_config_after_response(inflight, committed=True, response_id="resp_a")
    assert "qwen_tool_turns_inflight" not in stored
    assert not stored.get("qwen_tool_turns")


def test_qwen_skips_a_suppressed_repeat_and_a_call_outside_the_window():
    plugin = qwen()
    suppressed = abandonment(ToolCallAbandonReason.SUPPRESSED)
    assert plugin.runtime_config_for_abandoned_tool_call(CONFIG, {}, suppressed, history_len=0) is None
    expired = {"qwen_expired_call_ids": ["call_a"]}
    timeout = abandonment(ToolCallAbandonReason.TIMEOUT)
    assert plugin.runtime_config_for_abandoned_tool_call(CONFIG, expired, timeout, history_len=0) is None


def test_qwen_capabilities_describe_a_turn_based_client_flow():
    capabilities = qwen().capabilities(max_sessions=1)
    assert capabilities.tool_mode == "declared"
    assert capabilities.function_call_output_continuation == "client_response_create"
    assert capabilities.clears_function_call_block_on_user_commit is True


# --------------------------------------------------------------------------- #
# Nemotron VoiceChat                                                          #
# --------------------------------------------------------------------------- #


def nemotron() -> NemotronVoiceChatDuplexPlugin:
    plugin = NemotronVoiceChatDuplexPlugin(lambda *_args: None)
    plugin._tokenizers["test-tokenizer"] = (None, _Tokenizer())
    return plugin


NEMOTRON_RUNTIME = {"nvc_tokenizer_ref": "test-tokenizer", "nvc_function_response_generation": 0}


@pytest.mark.parametrize("reason", [*REASONS_WITH_NOTICE, ToolCallAbandonReason.SUPPRESSED])
def test_nemotron_queues_a_notice_on_the_function_channel(reason):
    updated = nemotron().runtime_config_for_abandoned_tool_call(
        CONFIG, NEMOTRON_RUNTIME, abandonment(reason), history_len=0
    )
    batch = updated["nvc_function_response_batches"][0]
    text = "".join(chr(token_id) for token_id in batch["token_ids"])
    assert text.startswith("<TOOL_RESPONSE>")
    assert reason.value in text


def test_nemotron_leaves_a_cancelled_request_alone():
    cancelled = abandonment(ToolCallAbandonReason.EPOCH_CANCELLED)
    assert nemotron().runtime_config_for_abandoned_tool_call(CONFIG, NEMOTRON_RUNTIME, cancelled, history_len=0) is None


def test_nemotron_full_queue_skips_the_notice_instead_of_closing_the_session():
    current = {**NEMOTRON_RUNTIME, "nvc_function_response_batches": [{"generation": 0, "token_ids": [1]}] * 8}
    timeout = abandonment(ToolCallAbandonReason.TIMEOUT)
    assert nemotron().runtime_config_for_abandoned_tool_call(CONFIG, current, timeout, history_len=0) is None


def test_nemotron_capabilities_describe_a_native_flow():
    capabilities = nemotron().capabilities(max_sessions=1)
    assert capabilities.tool_mode == "declared"
    assert capabilities.function_call_output_continuation == "automatic"
    assert capabilities.max_tools == 5
    assert capabilities.clears_function_call_block_on_user_commit is False


# --------------------------------------------------------------------------- #
# JoyAI                                                                       #
# --------------------------------------------------------------------------- #


def joyai() -> JoyAIDuplexPlugin:
    return JoyAIDuplexPlugin(lambda *args: None)


QUESTION = json.dumps({"question": "Is the pot overheated?"})


@pytest.mark.parametrize("reason", [*REASONS_WITH_NOTICE, ToolCallAbandonReason.SUPPRESSED])
def test_joyai_carries_a_notice_to_the_next_observation(reason):
    item = abandonment(reason, name=DELEGATION_TOOL_NAME, arguments=QUESTION)
    updated = joyai().runtime_config_for_abandoned_tool_call(CONFIG, {}, item, history_len=4)
    record = updated["joyai_delegation_results"][0]
    assert record["question"] == "Is the pot overheated?"
    assert json.loads(record["output"])["reason"] == reason.value


def test_joyai_does_not_resurface_a_cancelled_question():
    item = abandonment(ToolCallAbandonReason.EPOCH_CANCELLED, name=DELEGATION_TOOL_NAME, arguments=QUESTION)
    assert joyai().runtime_config_for_abandoned_tool_call(CONFIG, {}, item, history_len=4) is None


def test_joyai_skips_a_call_without_a_question():
    item = abandonment(ToolCallAbandonReason.TIMEOUT, name=DELEGATION_TOOL_NAME, arguments="{}")
    assert joyai().runtime_config_for_abandoned_tool_call(CONFIG, {}, item, history_len=4) is None


def test_joyai_capabilities_describe_delegation():
    capabilities = joyai().capabilities(max_sessions=1)
    assert capabilities.tool_mode == "delegate_only"
    assert capabilities.function_call_output_continuation == "next_observation"
    assert capabilities.max_tools == 1
    assert capabilities.clears_function_call_block_on_user_commit is False
