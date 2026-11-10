# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Source-order and expired-media boundaries for budget selection."""

import pytest

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.session.context_ledger import DuplexSourceRef
from vllm_omni.model_executor.models.qwen3_omni.duplex.context_budget import ContextSummary, select_history, whole_turns
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _cost(messages, summary):
    return len(messages) * 10 + (10 if summary else 0)


def test_intervening_system_preserves_exchange_and_source_order():
    messages = [
        {"role": role, "content": str(i), "_src": i} for i, role in enumerate(["user", "system", "assistant", "user"])
    ]
    assert whole_turns(messages)[0] == [messages[0], messages[2]]
    selected = select_history(messages, limit=40, generation_tokens=10, count=_cost)
    assert selected.messages == [messages[1], messages[3]]
    selected = select_history(messages, limit=50, generation_tokens=10, count=_cost)
    assert selected.messages == messages


def test_summary_survives_when_expired_media_is_absent_from_fitting_candidates():
    sources = tuple(DuplexSourceRef(i, 0) for i in range(5))
    messages = [{"role": "user", "content": "current", "_src": 4}]
    summary = ContextSummary("The earlier audio transcript contained a preference.", sources[:2], 1)
    selected = select_history(messages, limit=100, generation_tokens=10, count=_cost, sources=sources, summary=summary)
    assert selected.summary == summary
    assert selected.omitted == sources[:4]


def test_missing_old_audio_does_not_leave_answer_attached_to_system():
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin.context_view_policy(stage_id=0, runtime_config=DuplexSessionRuntimeConfig(qwen_context_policy="budget"))
    history = [
        {"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "native-duplex:audio"}}]},
        {"role": "system", "content": "Be concise."},
        {"role": "assistant", "content": "orphaned answer"},
        {"role": "user", "content": "current"},
    ]
    config = plugin.prepare_prompt_config(
        {"conversation": history}, state=plugin.create_session_state(), payload={"type": "conversation"}
    )
    assert [m["content"] for m in config["qwen_messages"]] == ["Be concise.", "current"]


def test_tool_result_with_missing_original_media_fails_before_rendering():
    from vllm import SamplingParams

    from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError

    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin.context_view_policy(stage_id=0, runtime_config=DuplexSessionRuntimeConfig(qwen_context_policy="budget"))
    plugin.max_model_len = 4096
    conversation: list[dict[str, object]] = [
        {"role": "user", "content": "unrelated"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": [{"type": "audio_url"}]},
        {"role": "assistant", "content": "tool call"},
        {"role": "user", "content": "current"},
    ]
    with pytest.raises(DuplexRuntimeConfigError) as error:
        plugin.plan_append(
            request_id="missing-tool-source",
            fence=None,
            session_config={
                "conversation": conversation,
                "qwen_messages": [dict(conversation[i], _src=i) for i in (0, 1, 4)],
                "qwen_tool_turns": [{"history_len": 3, "messages": []}],
            },
            runtime_config={},
            seq=1,
            turn_seq=1,
            payload={"type": "conversation"},
            final=True,
            sampling_params=SamplingParams(max_tokens=128),
        )
    assert error.value.code == "context_source_unavailable"


@pytest.mark.parametrize("current", [(), (DuplexSourceRef(10, 1),)])
def test_tool_continuation_rejects_deleted_or_revised_source(current):
    from vllm import SamplingParams

    from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError
    from vllm_omni.model_executor.models.qwen3_omni.duplex.context_budget import (
        CONTEXT_SOURCES,
        CONTEXT_TOOL_INPUT,
        CONTEXT_TOOL_SOURCES,
    )

    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin.context_view_policy(stage_id=0, runtime_config=DuplexSessionRuntimeConfig(qwen_context_policy="budget"))
    plugin.max_model_len = 4096
    runtime = plugin.note_published_function_call(
        {CONTEXT_TOOL_INPUT: (DuplexSourceRef(10, 0),)}, call_id="call-1", history_len=1
    )
    assert runtime[CONTEXT_TOOL_SOURCES]["call-1"] == (DuplexSourceRef(10, 0),)
    history = [{"role": "user", "content": "replacement", "_src": 0}]
    pair = [{"role": "tool", "tool_call_id": "call-1", "content": "result"}]
    with pytest.raises(DuplexRuntimeConfigError) as error:
        plugin.plan_append(
            request_id="edited-tool-source",
            fence=None,
            session_config={
                "conversation": history,
                "qwen_messages": history,
                CONTEXT_SOURCES: current,
                "qwen_tool_turns": [{"history_len": 1, "messages": pair}],
            },
            runtime_config=runtime,
            seq=1,
            turn_seq=1,
            payload={"type": "conversation"},
            final=True,
            sampling_params=SamplingParams(max_tokens=128),
        )
    assert error.value.code == "context_source_unavailable"
