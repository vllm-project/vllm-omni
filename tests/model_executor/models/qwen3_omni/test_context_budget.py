# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError
from vllm_omni.engine.duplex.session.context_ledger import DuplexSourceRef
from vllm_omni.model_executor.models.qwen3_omni.duplex.context_budget import ContextSummary, select_history

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _messages(*roles):
    return [{"role": role, "content": str(i), "_src": i} for i, role in enumerate(roles)]


def _cost(messages, summary):
    # Ten tokens per message makes budget boundaries independent of tokenizer.
    return 10 * len(messages) + (15 if summary else 0)


def test_exact_budget_keeps_more_than_legacy_sixteen_messages():
    messages = _messages(*(["user", "assistant"] * 12), "user")
    selected = select_history(messages, limit=270, generation_tokens=20, count=_cost)
    assert selected.messages == messages
    assert selected.prompt_tokens == 250
    assert selected.omitted == ()


def test_one_token_overflow_drops_a_complete_old_turn():
    messages = _messages("user", "assistant", "user", "assistant", "user")
    sources = tuple(DuplexSourceRef(i + 100, 1) for i in range(5))
    selected = select_history(messages, limit=69, generation_tokens=20, count=_cost, sources=sources)
    assert selected.messages == messages[2:]
    assert selected.omitted == sources[:2]
    assert selected.prompt_tokens + 20 <= 69


@pytest.mark.parametrize("limit,reserve", [(29, 20), (20, 20), (21, 0)])
def test_mandatory_overflow_is_explicit(limit, reserve):
    with pytest.raises(DuplexRuntimeConfigError) as error:
        select_history(_messages("user"), limit=limit, generation_tokens=reserve, count=_cost)
    assert error.value.code == "context_budget_exceeded"


def test_system_and_consecutive_current_users_survive_as_a_unit():
    messages = _messages("user", "assistant", "system", "user", "user", "system")
    selected = select_history(messages, limit=60, generation_tokens=20, count=_cost)
    assert selected.messages == messages[2:]


def test_pending_tool_pair_pins_original_user_turn_even_with_newer_input():
    messages = _messages("user", "assistant", "user", "assistant", "user")
    selected = select_history(
        messages,
        limit=50,
        generation_tokens=20,
        count=_cost,
        tool_turns=[{"history_len": 2, "messages": []}],
    )
    assert selected.messages == [*messages[:2], messages[-1]]


def test_recent_turn_precedes_summary_and_summary_precedes_older_verbatim():
    messages = _messages(*(["user", "assistant"] * 4), "user")
    sources = tuple(DuplexSourceRef(i + 100, 1) for i in range(len(messages)))
    summary = ContextSummary("User prefers quiet hotels.", sources[:4], 1)
    selected = select_history(messages, limit=65, generation_tokens=20, count=_cost, sources=sources, summary=summary)
    assert selected.messages == messages[6:]
    assert selected.summary == summary
    assert selected.prompt_tokens == 45
    assert selected.omitted == sources[:6]


def test_summary_that_does_not_fit_falls_back_to_recent_verbatim():
    messages = _messages(*(["user", "assistant"] * 3), "user")
    summary = ContextSummary("Earlier preference", (DuplexSourceRef(100, 1),), 1)
    selected = select_history(messages, limit=50, generation_tokens=20, count=_cost, summary=summary)
    assert selected.messages == messages[-3:]
    assert selected.summary is None


def test_summary_never_duplicates_a_retained_source():
    messages = _messages(*(["user", "assistant"] * 3), "user")
    sources = tuple(DuplexSourceRef(i + 100, 1) for i in range(len(messages)))
    summary = ContextSummary("Recent preference", sources[4:6], 1)
    selected = select_history(messages, limit=65, generation_tokens=20, count=_cost, sources=sources, summary=summary)
    assert selected.messages == messages[-3:]
    assert selected.summary is None


def test_full_verbatim_history_wins_when_it_fits():
    messages = _messages("user", "assistant", "user")
    summary = ContextSummary("Earlier preference", (DuplexSourceRef(100, 1),), 1)
    selected = select_history(messages, limit=100, generation_tokens=20, count=_cost, summary=summary)
    assert selected.messages == messages
    assert selected.summary is None
