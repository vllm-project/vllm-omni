# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Mixed context work must share an explicit, measured session allowance."""

import pytest

from tests.engine.duplex.test_qwen_context_memory import Processor, snapshot
from tests.engine.duplex.test_session_auxiliary import wait_until
from tests.engine.duplex.test_session_manager import FakeClock
from tests.engine.duplex.test_session_runner import close_harness, open_harness
from tests.model_executor.models.qwen3_omni.test_duplex_context_view_processor import (
    qwen_processor as _processor_fixture,
)
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.commands import Heartbeat
from vllm_omni.engine.duplex.contracts import duplex_is_auxiliary_request
from vllm_omni.engine.duplex.session.context_ledger import DuplexContextLedger, source_refs
from vllm_omni.engine.duplex.session.memory_runtime import SessionMemoryRuntime
from vllm_omni.model_executor.models.qwen3_omni.duplex.context_budget import CONTEXT_OMITTED
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
qwen_processor = _processor_fixture

_LIMITS = {
    "max_input_tokens": 8192,
    "max_output_tokens": 1024,
    "window_s": 60.0,
    "cooldown_s": 0.0,
    "idle_s": 0.0,
}


def _runtime(**changes):
    return DuplexSessionRuntimeConfig(
        **{
            "qwen_context_policy": "budget_summary",
            "qwen_summary_max_tokens": 128,
            "context_auxiliary_budget": dict(_LIMITS),
            **changes,
        }
    )


def _plugin(runtime):
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin.processor = Processor()
    plugin.max_model_len = 16384
    plugin.configure_session_runtime(runtime)
    return plugin


def test_default_shared_allowance_is_disabled():
    runtime = DuplexSessionRuntimeConfig()
    assert getattr(runtime, "context_auxiliary_budget", "missing") is None
    plugin = _plugin(runtime)
    assert plugin.auxiliary_budget_limits() is None


def test_mixed_extraction_requires_explicit_common_allowance():
    runtime = DuplexSessionRuntimeConfig(qwen_context_policy="budget_summary", context_memory={})
    with pytest.raises(ValueError, match="context_auxiliary_budget"):
        _plugin(runtime)


@pytest.mark.parametrize("mixed", [False, True])
def test_summary_and_cards_use_the_same_configured_limits(mixed):
    runtime = _runtime(context_memory={} if mixed else None)
    plugin = _plugin(runtime)
    common = plugin.auxiliary_budget_limits()
    assert common is not None
    for name, value in _LIMITS.items():
        assert getattr(common, name) == value
    if mixed:
        policy = plugin.context_auxiliary_policy(runtime_config={})
        assert policy is not None
        assert policy.compute_budget == common


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_input_tokens", True),
        ("max_input_tokens", 0),
        ("max_input_tokens", 2**31),
        ("max_output_tokens", -1),
        ("window_s", 0),
        ("window_s", float("inf")),
        ("cooldown_s", -1),
        ("idle_s", float("nan")),
        ("idle_s", False),
    ],
)
def test_invalid_common_limits_rejected_before_session(field, value):
    with pytest.raises(ValueError):
        _runtime(context_auxiliary_budget={**_LIMITS, field: value})


@pytest.mark.parametrize("missing", list(_LIMITS))
def test_common_limits_require_explicit_fields(missing):
    with pytest.raises(ValueError):
        _runtime(context_auxiliary_budget={key: value for key, value in _LIMITS.items() if key != missing})


@pytest.mark.parametrize("mixed", [False, True])
def test_common_allowance_must_fit_reserved_output(mixed):
    runtime = _runtime(
        context_memory={} if mixed else None, context_auxiliary_budget={**_LIMITS, "max_output_tokens": 127}
    )
    with pytest.raises(ValueError, match="context_auxiliary_budget"):
        _plugin(runtime)


def test_client_cannot_override_common_allowance():
    plugin = _plugin(DuplexSessionRuntimeConfig())
    with pytest.raises(ValueError):
        plugin.validate_client_extra_body({"context_auxiliary_budget": dict(_LIMITS)})


@pytest.mark.asyncio
async def test_summary_measured_ids_match_original_prompt_encoding():
    legacy = _plugin(DuplexSessionRuntimeConfig(qwen_context_policy="budget_summary", qwen_summary_max_tokens=128))
    shared = _plugin(_runtime())
    history = snapshot("The user likes tea.", "Noted.")
    refs = source_refs(history, *history.source_ids)
    requests = []
    for plugin in (legacy, shared):
        state = plugin.create_session_state()
        plugin.observe_history_context(
            snapshot=history,
            ledger=DuplexContextLedger(),
            state=state,
            runtime_config={CONTEXT_OMITTED: refs},
        )
        request = await plugin.prepare_auxiliary_request(history=(), runtime_config={}, state=state)
        assert request is not None
        requests.append(request)
    original, measured = requests
    assert original.prompt.keys() == {"prompt"}
    assert measured.prompt.keys() == {"prompt_token_ids"}
    expected = legacy.processor.tokenizer(text=[original.prompt["prompt"]], add_special_tokens=False)["input_ids"][0]
    assert measured.prompt["prompt_token_ids"] == expected
    assert SessionMemoryRuntime._measured_cost(measured) == (len(expected), 128)
    assert measured.model_source == original.model_source


@pytest.mark.asyncio
async def test_measured_summary_uses_pinned_processor_tokens(qwen_processor):
    original = _plugin(DuplexSessionRuntimeConfig(qwen_context_policy="budget_summary", qwen_summary_max_tokens=128))
    measured = _plugin(_runtime())
    history = snapshot("The user likes tea. 用户住杭州。", "Noted.")
    requests = []
    for plugin in (original, measured):
        plugin.processor = qwen_processor
        state = plugin.create_session_state()
        plugin.observe_history_context(
            snapshot=history,
            ledger=DuplexContextLedger(),
            state=state,
            runtime_config={CONTEXT_OMITTED: source_refs(history, *history.source_ids)},
        )
        request = await plugin.prepare_auxiliary_request(history=(), runtime_config={}, state=state)
        assert request is not None
        requests.append(request)
    prompt = requests[0].prompt["prompt"]
    assert requests[1].prompt == {"prompt_token_ids": qwen_processor.tokenizer.encode(prompt)}
    assert qwen_processor.tokenizer.decode(requests[1].prompt["prompt_token_ids"]) == prompt


def test_unsupported_plugin_rejects_common_allowance():
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import MiniCPMO45DuplexPlugin

    plugin = MiniCPMO45DuplexPlugin(lambda *args: "")
    with pytest.raises(ValueError, match="context_auxiliary_budget"):
        plugin.configure_session_runtime(_runtime())


@pytest.mark.asyncio
async def test_unsubmitted_summary_is_retryable_after_budget_returns():
    clock, runtime = FakeClock(), _runtime(context_auxiliary_budget={**_LIMITS, "window_s": 1.0})
    plugin = _plugin(runtime)
    h = await open_harness(plugin=plugin, auto_response=False, clock=clock, runtime_config=runtime)
    try:
        h.session.register_history_item("fact", {"role": "user", "content": "The user likes tea."})
        h.session.register_history_item("ack", {"role": "assistant", "content": "Noted."})
        history = h.session.history_snapshot()
        h.session.replace_runtime_config({CONTEXT_OMITTED: source_refs(history, *history.source_ids)})
        limits = plugin.auxiliary_budget_limits()
        assert limits is not None
        budget = h.session.context_compute_budget(limits)
        reservation = budget.try_reserve(limits.max_input_tokens - 1, 1, now=clock())
        assert reservation is not None
        assert budget.settle(reservation, submitted=True, now=clock())

        await h.run(Heartbeat())
        await wait_until(lambda: h.runner.memory._resource_waiting)
        assert h.runner.model_state.context_memory.attempted is None
        assert not [s for s in h.port.submissions if duplex_is_auxiliary_request(s.context.request_id)]

        clock.advance(limits.window_s)
        h.runner.memory.resources_available()
        await h.run(Heartbeat())
        await wait_until(lambda: any(duplex_is_auxiliary_request(s.context.request_id) for s in h.port.submissions))
        submissions = [s for s in h.port.submissions if duplex_is_auxiliary_request(s.context.request_id)]
        assert len(submissions) == 1
        assert h.runner.model_state.context_memory.attempted is not None
        assert budget.usage(now=clock()) == (len(submissions[0].prompt["prompt_token_ids"]), 128)
    finally:
        await close_harness(h)
