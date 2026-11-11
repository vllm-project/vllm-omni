# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The plugin prepares results using session-owned call metadata."""

from copy import deepcopy

import pytest

from tests.engine.duplex.test_session_runner_nemotron_voicechat import (
    TIMEOUT_S,
    answer,
    error_codes,
    errors,
    generation,
    open_timed_harness,
    output_item,
    prepare,
)
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError
from vllm_omni.engine.duplex.session.tool_calls import ToolCallOrigin, ToolCallState
from vllm_omni.protocol.realtime.events import ItemCreated

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("forged_hints", [False, True])
async def test_plugin_gets_trusted_context_without_changing_native_candidate_or_wire(monkeypatch, forged_hints):
    async with open_timed_harness(monkeypatch) as h:
        call_id = await prepare(h)
        origin = h.runner.tool_calls.record(call_id).origin
        assert origin.name and isinstance(origin.arguments, str)
        hints = {"name": "forged_name", "arguments": '{"forged":true}'} if forged_hints else {}
        submitted = output_item(call_id, **hints)
        seen = []
        real_convert = h.runner.plugin.runtime_config_for_function_output

        def convert(config, current, prepared, **kwargs):
            seen.append(deepcopy(prepared))
            expected_native_candidate = real_convert(config, deepcopy(current), submitted, **kwargs)
            candidate = real_convert(config, current, prepared, **kwargs)
            assert candidate == expected_native_candidate
            # A plugin's writes to its preparation copy must not rewrite the
            # client's item or the acceptance acknowledgement, even nested data.
            prepared["output"] = "plugin-local output"
            prepared["content"].append({"type": "text", "text": "plugin-local metadata"})
            return candidate

        monkeypatch.setattr(h.runner.plugin, "runtime_config_for_function_output", convert)
        events = await h.run(commands.CreateItem(item=submitted))
        assert error_codes(events) == []
        assert seen[0].get("name") == origin.name
        assert seen[0].get("arguments") == origin.arguments
        assert seen[0]["call_id"] == call_id
        created = next(event for event in events if isinstance(event, ItemCreated))
        assert created.item == submitted
        assert submitted["output"] == "20" and submitted["content"] == []
        assert generation(h) == 1


@pytest.mark.asyncio
async def test_retry_uses_same_origin_despite_changing_client_hints(monkeypatch):
    async with open_timed_harness(monkeypatch) as h:
        call_id = await prepare(h)
        origin = h.runner.tool_calls.record(call_id).origin
        seen = []
        real_convert = h.runner.plugin.runtime_config_for_function_output

        def convert(config, current, prepared, **kwargs):
            seen.append((prepared.get("name"), prepared.get("arguments")))
            if len(seen) == 1:
                raise DuplexRuntimeConfigError("probe refuses the first preparation", code="prepare_refused")
            return real_convert(config, current, prepared, **kwargs)

        monkeypatch.setattr(h.runner.plugin, "runtime_config_for_function_output", convert)
        refused = await h.run(
            commands.CreateItem(item=output_item(call_id, name="first_forgery"), event_id="context-first")
        )
        assert error_codes(refused) == ["prepare_refused"]
        assert errors(refused)[0].related_event_id == "context-first"
        assert generation(h) == 0
        accepted = await answer(h, call_id, name="different_forgery", arguments="false")
        assert error_codes(accepted) == []
        assert seen == [(origin.name, origin.arguments)] * 2
        assert generation(h) == 1


@pytest.mark.asyncio
async def test_metadata_less_origin_never_falls_back_to_client_hints(monkeypatch):
    async with open_timed_harness(monkeypatch) as h:
        await prepare(h)
        call_id = "legacy-origin"
        h.runner.tool_calls.register(
            ToolCallOrigin(
                call_id=call_id,
                epoch=h.session.epoch,
                turn_id=h.session.turn_id,
                response_id="legacy-response",
                item_id="legacy-item",
            )
        )
        seen = []
        real_convert = h.runner.plugin.runtime_config_for_function_output

        def convert(config, current, prepared, **kwargs):
            seen.append(dict(prepared))
            return real_convert(config, current, prepared, **kwargs)

        monkeypatch.setattr(h.runner.plugin, "runtime_config_for_function_output", convert)
        assert error_codes(await answer(h, call_id, name="forged_name", arguments="{}")) == []
        assert "name" not in seen[0] and "arguments" not in seen[0]


@pytest.mark.asyncio
async def test_missing_context_refusals_preserve_the_original_deadline(monkeypatch):
    async with open_timed_harness(monkeypatch) as h:
        await prepare(h)
        call_id = "legacy-requires-context"
        call = h.runner.tool_calls.register(
            ToolCallOrigin(
                call_id=call_id,
                epoch=h.session.epoch,
                turn_id=h.session.turn_id,
                response_id="legacy-response",
                item_id="legacy-item",
            )
        )
        original_deadline = call.deadline

        def require_context(config, current, prepared, **kwargs):
            del kwargs
            assert "name" not in prepared and "arguments" not in prepared
            raise DuplexRuntimeConfigError("session-owned metadata is required", code="missing_call_context")

        monkeypatch.setattr(h.runner.plugin, "runtime_config_for_function_output", require_context)
        for index in range(2):
            h.clock.advance(1.0)
            events = await h.run(
                commands.CreateItem(
                    item=output_item(call_id, name="forged", arguments="{}"), event_id=f"missing-context-{index}"
                )
            )
            assert error_codes(events) == ["missing_call_context"]
            assert errors(events)[0].related_event_id == f"missing-context-{index}"
            assert call.state is ToolCallState.PENDING and call.deadline == original_deadline
            assert generation(h) == 0
        h.clock.advance(TIMEOUT_S - 2.0)
        h.scheduler.fire_all()
        await h.settle()
        assert call.state is ToolCallState.TIMED_OUT
        assert error_codes(await answer(h, call_id)) == ["invalid_function_call_output"]
        assert generation(h) == 0
