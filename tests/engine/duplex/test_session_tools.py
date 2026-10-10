# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Session-owned function-call ledger and the two plugin hooks.

The ledger does not execute tools and does not gate audio append.
"""

from __future__ import annotations

import pytest

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.plugin import DuplexModelPlugin
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession
from vllm_omni.engine.duplex.session.model_channel import (
    _INVALID_FUNCTION_CALL,
    _recognized_function_call,
)
from vllm_omni.engine.duplex.session.tools import (
    DuplexToolLedger,
    DuplexToolLedgerError,
    ToolCallStatus,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _ParsePlugin:
    def parse_function_call(self, model_result: dict[str, object]) -> dict[str, str] | None:
        if model_result.get("marker") != "call":
            return None
        call_id = model_result.get("call_id")
        name = model_result.get("name")
        if not isinstance(call_id, str) or not isinstance(name, str):
            return {"call_id": "", "name": ""}
        return {
            "call_id": call_id,
            "name": name,
            "arguments": str(model_result.get("arguments", "")),
        }

    def runtime_config_for_function_output(self, config, current, item):
        del config, current
        return {"fed": item.get("output")}


def _session() -> DuplexEngineSession:
    return DuplexEngineSession(session_id="sid", config=DuplexSessionConfig(model="fake-model"))


def test_default_plugin_hooks_are_noops() -> None:
    assert DuplexModelPlugin.parse_function_call(DuplexModelPlugin, {"function_call": True}) is None
    assert (
        DuplexModelPlugin.runtime_config_for_function_output(
            DuplexModelPlugin,
            DuplexSessionConfig(model="fake-model"),
            {},
            {"type": "function_call_output"},
        )
        is None
    )


def test_open_call_and_accept_result() -> None:
    ledger = DuplexToolLedger()
    opened = ledger.open_call(call_id="c1", name="lookup", arguments="{}", epoch=0)
    assert opened.status is ToolCallStatus.OPEN
    accepted = ledger.accept_result("c1", epoch=0)
    assert accepted.status is ToolCallStatus.COMPLETED


def test_duplicate_open_is_rejected() -> None:
    ledger = DuplexToolLedger()
    ledger.open_call(call_id="c1", name="lookup", arguments="{}", epoch=0)
    with pytest.raises(DuplexToolLedgerError) as exc:
        ledger.open_call(call_id="c1", name="lookup", arguments="{}", epoch=0)
    assert exc.value.code == "duplicate_function_call"


def test_unknown_and_duplicate_results() -> None:
    ledger = DuplexToolLedger()
    with pytest.raises(DuplexToolLedgerError) as missing:
        ledger.accept_result("missing", epoch=0)
    assert missing.value.code == "unknown_function_call"
    ledger.open_call(call_id="c1", name="lookup", arguments="{}", epoch=0)
    ledger.accept_result("c1", epoch=0)
    with pytest.raises(DuplexToolLedgerError) as duplicate:
        ledger.accept_result("c1", epoch=0)
    assert duplicate.value.code == "duplicate_function_call_output"


def test_cancel_then_late_result_is_not_accepted() -> None:
    ledger = DuplexToolLedger()
    ledger.open_call(call_id="c1", name="lookup", arguments="{}", epoch=0)
    cancelled = ledger.cancel("c1", reason="timeout")
    assert cancelled.status is ToolCallStatus.CANCELLED
    assert cancelled.cancel_reason == "timeout"
    with pytest.raises(DuplexToolLedgerError) as late:
        ledger.accept_result("c1", epoch=0)
    assert late.value.code == "late_function_call_output"
    assert ledger.get("c1").status is ToolCallStatus.CANCELLED


def test_epoch_change_makes_open_call_stale() -> None:
    session = _session()
    session.tool_ledger.open_call(call_id="c1", name="lookup", arguments="{}", epoch=session.epoch)
    session.barge_in()
    with pytest.raises(DuplexToolLedgerError) as stale:
        session.tool_ledger.accept_result("c1", epoch=session.epoch)
    assert stale.value.code == "stale_function_call_output"
    assert session.tool_ledger.get("c1").status is ToolCallStatus.STALE


def test_parse_hook_recognizes_a_call_and_legacy_flag_still_opens() -> None:
    plugin = _ParsePlugin()
    recognized = _recognized_function_call(plugin, {"marker": "call", "call_id": "c1", "name": "lookup"})
    assert recognized == {"call_id": "c1", "name": "lookup", "arguments": ""}
    assert _recognized_function_call(plugin, {"marker": "call"}) is _INVALID_FUNCTION_CALL
    legacy = _recognized_function_call(
        plugin,
        {"function_call": True, "call_id": "c2", "name": "lookup", "arguments": "{}"},
    )
    assert legacy == {"call_id": "c2", "name": "lookup", "arguments": "{}"}
    assert plugin.runtime_config_for_function_output(None, {}, {"output": "20"}) == {"fed": "20"}
