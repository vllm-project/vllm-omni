# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The vLLM 0.28 append capability probe must not install compatibility APIs."""

from __future__ import annotations

import importlib
import inspect
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_omni.engine.kv_append import (
    scheduler_native_append_available,
    scheduler_native_append_unavailable_reason,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MISSING = object()
_LEGACY_ATTRIBUTES = (
    ("vllm.v1.engine.core", "EngineCore", "append_streaming_prompt_unit"),
    ("vllm.v1.engine.core", "EngineCore", "append_streaming_prompt_tokens_with_metadata"),
    ("vllm.v1.engine.core_client", "AsyncMPClient", "add_streaming_prompt_request_async"),
    ("vllm.v1.request", "RequestStatus", "WAITING_FOR_STREAMING_PROMPT"),
)


def _legacy_attribute_snapshot() -> tuple[object, ...]:
    return tuple(
        inspect.getattr_static(getattr(importlib.import_module(module), owner), attribute, _MISSING)
        for module, owner, attribute in _LEGACY_ATTRIBUTES
    )


def test_import_does_not_modify_upstream_append_attributes_in_fresh_process() -> None:
    # The parent pytest process may already have imported Omni. Take the
    # baseline in a child, before importing Omni or running its capability
    # probe. Deliberately do not assert about unrelated vllm_omni.patch hooks.
    script = """
import importlib
import inspect
import sys

attributes = (
    ("vllm.v1.engine.core", "EngineCore", "append_streaming_prompt_unit"),
    ("vllm.v1.engine.core", "EngineCore", "append_streaming_prompt_tokens_with_metadata"),
    ("vllm.v1.engine.core_client", "AsyncMPClient", "add_streaming_prompt_request_async"),
    ("vllm.v1.request", "RequestStatus", "WAITING_FOR_STREAMING_PROMPT"),
)
missing = object()

def snapshot():
    return tuple(
        inspect.getattr_static(getattr(importlib.import_module(module), owner), attribute, missing)
        for module, owner, attribute in attributes
    )

before = snapshot()
assert "vllm_omni" not in sys.modules, "upstream imports loaded Omni before the baseline"
import vllm_omni
from vllm_omni.engine.kv_append import (
    scheduler_native_append_available,
    scheduler_native_append_unavailable_reason,
)

assert scheduler_native_append_available(), scheduler_native_append_unavailable_reason()
assert scheduler_native_append_unavailable_reason() is None
after = snapshot()
for target, previous, current in zip(attributes, before, after, strict=True):
    assert previous is current, f"Omni import/probe modified {target}"
print("kv-append-import-boundary-ok")
"""
    env = os.environ.copy()
    # Prevent automatic plugin discovery from importing Omni before the
    # baseline, and ignore conflicting visibility settings from other tests.
    env["VLLM_PLUGINS"] = ""
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("HIP_VISIBLE_DEVICES", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "kv-append-import-boundary-ok" in result.stdout


def test_probe_accepts_installed_retained_request_contract_without_mutation() -> None:
    before = _legacy_attribute_snapshot()
    assert scheduler_native_append_unavailable_reason() is None
    assert scheduler_native_append_available() is True
    assert all(previous is current for previous, current in zip(before, _legacy_attribute_snapshot(), strict=True))


@pytest.mark.parametrize(
    "module_name,owner_name,method_name",
    [
        ("vllm.v1.engine.core_client", "AsyncMPClient", "add_request_async"),
        ("vllm.v1.engine.core_client", "AsyncMPClient", "call_utility_async"),
        ("vllm.v1.core.sched.scheduler", "Scheduler", "_update_request_as_session"),
    ],
)
def test_probe_fails_closed_without_required_method(monkeypatch, module_name, owner_name, method_name) -> None:
    owner = getattr(importlib.import_module(module_name), owner_name)
    before = _legacy_attribute_snapshot()
    monkeypatch.setattr(owner, method_name, None)

    reason = scheduler_native_append_unavailable_reason()
    assert reason is not None and f"{owner_name}.{method_name}" in reason
    assert scheduler_native_append_available() is False
    assert getattr(owner, method_name) is None
    assert all(previous is current for previous, current in zip(before, _legacy_attribute_snapshot(), strict=True))


@pytest.mark.parametrize(
    "module_name,owner_name,field_name,expected_reason",
    [
        ("vllm.v1.core.sched.output", "CachedRequestData", "new_token_ids", "CachedRequestData.new_token_ids"),
        ("vllm.v1.request", "StreamingUpdate", "prompt_token_ids", "StreamingUpdate prompt fields"),
        ("vllm.v1.request", "StreamingUpdate", "max_tokens", "StreamingUpdate prompt fields"),
        ("vllm.v1.request", "StreamingUpdate", "arrival_time", "StreamingUpdate prompt fields"),
        ("vllm.v1.request", "StreamingUpdate", "sampling_params", "StreamingUpdate prompt fields"),
    ],
)
def test_probe_fails_closed_without_required_field(
    monkeypatch, module_name, owner_name, field_name, expected_reason
) -> None:
    owner = getattr(importlib.import_module(module_name), owner_name)
    fields = {name: field for name, field in owner.__dataclass_fields__.items() if name != field_name}
    before = _legacy_attribute_snapshot()
    monkeypatch.setattr(owner, "__dataclass_fields__", fields)

    reason = scheduler_native_append_unavailable_reason()
    assert reason is not None and expected_reason in reason
    assert scheduler_native_append_available() is False
    assert field_name not in owner.__dataclass_fields__
    assert all(previous is current for previous, current in zip(before, _legacy_attribute_snapshot(), strict=True))


def test_probe_fails_closed_without_streaming_status_and_does_not_invent_alias(monkeypatch) -> None:
    request_module = importlib.import_module("vllm.v1.request")
    statuses = SimpleNamespace()
    monkeypatch.setattr(request_module, "RequestStatus", statuses)

    reason = scheduler_native_append_unavailable_reason()
    assert reason is not None and "RequestStatus.WAITING_FOR_STREAMING_REQ" in reason
    assert scheduler_native_append_available() is False
    assert not hasattr(statuses, "WAITING_FOR_STREAMING_REQ")
    assert not hasattr(statuses, "WAITING_FOR_STREAMING_PROMPT")


def test_probe_fails_closed_when_streaming_update_cannot_be_imported(monkeypatch) -> None:
    request_module = importlib.import_module("vllm.v1.request")
    before = _legacy_attribute_snapshot()
    monkeypatch.delattr(request_module, "StreamingUpdate")

    reason = scheduler_native_append_unavailable_reason()
    assert reason is not None and "imports are unavailable" in reason
    assert scheduler_native_append_available() is False
    assert not hasattr(request_module, "StreamingUpdate")
    assert all(previous is current for previous, current in zip(before, _legacy_attribute_snapshot(), strict=True))
