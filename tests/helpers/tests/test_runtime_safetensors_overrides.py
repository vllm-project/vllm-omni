# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""OmniServer and OmniRunner share the same Stage-0 prefetch merge."""

from __future__ import annotations

import json

import pytest

from tests.helpers.runtime import (
    OmniRunner,
    OmniServer,
    _inject_stage_safetensors_load_overrides,
    _merge_stage_safetensors_load_overrides,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_merge_fills_defaults_and_lets_caller_win() -> None:
    merged = _merge_stage_safetensors_load_overrides(
        {"0": {"gpu_memory_utilization": 0.5, "safetensors_load_strategy": "lazy"}, "3": {"devices": "0"}}
    )

    assert merged["0"]["safetensors_load_strategy"] == "lazy"
    assert merged["0"]["gpu_memory_utilization"] == 0.5
    assert merged["1"] == {"safetensors_load_strategy": "lazy"}
    assert merged["2"] == {"safetensors_load_strategy": "lazy"}
    assert merged["3"] == {"devices": "0"}


def test_server_merges_existing_stage_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tests.helpers.runtime.cleanup_test_environment", lambda: None)
    monkeypatch.setattr("tests.helpers.runtime.get_open_port", lambda: 9)

    server = OmniServer(
        "fake-model",
        ["--stage-overrides", json.dumps({"0": {"gpu_memory_utilization": 0.4}})],
    )
    idx = server.serve_args.index("--stage-overrides")
    parsed = json.loads(server.serve_args[idx + 1])

    assert parsed["0"]["gpu_memory_utilization"] == 0.4
    assert parsed["0"]["safetensors_load_strategy"] == "prefetch"
    assert parsed["1"]["safetensors_load_strategy"] == "lazy"


def test_server_equals_form_and_global_strategy_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tests.helpers.runtime.cleanup_test_environment", lambda: None)
    monkeypatch.setattr("tests.helpers.runtime.get_open_port", lambda: 9)

    equals = OmniServer("fake-model", ['--stage-overrides={"0":{"devices":"0"}}'])
    token = next(a for a in equals.serve_args if a.startswith("--stage-overrides="))
    parsed = json.loads(token.split("=", 1)[1])
    assert parsed["0"]["devices"] == "0"
    assert parsed["0"]["safetensors_load_strategy"] == "prefetch"

    skipped = OmniServer(
        "fake-model",
        ["--safetensors-load-strategy", "lazy", "--stage-overrides", "{}"],
    )
    assert skipped.serve_args[skipped.serve_args.index("--stage-overrides") + 1] == "{}"


def test_inject_helper_matches_runner_kwargs() -> None:
    existing = {"1": {"tensor_parallel_size": 2}}
    injected = _inject_stage_safetensors_load_overrides(
        ["--stage-overrides", json.dumps(existing)],
    )
    parsed = json.loads(injected[injected.index("--stage-overrides") + 1])
    assert parsed == _merge_stage_safetensors_load_overrides(existing)


def test_runner_merges_existing_stage_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeOmni:
        def __init__(self, *args, **kwargs) -> None:
            del args
            captured.update(kwargs)

    monkeypatch.setattr("tests.helpers.runtime.cleanup_test_environment", lambda: None)
    monkeypatch.setattr("vllm_omni.entrypoints.omni.Omni", _FakeOmni)

    OmniRunner("fake-model", stage_overrides={"0": {"gpu_memory_utilization": 0.3}})

    overrides = captured["stage_overrides"]
    assert isinstance(overrides, dict)
    assert overrides["0"]["gpu_memory_utilization"] == 0.3
    assert overrides["0"]["safetensors_load_strategy"] == "prefetch"
    assert overrides["1"]["safetensors_load_strategy"] == "lazy"
