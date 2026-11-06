# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lock the Entrypoint H100 turn-server grouping without booting a model.

pytest 9.1.1 (the repo pin) reuses a class-scoped fixture for equal param
values only while those tests stay adjacent. A different value in between
finalizes the server, and coming back starts another one.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SOURCE = Path(__file__).with_name("test_qwen3_omni_realtime_websocket.py")
_ASYNC_PARAMS = "realtime_async_chunk_server_params"
_VAD_PARAMS = "realtime_server_vad_server_params"
_SYNC_PARAMS = "realtime_sync_server_params"


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _parametrize_arg(func: ast.FunctionDef, argname: str) -> ast.AST | None:
    for decorator in func.decorator_list:
        if not isinstance(decorator, ast.Call) or _call_name(decorator.func) != "parametrize":
            continue
        if not decorator.args or not isinstance(decorator.args[0], ast.Constant):
            continue
        if decorator.args[0].value != argname or len(decorator.args) < 2:
            continue
        return decorator.args[1]
    return None


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    found = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    assert len(found) == 1, name
    return found[0]


def _fixture_scope(func: ast.FunctionDef) -> str | None:
    for decorator in func.decorator_list:
        if not isinstance(decorator, ast.Call) or _call_name(decorator.func) != "fixture":
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "scope" and isinstance(keyword.value, ast.Constant):
                return keyword.value.value
    return None


def test_turn_servers_stay_in_equal_param_groups() -> None:
    """async_chunk and server VAD share one value; the sync server is a single tail."""
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    turn = _class(tree, "TestQwen3OmniRealtimeWebSocket")
    groups: list[str] = []
    for node in turn.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        param = _parametrize_arg(node, "omni_server")
        assert isinstance(param, ast.Name), node.name
        groups.append(param.id)

    assert groups == [
        _ASYNC_PARAMS,
        _ASYNC_PARAMS,
        _ASYNC_PARAMS,
        _ASYNC_PARAMS,
        _ASYNC_PARAMS,
        _ASYNC_PARAMS,
        _VAD_PARAMS,
        _SYNC_PARAMS,
    ], groups

    names = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
    }
    for list_name in (_ASYNC_PARAMS, _VAD_PARAMS):
        value = names[list_name]
        assert isinstance(value, ast.List) and len(value.elts) == 1
        param = value.elts[0]
        assert isinstance(param, ast.Call)
        assert isinstance(param.args[0], ast.Name)
        assert param.args[0].id == "_async_chunk_server"

    server = names["_async_chunk_server"]
    assert isinstance(server, ast.Call)
    keywords = {keyword.arg: keyword.value for keyword in server.keywords}
    assert isinstance(keywords["stage_config_path"], ast.Name)
    assert keywords["stage_config_path"].id == "async_chunk_stage_config"
    assert isinstance(keywords["server_args"], ast.List)
    server_args = [elt.value for elt in keywords["server_args"].elts]
    assert server_args == ["--tool-call-parser", "hermes", "--enable-auto-tool-choice"]
    assert "--async-chunk" not in server_args

    sync_param = names[_SYNC_PARAMS].elts[0]
    assert isinstance(sync_param, ast.Call)
    sync_server = sync_param.args[0]
    assert isinstance(sync_server, ast.Call)
    sync_keywords = {keyword.arg: keyword.value for keyword in sync_server.keywords}
    assert isinstance(sync_keywords["stage_config_path"], ast.Name)
    assert sync_keywords["stage_config_path"].id == "default_stage_config"


def test_jit_replay_keeps_a_function_scoped_server() -> None:
    """The warmup check needs a fresh process; sharing it would hide runtime JIT."""
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    cold = _class(tree, "TestQwen3OmniRealtimeColdStart")
    fixtures = [node for node in cold.body if isinstance(node, ast.FunctionDef) and node.name == "cold_omni_server"]
    assert len(fixtures) == 1
    assert _fixture_scope(fixtures[0]) == "function"
    tests = [node for node in cold.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")]
    assert len(tests) == 1
    param = _parametrize_arg(tests[0], "cold_omni_server")
    assert isinstance(param, ast.Name)
    assert param.id == _ASYNC_PARAMS


def test_duplex_playback_keeps_one_class_scoped_server() -> None:
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    fixtures = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "qwen_duplex_server"]
    assert len(fixtures) == 1
    assert _fixture_scope(fixtures[0]) == "class"
    duplex = _class(tree, "TestQwen3OmniDuplexPlayback")
    test_names = [
        node.name for node in duplex.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    assert "test_speech_interrupts_playback" in test_names
    assert "test_semantic_vad_speech_interrupts_playback" in test_names
    assert "test_semantic_vad_multi_turn_without_client_commit" in test_names
