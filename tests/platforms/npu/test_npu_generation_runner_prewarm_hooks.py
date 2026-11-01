# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AST checks on where both generation runners call the async-chunk prewarm hooks.

The NPU runner imports ``vllm_ascend`` at module level, so it cannot be imported
on CPU; the GPU runner is checked the same way to keep the two in lock-step (its
behavior is covered by ``tests/worker/test_generation_runner_prewarm_hooks.py``).
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SRC = Path(__file__).resolve().parents[3] / "vllm_omni"
_RUNNERS = [
    pytest.param((_SRC / "platforms/npu/worker/npu_generation_model_runner.py", "NPUGenerationModelRunner"), id="npu"),
    pytest.param((_SRC / "worker/gpu_generation_model_runner.py", "GPUGenerationModelRunner"), id="gpu"),
]
_HOOK = "_call_optional_model_hook"


def _method(runner: tuple[Path, str], name: str) -> ast.FunctionDef:
    path, class_name = runner
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef) and stmt.name == name:
                    return stmt
    raise AssertionError(f"{class_name}.{name} not found in {path}")


def _call(attr: str, first_arg: str | None = None) -> Callable[[ast.AST], bool]:
    def match(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == attr
            and (first_arg is None or (bool(node.args) and getattr(node.args[0], "value", None) == first_arg))
        )

    return match


def _is_idle_branch(node: ast.AST) -> bool:
    return isinstance(node, ast.If) and ast.unparse(node.test) == "num_scheduled_tokens <= 0"


def _index_of(stmts: list[ast.stmt], pred: Callable[[ast.AST], bool]) -> int | None:
    return next((i for i, stmt in enumerate(stmts) if any(pred(n) for n in ast.walk(stmt))), None)


@pytest.mark.parametrize("runner", _RUNNERS)
def test_on_requests_added_follows_on_requests_finished(runner) -> None:
    execute_model = _method(runner, "execute_model")
    is_added = _call(_HOOK, "on_requests_added")
    assert sum(is_added(n) for n in ast.walk(execute_model)) == 1

    # Sibling statements: finished, then added, then the zero-token early return,
    # so an idle step (placeholder waiting for chunk 0) still gets the payload.
    sites = []
    for node in ast.walk(execute_model):
        for stmts in (getattr(node, field, None) for field in ("body", "orelse", "finalbody")):
            if isinstance(stmts, list) and stmts and isinstance(stmts[0], ast.stmt):
                finished, added = _index_of(stmts, _call("on_requests_finished")), _index_of(stmts, is_added)
                if finished is not None and added is not None and finished != added:
                    sites.append((stmts, finished, added))
    assert len(sites) == 1
    stmts, finished, added = sites[0]
    zero_token = _index_of(stmts, _is_idle_branch)
    assert zero_token is not None and finished < added < zero_token
    assert "pending_request_prewarms" in ast.unparse(stmts[added])


@pytest.mark.parametrize("runner", _RUNNERS)
def test_run_idle_prefetch_only_in_zero_token_branch_before_return(runner) -> None:
    execute_model = _method(runner, "execute_model")
    is_idle = _call(_HOOK, "run_idle_prefetch")
    assert sum(is_idle(n) for n in ast.walk(execute_model)) == 1
    branches = [n for n in ast.walk(execute_model) if _is_idle_branch(n)]
    assert len(branches) == 1
    body = branches[0].body

    # Exactly `if prev_step_idle: <hook>`, after the DP _dummy_run, before any return.
    idle = _index_of(body, is_idle)
    assert idle is not None
    guard = body[idle]
    assert isinstance(guard, ast.If) and ast.unparse(guard.test) == "prev_step_idle" and not guard.orelse
    assert len(guard.body) == 1 and isinstance(guard.body[0], ast.Expr) and is_idle(guard.body[0].value)
    dummy_run = _index_of(body, _call("_dummy_run"))
    first_return = _index_of(body, lambda n: isinstance(n, ast.Return))
    assert dummy_run is not None and first_return is not None
    assert dummy_run < idle < first_return


@pytest.mark.parametrize("runner", _RUNNERS)
def test_prev_step_idle_recorded_before_any_return(runner) -> None:
    execute_model = _method(runner, "execute_model")
    top_level = [ast.unparse(stmt) for stmt in execute_model.body]
    # Top-level, so no path skips them: read the old value, then record this step.
    count = top_level.index("num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens")
    read = top_level.index("prev_step_idle = self._prev_step_idle")
    write = top_level.index("self._prev_step_idle = num_scheduled_tokens <= 0")
    assert count < read < write
    first_return = min(n.lineno for n in ast.walk(execute_model) if isinstance(n, ast.Return))
    assert execute_model.body[write].lineno < first_return
    # ...and nowhere else.
    targets = [ast.unparse(t) for n in ast.walk(execute_model) if isinstance(n, ast.Assign) for t in n.targets]
    assert targets.count("self._prev_step_idle") == 1

    init = {ast.unparse(n) for n in ast.walk(_method(runner, "__init__")) if isinstance(n, (ast.Assign, ast.AnnAssign))}
    assert "self._prev_step_idle = False" in init
    assert "self._failed_optional_model_hooks: set[str] = set()" in init


@pytest.mark.parametrize("runner", _RUNNERS)
def test_hooks_run_under_inference_mode_and_catch_only_exception(runner) -> None:
    decorators = [ast.unparse(d) for d in _method(runner, "execute_model").decorator_list]
    assert "torch.inference_mode()" in decorators
    handlers = [n for n in ast.walk(_method(runner, _HOOK)) if isinstance(n, ast.ExceptHandler)]
    assert [h.type and ast.unparse(h.type) for h in handlers] == ["Exception"]
