# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise NPU state-update orchestration without importing vllm-ascend."""

import ast
import builtins
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_reanchor_runs_after_state_install_and_propagates_failure():
    root = Path(__file__).resolve().parents[3]
    path = root / "vllm_omni/platforms/npu/worker/npu_ar_model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    runner_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUARModelRunner")
    method = next(n for n in runner_class.body if isinstance(n, ast.FunctionDef) and n.name == "_update_states")
    events = []
    deferred = object()
    output = object()

    class Parent:
        def _update_states(self, scheduler_output):
            assert scheduler_output is output
            events.append("install_compacted_state")
            return deferred

    def reanchor(runner, scheduler_output):
        assert scheduler_output is output
        assert events == ["install_compacted_state", "sampling"]
        events.append("reanchor")
        if runner.fail:
            raise RuntimeError("rotation failed")

    def import_hook(name, *args, **kwargs):
        if name == "vllm_omni.worker.gpu_ar_model_runner":
            return SimpleNamespace(GPUARModelRunner=SimpleNamespace(_maybe_apply_duplex_window_reanchor=reanchor))
        return builtins.__import__(name, *args, **kwargs)

    # Execute the actual method in a minimal class: zero-argument super() must
    # retain its real ordering and return semantics, including deferred updates.
    cls = ast.ClassDef(
        name="Runner", bases=[ast.Name(id="Parent", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[]
    )
    namespace: dict[str, Any] = {
        "Parent": Parent,
        "SchedulerOutput": object,
        "__builtins__": dict(vars(builtins), __import__=import_hook),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), namespace)
    runner = namespace["Runner"]()
    runner._update_duplex_sampling_states = lambda _: events.append("sampling")
    runner.fail = False
    assert runner._update_states(output) is deferred
    assert events == ["install_compacted_state", "sampling", "reanchor"]
    events.clear()
    runner.fail = True
    with pytest.raises(RuntimeError, match="rotation failed"):
        runner._update_states(output)
