# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU coverage for shared dispatch and backend ordering.

AST extraction is limited to runner wiring because importing the NPU runner
requires vllm-ascend. Dispatch behavior is exercised directly, without patching
imports; update the wiring checks if runner orchestration is refactored.
"""

import ast
from pathlib import Path
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

    # Execute the actual method in a minimal class: zero-argument super() must
    # retain its real ordering and return semantics, including deferred updates.
    cls = ast.ClassDef(
        name="Runner", bases=[ast.Name(id="Parent", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[]
    )
    namespace: dict[str, Any] = {
        "Parent": Parent,
        "SchedulerOutput": object,
        "apply_duplex_window_reanchor": reanchor,
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


@pytest.mark.parametrize("backend", ["npu", "gpu"])
def test_backend_reanchor_order(backend):
    root = Path(__file__).resolve().parents[3] / "vllm_omni"
    path = root / (
        "platforms/npu/worker/npu_ar_model_runner.py" if backend == "npu" else "worker/gpu_ar_model_runner.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == ("NPUARModelRunner" if backend == "npu" else "GPUARModelRunner")
    )
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_update_states")
    calls = [
        n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
        for stmt in method.body
        for n in ast.walk(stmt)
        if isinstance(n, ast.Call) and isinstance(n.func, (ast.Attribute, ast.Name))
    ]
    dispatch = "apply_duplex_window_reanchor" if backend == "npu" else "_maybe_apply_duplex_window_reanchor"
    assert calls.index("_update_states") < calls.index("_update_duplex_sampling_states") < calls.index(dispatch)


@pytest.mark.parametrize("use_helper", [False, True])
def test_shared_dispatch(use_helper):
    from types import SimpleNamespace

    path = Path(__file__).resolve().parents[3] / "vllm_omni/worker/duplex_reanchor.py"
    namespace: dict[str, Any] = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    events = []

    def hook(runner, scheduler_output=None):
        events.append((runner, scheduler_output))

    runner = SimpleNamespace(model=SimpleNamespace())
    if use_helper:
        runner._duplex_window_helper = SimpleNamespace(maybe_apply_reanchor=hook)
    else:
        runner.model.apply_duplex_kv_reanchor = hook
    output = object()
    namespace["apply_duplex_window_reanchor"](runner, output)
    assert events == [(runner, output)]


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("use_helper", [False, True])
def test_internal_typeerror_never_retries(legacy, use_helper):
    from types import SimpleNamespace

    path = Path(__file__).resolve().parents[3] / "vllm_omni/worker/duplex_reanchor.py"
    namespace: dict[str, Any] = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    effects = []

    def old_hook(runner):
        effects.append("rotated")
        raise TypeError("internal failure")

    def new_hook(runner, scheduler_output=None):
        effects.append("rotated")
        raise TypeError("internal failure")

    hook = old_hook if legacy else new_hook
    runner = SimpleNamespace(model=SimpleNamespace())
    if use_helper:
        runner._duplex_window_helper = SimpleNamespace(maybe_apply_reanchor=hook)
    else:
        runner.model.apply_duplex_kv_reanchor = hook
    with pytest.raises(TypeError, match="internal failure"):
        namespace["apply_duplex_window_reanchor"](runner, object())
    assert effects == ["rotated"]


@pytest.mark.parametrize("has_command,is_minicpmo", [(True, True), (False, True), (True, False)])
def test_shared_dispatch_fallback(monkeypatch, has_command, is_minicpmo):
    import sys
    from types import ModuleType, SimpleNamespace

    path = Path(__file__).resolve().parents[3] / "vllm_omni/worker/duplex_reanchor.py"
    namespace: dict[str, Any] = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    calls = []
    module = ModuleType("vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_kv")
    setattr(
        module,
        "MiniCPMO45DuplexWorkerHelper",
        SimpleNamespace(maybe_apply_reanchor=lambda runner, scheduler_output: calls.append((runner, scheduler_output))),
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    model = type("Model", (), {"__module__": "minicpmo_4_5" if is_minicpmo else "other"})()
    runner = SimpleNamespace(
        model=model,
        input_batch=SimpleNamespace(num_reqs=1, req_ids=["r"]),
        model_intermediate_buffer={"r": {"duplex": {"stage0_reanchor": {}} if has_command else {}}},
    )
    output = object()
    namespace["apply_duplex_window_reanchor"](runner, output)
    assert calls == ([(runner, output)] if has_command and is_minicpmo else [])


@pytest.mark.parametrize("backend", ["npu", "gpu"])
def test_cache_validation_follows_allocation(backend):
    root = Path(__file__).resolve().parents[3] / "vllm_omni"
    path = root / (
        "platforms/npu/worker/npu_ar_model_runner.py" if backend == "npu" else "worker/gpu_ar_model_runner.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "initialize_kv_cache")
    calls = sorted((node.lineno, ast.unparse(node.func)) for node in ast.walk(method) if isinstance(node, ast.Call))
    names = [name for _, name in calls]
    assert names.index("super().initialize_kv_cache") < names.index("validate_duplex_kv_cache")


@pytest.mark.parametrize("unhashable", [False, True])
def test_hook_dispatch_does_not_retain_callable(unhashable):
    import gc
    import weakref

    path = Path(__file__).resolve().parents[3] / "vllm_omni/worker/duplex_reanchor.py"
    namespace: dict[str, Any] = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    calls = []

    class Hook:
        def __call__(self, runner, scheduler_output=None):
            calls.append(scheduler_output)

    class UnhashableHook(Hook):
        def __eq__(self, other: object) -> bool:
            return self is other

    owner = UnhashableHook() if unhashable else Hook()
    reference = weakref.ref(owner)
    hook = owner if unhashable else owner.__call__
    namespace["_call_reanchor"](hook, object(), "output")
    assert calls == ["output"]
    del hook, owner
    gc.collect()
    assert reference() is None
