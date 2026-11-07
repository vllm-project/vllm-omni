# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional prewarm hooks driven by ``GPUGenerationModelRunner.execute_model``.

``on_requests_added`` gets the step's prewarms right after
``on_requests_finished``; ``run_idle_prefetch`` runs on a zero-token step that
follows another zero-token step. Both are optional and must never fail the step.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

import vllm_omni.worker.gpu_generation_model_runner as gen_runner_module
from vllm_omni.core.sched.output import OmniRequestPrewarm
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PREWARMS = [OmniRequestPrewarm(request_id="req-new", payload={})]


class _StopExecutionError(Exception):
    """Raised by the stubbed ``_prepare_inputs`` (the step got past the idle branch)."""


class _NotAnExceptionError(BaseException):
    """Outside ``Exception``, like KeyboardInterrupt/SystemExit."""


class _HookModel:
    """Records each lifecycle hook call into ``events``; ``errors`` maps "added"/"idle" to what that hook raises."""

    def __init__(self, events: list[tuple], errors: dict[str, BaseException] | None = None) -> None:
        self._events = events
        self._errors = errors or {}

    def _record(self, *event) -> None:
        self._events.append(event)
        if event[0] in self._errors:
            raise self._errors[event[0]]

    def on_requests_finished(self, finished_req_ids) -> None:
        self._record("finished", set(finished_req_ids))

    def on_requests_added(self, prewarms) -> None:
        self._record("added", list(prewarms))

    def run_idle_prefetch(self) -> None:
        self._record("idle")


@pytest.fixture(autouse=True)
def _no_transfer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gen_runner_module, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(gen_runner_module, "has_kv_transfer_group", lambda: False)


@pytest.fixture
def mock_logger(mocker):
    return mocker.patch.object(gen_runner_module, "logger")


def _hook_failures(mock_logger) -> list[tuple[str, str, object]]:
    """(level, hook name, exc_info) of each optional-hook failure logged."""
    return [
        (level, args[1], kwargs.get("exc_info"))
        for level, args, kwargs in mock_logger.method_calls
        if "Optional model hook" in args[0]
    ]


def _sched(total: int, *, finished=(), prewarms=None) -> SimpleNamespace:
    """The SchedulerOutput fields execute_model reads up to ``_prepare_inputs``."""
    out = SimpleNamespace(
        total_num_scheduled_tokens=total,
        num_scheduled_tokens={"req-1": total},
        finished_req_ids=set(finished),
        kv_connector_metadata=None,
    )
    if prewarms is not None:
        out.pending_request_prewarms = prewarms
    return out


def _make_runner(model, events: list[tuple], *, prev_step_idle: bool = True) -> GPUGenerationModelRunner:
    """A runner stubbed up to ``_prepare_inputs``; ``__init__`` sets ``prev_step_idle=False``."""
    runner = object.__new__(GPUGenerationModelRunner)
    runner._prev_step_idle = prev_step_idle
    runner._failed_optional_model_hooks = set()
    runner.execute_model_state = None
    runner.routed_experts_initialized = False
    runner.speculative_config = None
    runner.model_config = SimpleNamespace(async_chunk=False)
    runner.cache_config = SimpleNamespace(kv_sharing_fast_prefill=False)
    runner.input_batch = SimpleNamespace(num_reqs=1, req_ids=["req-1"])
    runner.model = model
    runner.parallel_config = SimpleNamespace(distributed_executor_backend=None, data_parallel_size=1)
    runner._dummy_run = lambda num_tokens: events.append(("dummy_run", num_tokens))
    runner._update_states = lambda scheduler_output: None
    runner.synchronize_input_prep = contextlib.nullcontext
    runner.attach_omni_connector_output = lambda result: result

    def _prepare_inputs(scheduler_output, num_scheduled_tokens_np):
        events.append(("prepare_inputs",))
        raise _StopExecutionError

    runner._prepare_inputs = _prepare_inputs
    return runner


def _step(runner: GPUGenerationModelRunner, scheduler_output: SimpleNamespace):
    """One execute_model call; a positive-token step stops at ``_prepare_inputs``."""
    if scheduler_output.total_num_scheduled_tokens > 0:
        with pytest.raises(_StopExecutionError):
            runner.execute_model(scheduler_output)
        return None
    return runner.execute_model(scheduler_output)


@pytest.mark.parametrize(
    ("total", "last_event"),
    [
        pytest.param(0, ("idle",), id="zero"),
        pytest.param(-1, ("idle",), id="negative"),
        pytest.param(3, ("prepare_inputs",), id="positive"),
    ],
)
def test_step_runs_finished_then_added_then_idle(total, last_event) -> None:
    """Prewarms go out after on_requests_finished (an id freed and re-added in
    one step keeps the new payload) on every step; only idle steps prefetch."""
    events: list[tuple] = []
    runner = _make_runner(_HookModel(events), events)

    output = _step(runner, _sched(total, finished={"req-old"}, prewarms=_PREWARMS))

    assert events == [("finished", {"req-old"}), ("added", _PREWARMS), last_event]
    if total <= 0:
        assert output is EMPTY_MODEL_RUNNER_OUTPUT


def test_idle_prefetch_between_dp_dummy_run_and_kv_connector_no_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """The DP-sync dummy run goes first so other DP ranks never wait on the
    prefetch; the kv-connector return path still records the idle flag."""
    monkeypatch.setattr(gen_runner_module, "has_kv_transfer_group", lambda: True)
    events: list[tuple] = []
    runner = _make_runner(_HookModel(events), events, prev_step_idle=False)
    runner.parallel_config = SimpleNamespace(distributed_executor_backend="external_launcher", data_parallel_size=2)
    runner.vllm_config = object()
    sentinel = object()

    def _kv_connector_no_forward(scheduler_output, vllm_config):
        events.append(("kv_no_forward",))
        return sentinel

    runner.kv_connector_no_forward = _kv_connector_no_forward

    for _ in range(2):
        assert _step(runner, _sched(0)) is sentinel

    assert events == [("dummy_run", 1), ("kv_no_forward",), ("dummy_run", 1), ("idle",), ("kv_no_forward",)]


def test_idle_prefetch_needs_two_idle_steps_in_a_row() -> None:
    """A fresh runner and the first idle step after a busy one skip the prefetch
    (the busy step's output may still be in flight); prewarms go out every step."""
    events: list[tuple] = []
    runner = _make_runner(_HookModel(events), events, prev_step_idle=False)

    for total, expected in [
        (-1, ["added"]),  # fresh runner
        (0, ["added", "idle"]),  # a negative total also records the idle flag
        (3, ["added", "prepare_inputs"]),
        (0, ["added"]),  # first idle step after a busy one
        (-1, ["added", "idle"]),
    ]:
        events.clear()
        _step(runner, _sched(total, prewarms=_PREWARMS))
        assert [name for name, *_ in events] == expected


@pytest.mark.parametrize("prewarms", [None, []], ids=["attribute-missing", "empty-list"])
def test_on_requests_added_skipped_without_prewarms(prewarms) -> None:
    events: list[tuple] = []
    runner = _make_runner(_HookModel(events), events)

    assert _step(runner, _sched(0, prewarms=prewarms)) is EMPTY_MODEL_RUNNER_OUTPUT
    assert events == [("idle",)]


@pytest.mark.parametrize(
    "model",
    [object(), SimpleNamespace(on_requests_added=None, run_idle_prefetch="not-callable")],
    ids=["no-hooks", "non-callable-hooks"],
)
def test_models_without_callable_hooks_are_skipped(mock_logger, model) -> None:
    runner = _make_runner(model, [])

    assert _step(runner, _sched(0, finished={"req-old"}, prewarms=_PREWARMS)) is EMPTY_MODEL_RUNNER_OUTPUT
    assert _hook_failures(mock_logger) == []


def test_hook_exceptions_are_swallowed_and_warned_once_per_hook(mock_logger) -> None:
    """An escaping exception would kill the EngineCore; run_idle_prefetch runs
    on every idle step (~1 kHz), so each hook warns on its own first failure,
    then logs at debug, and is still tried on later steps."""
    events: list[tuple] = []
    errors = {"added": RuntimeError("bad payload"), "idle": RuntimeError("prefetch failed")}
    runner = _make_runner(_HookModel(events, errors), events)

    for _ in range(2):
        assert _step(runner, _sched(0, prewarms=_PREWARMS)) is EMPTY_MODEL_RUNNER_OUTPUT

    assert [name for name, *_ in events] == ["added", "idle", "added", "idle"]
    assert _hook_failures(mock_logger) == [
        ("warning", "on_requests_added", True),
        ("warning", "run_idle_prefetch", True),
        ("debug", "on_requests_added", True),
        ("debug", "run_idle_prefetch", True),
    ]


def test_hook_base_exceptions_propagate(mock_logger) -> None:
    """Only ``Exception`` is contained, so KeyboardInterrupt/SystemExit still stop the engine."""
    events: list[tuple] = []
    runner = _make_runner(_HookModel(events, {"idle": _NotAnExceptionError()}), events)

    with pytest.raises(_NotAnExceptionError):
        _step(runner, _sched(0))

    assert _hook_failures(mock_logger) == []
