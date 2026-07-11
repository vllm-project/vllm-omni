# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import contextlib
import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _od_config(**overrides) -> OmniDiffusionConfig:
    kwargs = dict(model_class_name="QwenImagePipeline")
    kwargs.update(overrides)
    return OmniDiffusionConfig(**kwargs)


@contextlib.contextmanager
def _stub_process_funcs():
    with (
        patch("vllm_omni.diffusion.diffusion_engine.get_diffusion_post_process_func", return_value=None),
        patch("vllm_omni.diffusion.diffusion_engine.get_diffusion_action_post_process_func", return_value=None),
        patch("vllm_omni.diffusion.diffusion_engine.get_diffusion_pre_process_func", return_value=None),
    ):
        yield


def test_flag_off_builds_legacy_path():
    od_config = _od_config(enable_runtime_v2=False)

    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    with (
        _stub_process_funcs(),
        patch("vllm_omni.diffusion.executor.abstract.DiffusionExecutor.get_class") as get_class,
        patch.object(DiffusionEngine, "_dummy_run") as dummy_run,
    ):
        get_class.return_value = MagicMock()

        engine = DiffusionEngine(od_config)
        try:
            assert engine.enable_runtime_v2 is False
            assert engine.runtime_v2_runner is None
            assert engine.scheduler is not None
            assert engine.executor is not None
            assert engine.execute_fn is not None
            dummy_run.assert_called_once()
        finally:
            engine.close()


def test_flag_on_builds_runtime_v2_path_and_skips_warmup():
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    od_config = _od_config(enable_runtime_v2=True)

    with (
        _stub_process_funcs(),
        patch("vllm_omni.diffusion.runtime_v2.multiproc_worker.MultiprocWorkerPool.start") as pool_start,
        patch.object(DiffusionEngine, "_dummy_run") as dummy_run,
        patch("vllm_omni.diffusion.executor.abstract.DiffusionExecutor.get_class") as get_class,
    ):
        get_class.return_value = MagicMock()

        engine = DiffusionEngine(od_config)
        try:
            assert engine.enable_runtime_v2 is True
            assert engine.runtime_v2_runner is not None
            from vllm_omni.diffusion.runtime_v2.runner import RuntimeV2Runner

            assert isinstance(engine.runtime_v2_runner, RuntimeV2Runner)
            assert engine.scheduler is None
            assert engine.executor is None
            assert engine.execute_fn is None
            dummy_run.assert_not_called()
            pool_start.assert_called_once()
        finally:
            engine.close()


def test_abort_is_handed_to_the_runtime_v2_owner_thread():
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    engine = object.__new__(DiffusionEngine)
    engine.enable_runtime_v2 = True
    engine._closed = False
    engine._cv = threading.Condition()
    engine._runtime_v2_inflight = {"r"}
    engine._runtime_v2_commands = queue.Queue()

    engine.abort("r")

    assert engine._runtime_v2_commands.get_nowait() == ("abort", "r")


def _bare_runtime_engine(runner):
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    engine = object.__new__(DiffusionEngine)
    engine.enable_runtime_v2 = True
    engine.runtime_v2_runner = runner
    engine.stop_event = threading.Event()
    engine._cv = threading.Condition()
    engine._runtime_v2_inflight = {"r"}
    engine._runtime_v2_commands = queue.Queue()
    engine._out_queue = {}
    loop = asyncio.new_event_loop()
    future = loop.create_future()
    engine._out_queue["r"] = future
    return engine, future, loop


def test_busy_loop_delivers_finished_request_and_releases_state():
    request = SimpleNamespace(request_id="r")
    output = DiffusionOutput(output={"image": "done"})
    runner = SimpleNamespace(
        submit=MagicMock(),
        poll_once=MagicMock(),
        get_request_status=MagicMock(return_value=("finished", output)),
        release_request=MagicMock(),
    )
    engine, future, loop = _bare_runtime_engine(runner)
    runner.release_request.side_effect = lambda _request_id: engine.stop_event.set()
    engine._runtime_v2_commands.put(("submit", request))

    try:
        engine._runtime_v2_busy_loop()
        assert future.result() is output
        runner.submit.assert_called_once_with(request)
        runner.release_request.assert_called_once_with("r")
    finally:
        loop.close()


def test_busy_loop_preserves_submit_then_abort_order():
    calls = []
    runner = SimpleNamespace(
        submit=MagicMock(side_effect=lambda _request: calls.append("submit")),
        abort_request=MagicMock(side_effect=lambda _request_id: calls.append("abort")),
    )
    engine, future, loop = _bare_runtime_engine(runner)
    runner.abort_request.side_effect = lambda _request_id: (calls.append("abort"), engine.stop_event.set())
    engine._runtime_v2_commands.put(("submit", SimpleNamespace(request_id="r")))
    engine._runtime_v2_commands.put(("abort", "r"))

    try:
        engine._runtime_v2_busy_loop()
        assert calls == ["submit", "abort"]
        assert future.result().aborted is True
    finally:
        loop.close()


def test_busy_loop_latches_poll_failure_and_aborts_request():
    error = RuntimeError("control plane failed")
    runner = SimpleNamespace(
        submit=MagicMock(),
        poll_once=MagicMock(side_effect=error),
        abort_request=MagicMock(),
    )
    engine, future, loop = _bare_runtime_engine(runner)
    engine._runtime_v2_commands.put(("submit", SimpleNamespace(request_id="r")))

    try:
        engine._runtime_v2_busy_loop()
        assert future.result().error
        assert engine._runtime_v2_fatal_error is error
        runner.abort_request.assert_called_once_with("r")
        assert engine.is_backend_dead()
    finally:
        loop.close()


@pytest.mark.parametrize("error,expected", [(None, False), (RuntimeError("dead"), True)])
def test_runtime_v2_backend_health(error, expected):
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    engine = object.__new__(DiffusionEngine)
    engine.enable_runtime_v2 = True
    runner = SimpleNamespace(check_health=MagicMock(side_effect=error))
    engine.runtime_v2_runner = runner

    assert engine.is_backend_dead() is expected
