# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-only ownership tests; do not import the GPU/API server package initializer."""

import __future__

import ast
import asyncio
import copy
import importlib.util
import sys
import threading
import time
from http import HTTPStatus
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

from vllm_omni.engine.messages import ErrorMessage
from vllm_omni.errors import OmniClientError, is_client_error_status, raise_client_error_or
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

_PATH = Path(__file__).resolve().parents[3] / "vllm_omni/entrypoints/openai/video/generation/guided_lifetime.py"
_SPEC = importlib.util.spec_from_file_location("_test_video_guided_lifetime", _PATH)
lifetime = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = lifetime
_SPEC.loader.exec_module(lifetime)

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cpu,
    pytest.mark.skipif(sys.version_info < (3, 11), reason="Guided ownership requires Python 3.11+"),
]


def test_python310_rejects_guides_before_bundle_allocation(monkeypatch):
    owner = lifetime.GuidedRequestLifetime()
    construct = Mock(side_effect=AssertionError("Unsupported runtime must not allocate a bundle"))
    monkeypatch.setattr(lifetime, "version_info", (3, 10, 14))
    monkeypatch.setattr(lifetime, "GuidedRequestBundle", construct)
    with pytest.raises(HTTPException) as error:
        owner.reserve(4)
    assert error.value.status_code == 503
    assert "Python 3.11" in error.value.detail and "omit timeline_guides" in error.value.detail
    construct.assert_not_called()
    assert not owner.bundles


def test_python311_admits_guides(monkeypatch):
    monkeypatch.setattr(lifetime, "version_info", (3, 11, 0))
    owner = lifetime.GuidedRequestLifetime()
    bundle = owner.reserve(4)
    assert bundle in owner.bundles
    bundle.close()


def test_reservation_rejects_before_persistence_and_releases(tmp_path):
    owner = lifetime.GuidedRequestLifetime()
    bundle = owner.reserve(1)
    with pytest.raises(HTTPException) as exc:
        owner.reserve(1)
    assert exc.value.status_code == 503
    path = tmp_path / "guide.png"
    path.write_bytes(b"guide")
    bundle.paths.update([str(path), str(path)])
    bundle.close()
    bundle.close()
    assert not path.exists()
    assert not owner.bundles
    owner.reserve(1).close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_outer_cancellation_retains_all_inputs_and_capacity(tmp_path, failure):
    owner = lifetime.GuidedRequestLifetime()
    bundle = owner.reserve(1)
    paths = [tmp_path / name for name in ("guide", "video-reference", "audio-reference")]
    for path in paths:
        path.write_bytes(b"input")
        bundle.paths.add(str(path))
    entered, finish = asyncio.Event(), asyncio.Event()

    async def work():
        entered.set()
        await finish.wait()
        assert all(path.exists() for path in paths)
        if failure:
            raise ValueError("model error")
        return b"video"

    task = bundle.submit(work())

    async def response():
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            bundle.abandoned = True
            raise

    outer = asyncio.create_task(response())
    await entered.wait()
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer
    assert not task.done()
    assert all(path.exists() for path in paths)
    with pytest.raises(HTTPException):
        owner.reserve(1)
    finish.set()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    if not failure:
        assert task.result() is None
    assert not any(path.exists() for path in paths)
    assert not owner.bundles


@pytest.mark.asyncio
async def test_abandoned_before_start_skips_generation(tmp_path):
    owner = lifetime.GuidedRequestLifetime()
    bundle = owner.reserve(1)
    path = tmp_path / "guide"
    path.write_bytes(b"input")
    bundle.paths.add(str(path))
    called = False

    async def work():
        nonlocal called
        called = True

    task = bundle.submit(work())
    bundle.abandoned = True
    await task
    assert not called
    assert not path.exists()
    assert not owner.bundles


@pytest.mark.asyncio
async def test_cancelled_before_coroutine_start_releases(tmp_path):
    owner = lifetime.GuidedRequestLifetime()
    bundle = owner.reserve(1)
    path = tmp_path / "guide"
    path.write_bytes(b"input")
    bundle.paths.add(str(path))

    async def work():
        pytest.fail("Unstarted work must not run")

    task = bundle.submit(work())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert not path.exists()
    assert not owner.bundles


@pytest.mark.asyncio
async def test_submit_failure_closes_inputs_and_coroutine(tmp_path, monkeypatch):
    owner = lifetime.GuidedRequestLifetime()
    bundle = owner.reserve(1)
    path = tmp_path / "guide"
    path.write_bytes(b"input")
    bundle.paths.add(str(path))

    async def work():
        pytest.fail("Unsubmitted work must not run")

    def fail(coro):
        raise RuntimeError("submission failed")

    monkeypatch.setattr(asyncio, "create_task", fail)
    coro = work()
    with pytest.raises(RuntimeError, match="submission failed"):
        bundle.submit(coro)
    assert coro.cr_frame is None
    assert not path.exists()
    assert not owner.bundles


@pytest.mark.asyncio
async def test_shutdown_waits_for_true_completion_and_stops_admission(tmp_path):
    owner = lifetime.GuidedRequestLifetime()
    bundle = owner.reserve(1)
    path = tmp_path / "guide"
    path.write_bytes(b"input")
    bundle.paths.add(str(path))
    entered, finish = asyncio.Event(), asyncio.Event()

    async def work():
        entered.set()
        await finish.wait()
        assert path.exists()

    task = bundle.submit(work())
    await entered.wait()
    drain = asyncio.create_task(owner.drain())
    await asyncio.sleep(0)
    assert not drain.done() and not task.done()
    assert bundle.abandoned
    with pytest.raises(HTTPException):
        owner.reserve(2)
    finish.set()
    await drain
    assert not path.exists()
    assert not owner.bundles


@pytest.mark.asyncio
async def test_shutdown_cancellation_does_not_abort_inner_task(tmp_path):
    owner = lifetime.GuidedRequestLifetime()
    bundle = owner.reserve(1)
    path = tmp_path / "guide"
    path.write_bytes(b"input")
    bundle.paths.add(str(path))
    entered, finish = asyncio.Event(), asyncio.Event()

    async def work():
        entered.set()
        await finish.wait()

    task = bundle.submit(work())
    await entered.wait()
    drain = asyncio.create_task(owner.drain())
    await asyncio.sleep(0)
    drain.cancel()
    await asyncio.sleep(0)
    assert not drain.done()
    drain.cancel()
    await asyncio.sleep(0)
    assert not drain.done()
    assert not task.done() and path.exists()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await drain
    assert not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_nested_future", [False, True])
async def test_unexpected_inner_cancellation_does_not_claim_worker_quiescence(tmp_path, cancel_nested_future):
    owner = lifetime.GuidedRequestLifetime()
    bundle = owner.reserve(1)
    path = tmp_path / "guide"
    path.write_bytes(b"input")
    bundle.paths.add(str(path))
    entered = asyncio.Event()
    nested = asyncio.get_running_loop().create_future()

    async def work():
        entered.set()
        await nested

    task = bundle.submit(work())
    await entered.wait()
    if cancel_nested_future:
        nested.cancel()
    else:
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert path.exists()
    assert bundle in owner.bundles
    await owner.drain()
    assert path.exists()
    assert bundle in owner.bundles
    # A test double has no external worker; the test can establish quiescence.
    bundle.close()


@pytest.mark.asyncio
async def test_swallowed_inner_cancellation_still_retains_inputs(tmp_path):
    owner = lifetime.GuidedRequestLifetime()
    bundle = owner.reserve(1)
    path = tmp_path / "guide"
    path.write_bytes(b"input")
    bundle.paths.add(str(path))
    entered = asyncio.Event()

    async def work():
        entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            return b"not-a-quiescence-acknowledgment"

    task = bundle.submit(work())
    await entered.wait()
    task.cancel()
    assert await task is None
    await owner.drain()
    assert bundle.abandoned
    assert path.exists()
    assert bundle in owner.bundles
    bundle.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [EngineDeadError, EngineGenerateError, RuntimeError, HTTPException])
@pytest.mark.parametrize("swallowed", [False, True])
async def test_dispatched_engine_failure_retains_even_when_job_catches_error(tmp_path, error_type, swallowed):
    owner = lifetime.GuidedRequestLifetime()
    bundle = owner.reserve(1)
    path = tmp_path / "guide"
    path.write_bytes(b"input")
    bundle.paths.add(str(path))
    bundle.job_id = f"failed-{error_type.__name__}-{swallowed}"
    lifetime.GUIDED_JOBS[bundle.job_id] = bundle

    async def work():
        bundle.engine_started = True
        try:
            if error_type is HTTPException:
                raise HTTPException(400, "worker-side validation failure")
            raise error_type("worker failed without a quiescence acknowledgment")
        except Exception:
            if not swallowed:
                raise

    task = bundle.submit(work())
    await asyncio.gather(task, return_exceptions=True)
    assert path.exists() and bundle in owner.bundles
    assert lifetime.GUIDED_JOBS[bundle.job_id] is bundle
    with pytest.raises(HTTPException):
        owner.reserve(1)
    await owner.drain()
    assert path.exists() and bundle in owner.bundles
    bundle.close()  # Only this test can independently confirm it has no external workers.


def _load_isolated_functions(path, names, namespace):
    """Execute actual definitions without the API package's incompatible imports.

    Dependencies are explicit test doubles; this does not test FastAPI routing or
    serialization. Keeping the source bodies intact exercises the real wrapper
    and engine-boundary state transitions rather than reimplementing them here.
    """
    tree = ast.parse(path.read_text())
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name in names
    ]
    assert {node.name for node in nodes} == set(names)
    for node in nodes:
        node.decorator_list = []
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)


@pytest.fixture
def terminal_error_transport():
    namespace = {
        "ErrorMessage": ErrorMessage,
        "OmniClientError": OmniClientError,
        "is_client_error_status": is_client_error_status,
        "raise_client_error_or": raise_client_error_or,
    }
    api_root = _PATH.parents[2]
    _load_isolated_functions(api_root.parent / "omni_base.py", {"_raise_nonfatal_error_message"}, namespace)
    _load_isolated_functions(api_root.parents[1] / "engine/orchestrator.py", {"_handle_stage_error"}, namespace)
    return namespace


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["terminal", "unfinished", "nonfinal", "streaming", "companion", "server_error", "unknown"]
)
async def test_terminal_error_provenance_does_not_depend_on_abort_ack(terminal_error_transport, case):
    entered_cleanup, finish_cleanup = asyncio.Event(), asyncio.Event()

    async def cleanup(*args, **kwargs):
        entered_cleanup.set()
        await finish_cleanup.wait()

    orchestrator = SimpleNamespace(
        request_states={}
        if case == "unknown"
        else {
            "job": SimpleNamespace(
                final_stage_id=2 if case == "nonfinal" else 1, streaming=SimpleNamespace(enabled=case == "streaming")
            ),
        },
        _cfg_tracker=SimpleNamespace(
            is_companion=lambda key: False,
            has_companions=lambda key: case == "companion",
            cleanup_parent=lambda key: [],
        ),
        output_async_queue=asyncio.Queue(),
        _cleanup_request_ids=cleanup,
    )
    output = SimpleNamespace(
        request_id="job",
        error="invalid guide start",
        error_status_code=500 if case == "server_error" else 400,
        error_type="BadRequestError",
        finished=case != "unfinished",
    )
    task = asyncio.create_task(terminal_error_transport["_handle_stage_error"](orchestrator, 1, output))
    try:
        await asyncio.wait_for(entered_cleanup.wait(), 2)
        message = orchestrator.output_async_queue.get_nowait()
        assert not task.done()
        assert message.worker_finished is (case == "terminal")
        error_type = RuntimeError if case == "server_error" else OmniClientError
        with pytest.raises(error_type) as caught:
            terminal_error_transport["_raise_nonfatal_error_message"](None, message)
        if error_type is OmniClientError:
            assert caught.value.worker_finished is (case == "terminal")
        finish_cleanup.set()
        await task
        # An eventual abort acknowledgment must not upgrade an unknown origin.
        assert message.worker_finished is (case == "terminal")
    finally:
        finish_cleanup.set()
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped_job", [False, True])
@pytest.mark.parametrize(
    "error_type,terminal_origin",
    [
        (EngineDeadError, False),
        (EngineGenerateError, False),
        (RuntimeError, False),
        (OmniClientError, True),
        (OmniClientError, False),
        (None, False),
    ],
)
async def test_actual_serving_methods_release_terminal_errors_and_retain_engine_failures(
    tmp_path,
    wrapped_job,
    error_type,
    terminal_origin,
    terminal_error_transport,
):
    owner = lifetime.GuidedRequestLifetime()
    bundle = owner.reserve(4)
    path = tmp_path / "guide"
    path.write_bytes(b"input")
    bundle.paths.add(str(path))
    bundle.descriptors = [{"frame_index": 100000, "image": str(path)}]
    request = SimpleNamespace(
        _guide_bundle=bundle,
        prompt="test",
        negative_prompt=None,
        timeline_guides=[{}],
        extra_params=None,
        model_fields_set=set(),
        video_params=None,
        seconds=None,
        aspect_ratio=None,
        short_edge=None,
        start_time_seconds=None,
        boundary_ratio=None,
        lora=None,
        num_outputs_per_prompt=1,
        resolve_video_params=lambda: SimpleNamespace(width=None, height=None, num_frames=None, fps=24),
    )
    job_id = "engine-failure"
    job = SimpleNamespace(file_extension="mp4", status="queued")
    records = {job_id: job}
    error_message = (
        "timeline guide frame_index 100000 is outside the output"
        if error_type is OmniClientError
        else "worker completion is unknown"
    )
    error = error_type(error_message) if error_type is not None else None

    async def update_fields(key, updates):
        record = records.get(key)
        if record is not None:
            record.__dict__.update(updates)
        return record

    store = SimpleNamespace(
        get=AsyncMock(side_effect=records.get),
        update_fields=update_fields,
        pop=AsyncMock(side_effect=lambda key: records.pop(key, None)),
    )
    storage = SimpleNamespace(delete=AsyncMock(return_value=True), save=AsyncMock())
    status_code = error.status_code if isinstance(error, OmniClientError) else 500
    convert_error = Mock(return_value=SimpleNamespace(code=status_code, message=str(error)))
    terminate = Mock()
    namespace = {
        "asyncio": asyncio,
        "copy": copy,
        "time": time,
        "cast": cast,
        "HTTPException": HTTPException,
        "HTTPStatus": HTTPStatus,
        "EngineDeadError": EngineDeadError,
        "EngineGenerateError": EngineGenerateError,
        "OmniClientError": OmniClientError,
        "OmniTextPrompt": dict,
        "AsyncOmni": object,
        "VIDEO_STORE": store,
        "STORAGE_MANAGER": storage,
        "VideoGenerationStatus": SimpleNamespace(IN_PROGRESS="in_progress", FAILED="failed", COMPLETED="completed"),
        "logger": lifetime.logger,
        "_video_error_from_exception": convert_error,
        "terminate_if_errored": terminate,
        "is_video_generation_pipeline": lambda stages: True,
        "get_default_sampling_params_list": lambda engine: [],
        "build_stage_sampling_params_list": lambda *args, **kwargs: [kwargs["diffusion_params"]],
    }
    namespace.update(terminal_error_transport)
    api_root = _PATH.parents[2]
    serving_methods = {"_run_generation", "_run_and_extract", "generate_video_bytes"}
    _load_isolated_functions(api_root / "serving_video.py", serving_methods, namespace)
    _load_isolated_functions(
        _PATH.with_name("helpers.py"),
        {"_run_video_generation_job", "_run_guided_video_generation_job", "_cleanup_video", "_delete_guided_video"},
        namespace,
    )

    async def generate(**kwargs):
        nonlocal error
        assert kwargs["sampling_params_list"][0].extra_args["_minimax_h3_timeline_guides"][0]["frame_index"] == 100000
        # Model a request EngineCore actually accepted; every failure below is
        # a post-submission engine failure.
        kwargs["on_engine_admitted"]()
        if terminal_origin:
            request_id = kwargs["request_id"]
            orchestrator = SimpleNamespace(
                request_states={
                    request_id: SimpleNamespace(final_stage_id=1, streaming=SimpleNamespace(enabled=False))
                },
                _cfg_tracker=SimpleNamespace(
                    is_companion=lambda key: False, has_companions=lambda key: False, cleanup_parent=lambda key: []
                ),
                output_async_queue=asyncio.Queue(),
                _cleanup_request_ids=AsyncMock(),
            )
            output = SimpleNamespace(
                request_id=request_id,
                finished=True,
                error=str(error),
                error_status_code=400,
                error_type="BadRequestError",
            )
            await namespace["_handle_stage_error"](orchestrator, 1, output)
            message = orchestrator.output_async_queue.get_nowait()
            try:
                namespace["_raise_nonfatal_error_message"](None, message)
            except OmniClientError as received:
                error = received
        if error is not None:
            raise error
        return
        yield b"unreachable output"

    handler = SimpleNamespace(
        _stage_configs=["diffusion"],
        _engine_client=SimpleNamespace(generate=generate),
        _resolve_default_sampling_params=OmniDiffusionSamplingParams,
        _request_fps_provided=lambda request: False,
        _request_num_frames_provided=lambda request: False,
        _resolve_video_generation_defaults=lambda request: None,
        _apply_lora=lambda *args: None,
    )
    for method in serving_methods:
        setattr(handler, method, MethodType(namespace[method], handler))
    app_state = SimpleNamespace(server=object(), engine_client=object())
    if wrapped_job:
        work = namespace["_run_video_generation_job"](handler, request, job_id, app_state=app_state)
    else:
        work = handler.generate_video_bytes(request, job_id)
    task = bundle.submit(work)
    await asyncio.gather(task, return_exceptions=True)

    assert bundle.engine_started
    terminal = error_type is None or terminal_origin
    assert bundle.engine_completed is terminal
    if wrapped_job:
        assert task.exception() is None
        assert records[job_id].status == "failed"
        assert records[job_id].error.code == status_code
        reported_error = convert_error.call_args.args[0]
        if error_type is None:
            assert isinstance(reported_error, HTTPException) and reported_error.status_code == 500
        else:
            convert_error.assert_called_once_with(error)
        if error_type is EngineDeadError:
            terminate.assert_called_once_with(server=app_state.server, engine=app_state.engine_client)
    else:
        if error_type is None:
            assert isinstance(task.exception(), HTTPException) and task.exception().status_code == 500
        else:
            assert task.exception() is error
    assert path.exists() is not terminal
    assert (bundle in owner.bundles) is not terminal
    if terminal_origin:
        # More than the default admission budget must remain usable after
        # terminal validation failures through the actual serving methods.
        for index in range(5):
            next_bundle = owner.reserve(4)
            next_path = tmp_path / f"guide-{index}"
            next_path.write_bytes(b"input")
            next_bundle.paths.add(str(next_path))
            next_bundle.descriptors = [{"frame_index": 100000, "image": str(next_path)}]
            request._guide_bundle = next_bundle
            next_id = f"invalid-guide-{index}"
            records[next_id] = SimpleNamespace(file_extension="mp4", status="queued")
            work = (
                namespace["_run_video_generation_job"](handler, request, next_id, app_state=app_state)
                if wrapped_job
                else handler.generate_video_bytes(request, next_id)
            )
            await asyncio.gather(next_bundle.submit(work), return_exceptions=True)
            assert next_bundle.engine_completed
            assert not next_path.exists() and not owner.bundles
            if wrapped_job:
                assert records[next_id].status == "failed" and records[next_id].error.code == 400
    storage.save.assert_not_called()
    await owner.drain()
    assert path.exists() is not terminal
    assert (bundle in owner.bundles) is not terminal
    bundle.close()  # These isolated engine/storage doubles have no external readers.


@pytest.mark.asyncio
@pytest.mark.parametrize("dispatched,completed", [(False, False), (True, True)])
async def test_predispatch_or_postcompletion_failure_can_release_inputs(tmp_path, dispatched, completed):
    owner = lifetime.GuidedRequestLifetime()
    bundle = owner.reserve(1)
    path = tmp_path / "guide"
    path.write_bytes(b"input")
    bundle.paths.add(str(path))

    async def work():
        bundle.engine_started = dispatched
        bundle.engine_completed = completed
        raise ValueError("local validation or output encoding failure")

    await asyncio.gather(bundle.submit(work()), return_exceptions=True)
    assert not path.exists()
    assert not owner.bundles


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_save", [False, True])
async def test_threaded_save_and_cancelled_delete_cannot_recreate_artifact(tmp_path, cancel_save):
    owner = lifetime.GuidedRequestLifetime()
    bundle = owner.reserve(1)
    guide, artifact = tmp_path / "guide", tmp_path / "output"
    guide.write_bytes(b"input")
    bundle.paths.add(str(guide))
    records = {"job": {"status": "in_progress"}}
    saving, deleting = asyncio.Event(), asyncio.Event()
    release_write = threading.Event()
    write_finished = threading.Event()
    loop = asyncio.get_running_loop()

    def write():
        loop.call_soon_threadsafe(saving.set)
        assert release_write.wait(5)
        artifact.write_bytes(b"video")
        write_finished.set()

    async def persist():
        async with bundle.lock:
            await bundle.finish_storage(asyncio.to_thread(write))
            if bundle.abandoned:
                artifact.unlink(missing_ok=True)
            else:
                records["job"] = {"status": "completed"}

    async def remove():
        deleting.set()
        async with bundle.lock:
            artifact.unlink(missing_ok=True)
            records.pop("job", None)

    task = bundle.submit(persist())
    try:
        await asyncio.wait_for(saving.wait(), 2)
        if cancel_save:
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert bundle.lock.locked()
        bundle.abandoned = True
        deletion = asyncio.create_task(bundle.finish_storage(remove()))
        await asyncio.wait_for(deleting.wait(), 2)
        assert bundle.lock.locked()
        assert records["job"]["status"] == "in_progress"
        deletion.cancel()
        assert not deletion.done()
        if not cancel_save:
            assert task.cancelling() == 0
        release_write.set()
        await asyncio.gather(task, deletion)
        assert write_finished.is_set()
        assert not artifact.exists()
        assert not records
        await owner.drain()
        if cancel_save:
            assert guide.exists() and bundle in owner.bundles
            bundle.close()
        else:
            assert not guide.exists() and not owner.bundles
    finally:
        release_write.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_delete_waiting_on_lock_removes_record_without_cancelling_worker(tmp_path):
    owner = lifetime.GuidedRequestLifetime()
    bundle = owner.reserve(1)
    guide = tmp_path / "guide"
    guide.write_bytes(b"input")
    bundle.paths.add(str(guide))
    records = {"job": {"status": "in_progress"}}
    entered, finish, deleting = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def generate():
        entered.set()
        await finish.wait()
        assert guide.exists()
        return b"discarded video"

    async def remove():
        deleting.set()
        async with bundle.lock:
            records.pop("job", None)

    task = bundle.submit(generate())
    await entered.wait()
    await bundle.lock.acquire()
    bundle.abandoned = True
    deletion = asyncio.create_task(bundle.finish_storage(remove()))
    try:
        await asyncio.wait_for(deleting.wait(), 2)
        assert records["job"]["status"] == "in_progress"
        deletion.cancel()
        bundle.lock.release()
        await deletion
        assert not records
        assert not task.done() and task.cancelling() == 0
        assert guide.exists() and bundle in owner.bundles
        finish.set()
        assert await task is None
        assert not guide.exists() and not owner.bundles
    finally:
        if bundle.lock.locked():
            bundle.lock.release()
        finish.set()
        await asyncio.gather(task, deletion, return_exceptions=True)
