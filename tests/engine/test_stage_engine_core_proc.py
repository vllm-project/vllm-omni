# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreRequestType, UtilityOutput
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.request import RequestStatus
from vllm.v1.serial_utils import run_method

from vllm_omni.engine.stage_engine_core_proc import (
    StageEngineCoreProc,
    _bind_first_audio_sink,
    _extend_handshake_timeout,
)
from vllm_omni.worker_v2.first_audio_sender import supports_in_process_first_audio

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_ordered_abort_supports_scheduler_without_overflow_fences(mocker):
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.scheduler = mocker.Mock(spec=Scheduler)

    engine._handle_client_request(EngineCoreRequestType.ABORT, ["request"])

    engine.scheduler.finish_requests.assert_called_once_with(["request"], RequestStatus.FINISHED_ABORTED)


def test_release_reaches_native_data_plane_runner(mocker):
    """MRv2 sends from the runner, so the release must reach its data plane."""
    runner = mocker.Mock()
    worker_wrapper = SimpleNamespace(worker=SimpleNamespace(model_runner=runner))
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.scheduler = SimpleNamespace(chunk_transfer_adapter=None, _native_data_plane=True)
    engine.model_executor = SimpleNamespace(
        collective_rpc=lambda method, args=(): [run_method(worker_wrapper, method, args, {})]
    )

    engine.omni_release_request_resources(["a", "b"])

    runner.release_omni_shm_resources.assert_called_once_with(["a", "b"])


@pytest.mark.parametrize("native", [False, True])
def test_turn_retirement_preserves_resident_scheduler_request(mocker, native):
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    request = mocker.Mock(status=RequestStatus.WAITING_FOR_STREAMING_REQ)
    adapter = None if native else mocker.Mock()
    engine.scheduler = SimpleNamespace(
        requests={"resident": request}, chunk_transfer_adapter=adapter, _native_data_plane=native
    )
    runner = mocker.Mock()
    wrapper = SimpleNamespace(worker=SimpleNamespace(model_runner=runner))
    engine.model_executor = SimpleNamespace(
        collective_rpc=lambda method, args=(): [run_method(wrapper, method, args, {})]
    )
    result = engine.omni_finish_turn("resident")
    assert engine.scheduler.requests["resident"] is request
    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    if native:
        assert result is None
        runner.finish_omni_turn.assert_called_once_with("resident")
    else:
        assert adapter is not None
        adapter.release_shm_resources.assert_called_once_with("resident", wait=True)
        assert result is adapter.release_shm_resources.return_value
    request.status = RequestStatus.RUNNING
    with pytest.raises(RuntimeError, match="unfinished turn"):
        engine.omni_finish_turn("resident")


def test_turn_retirement_replies_after_cleanup_without_blocking_engine_loop(mocker):
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    cleanup: Future[None] = Future()
    adapter = mocker.Mock()
    adapter.release_shm_resources.return_value = cleanup
    engine.scheduler = SimpleNamespace(requests={}, chunk_transfer_adapter=adapter)
    replies: list[UtilityOutput] = []
    output = UtilityOutput(call_id=1)

    EngineCoreProc._invoke_utility_method(
        "omni_finish_turn", lambda: engine.omni_finish_turn("resident"), output, replies.append
    )
    # The engine loop continues; the utility reply waits for the save thread.
    assert replies == []
    cleanup.set_exception(OSError("unlink failed"))
    assert replies == [output]
    assert output.failure_message is not None and "unlink failed" in output.failure_message


class WrappedUniProcExecutor(UniProcExecutor):
    pass


@pytest.mark.parametrize(
    "backend,executor_class,tp,pp,expected",
    [
        ("uni", UniProcExecutor, 1, 1, True),
        ("mp", MultiprocExecutor, 1, 1, False),
        (UniProcExecutor, UniProcExecutor, 1, 1, True),
        (MultiprocExecutor, MultiprocExecutor, 1, 1, False),
        ("uni", WrappedUniProcExecutor, 1, 1, True),
        ("uni", UniProcExecutor, 2, 1, False),
        ("uni", UniProcExecutor, 1, 2, False),
    ],
)
def test_decoder_capability_matches_sink_binding(mocker, backend, executor_class, tp, pp, expected):
    executor = executor_class.__new__(executor_class)
    executor.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            distributed_executor_backend=backend, tensor_parallel_size=tp, pipeline_parallel_size=pp
        )
    )
    executor.driver_worker = mocker.Mock()
    runner = executor.driver_worker.worker.model_runner
    runner.model.first_frame_decoder = object()
    assert supports_in_process_first_audio(executor.vllm_config) is expected
    assert _bind_first_audio_sink(executor, mocker.Mock(), mocker.Mock()) is expected
    assert runner.model_state.set_first_audio_sink.call_count == int(expected)


@pytest.mark.parametrize(
    "first_decoder,stream_decoder,stream_first_audio,expected",
    [(True, False, False, True), (False, True, False, False), (False, True, True, True), (False, False, True, False)],
)
def test_first_audio_sink_preserves_first_decoder_and_opts_in_stream_decoder(
    mocker,
    first_decoder,
    stream_decoder,
    stream_first_audio,
    expected,
):
    executor = UniProcExecutor.__new__(UniProcExecutor)
    executor.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1)
    )
    executor.driver_worker = mocker.Mock()
    runner = executor.driver_worker.worker.model_runner
    runner.model.first_frame_decoder = object() if first_decoder else None
    runner.model.stream_decoder = object() if stream_decoder else None
    runner.model.stream_first_audio = stream_first_audio
    runner._omni_data_plane = None
    assert _bind_first_audio_sink(executor, mocker.Mock(), mocker.Mock()) is expected
    assert runner.model_state.set_first_audio_sink.call_count == int(expected)


def test_preprocess_add_request_preserves_omni_fields():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    request = SimpleNamespace(
        request_id="internal",
        external_req_id="external",
        additional_information={"conditioning": "payload"},
    )
    scheduler_request = SimpleNamespace()

    with patch.object(
        EngineCoreProc,
        "preprocess_add_request",
        return_value=(scheduler_request, 3),
    ):
        result, current_wave = engine.preprocess_add_request(request)

    assert result is scheduler_request
    assert current_wave == 3
    assert result.external_req_id == "external"
    assert result.additional_information == {"conditioning": "payload"}


def test_first_audio_binding_preserves_talker_marker():
    import queue

    import torch

    from vllm_omni.data_entry_keys import FIRST_AUDIO_KEY
    from vllm_omni.engine import stage_engine_core_proc as module

    calls: dict[str, Any] = {}
    runner = SimpleNamespace(
        model=SimpleNamespace(first_frame_decoder=object()),
        model_state=SimpleNamespace(set_first_audio_sink=lambda sink: calls.update(sink=sink)),
    )
    executor = UniProcExecutor.__new__(UniProcExecutor)
    executor.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1)
    )
    executor.driver_worker = SimpleNamespace(worker=SimpleNamespace(model_runner=runner))
    outputs: queue.Queue = queue.Queue()
    scheduler = SimpleNamespace(requests={"r": SimpleNamespace(client_index=0)})
    assert module._bind_first_audio_sink(executor, outputs, scheduler)
    calls["sink"].prepare(["r"])(["r"], [torch.ones(2)], torch.tensor(24000))
    assert outputs.get_nowait()[1].outputs[0].multimodal_output[FIRST_AUDIO_KEY]
    executor.vllm_config.parallel_config.tensor_parallel_size = 2
    assert not module._bind_first_audio_sink(executor, outputs, scheduler)


@pytest.mark.parametrize(("timeout_s", "expected_minutes"), [(900.0, 15), (901.0, 16), (60.0, 5)])
def test_handshake_wait_is_extended_but_never_shortened(monkeypatch, timeout_s, expected_minutes) -> None:
    import vllm.v1.engine.core as engine_core_module

    monkeypatch.setattr(engine_core_module, "HANDSHAKE_TIMEOUT_MINS", 5)
    _extend_handshake_timeout(timeout_s)
    assert engine_core_module.HANDSHAKE_TIMEOUT_MINS == expected_minutes


def test_platform_compat_runs_before_engine_construction(monkeypatch):
    from vllm_omni.engine import stage_engine_core_proc as module

    calls = []

    class ConstructionStoppedError(Exception):
        pass

    def construct(self, *args, **kwargs):
        calls.append("engine")
        raise ConstructionStoppedError

    monkeypatch.setattr(module.current_omni_platform, "prepare_stage_engine", lambda: calls.append("platform"))
    monkeypatch.setattr(EngineCoreProc, "__init__", construct)
    with pytest.raises(ConstructionStoppedError):
        StageEngineCoreProc()
    assert calls == ["platform", "engine"]
