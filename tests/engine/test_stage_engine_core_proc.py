# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.executor.uniproc_executor import UniProcExecutor

from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc, _bind_first_audio_sink
from vllm_omni.worker_v2.first_audio_sender import supports_in_process_first_audio

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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


@pytest.mark.parametrize("has_plane", [False, True])
def test_native_resource_release_reaches_worker_owned_plane(mocker, has_plane):
    from vllm_omni.worker.mixins import OmniWorkerMixin

    plane = mocker.Mock() if has_plane else None
    worker = SimpleNamespace(model_runner=SimpleNamespace(_omni_data_plane=plane))
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(use_v2_model_runner=True, supports_native_mrv2_data_plane=True)
    )
    engine.model_executor = mocker.Mock()
    engine.model_executor.collective_rpc.side_effect = lambda method, args: getattr(OmniWorkerMixin, method)(
        worker, *args
    )
    engine.omni_release_request_resources(["external"])
    engine.model_executor.collective_rpc.assert_called_once_with("omni_release_request_resources", args=(["external"],))
    if has_plane:
        plane.release_request_resources.assert_called_once_with(["external"])


def test_legacy_resource_release_reaches_scheduler_adapter(mocker):
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.vllm_config = SimpleNamespace(model_config=SimpleNamespace(use_v2_model_runner=False))
    adapter = mocker.Mock()
    engine.scheduler = SimpleNamespace(chunk_transfer_adapter=adapter)
    engine.omni_release_request_resources(["external"])
    adapter.release_request_resources.assert_called_once()
    args, kwargs = adapter.release_request_resources.call_args
    assert args == ("external",)
    assert kwargs["deadline"] > 0
    adapter.release_request_resources.return_value.result.assert_called_once()


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
