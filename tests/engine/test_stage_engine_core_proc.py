# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from typing import Any

import pytest
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.request import RequestStatus

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


def test_preprocess_add_request_preserves_omni_fields(mocker):
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    request = SimpleNamespace(
        request_id="internal",
        external_req_id="external",
        additional_information={"conditioning": "payload"},
    )
    scheduler_request = SimpleNamespace()

    mocker.patch.object(
        EngineCoreProc,
        "preprocess_add_request",
        return_value=(scheduler_request, 3),
    )
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


def test_talker_hasher_uses_resolved_scheduler_hash_size_and_configured_algorithm():
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256_cbor
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash

    from vllm_omni.request import OmniRequest

    init_none_hash(sha256_cbor)
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.scheduler = SimpleNamespace(hash_block_size=2)
    engine.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16, prefix_caching_hash_algo="sha256_cbor", enable_prefix_caching=True),
        model_config=SimpleNamespace(model_stage="talker", model_arch="Qwen3OmniMoeForConditionalGeneration"),
    )
    engine._init_talker_request_hasher()
    request = OmniRequest(
        request_id="resolved-hash",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        block_hasher=engine.request_block_hasher,
    )
    assert len(request.block_hashes) == 1
    hashes = request.block_hashes
    request.block_hashes = []
    assert hashes == get_request_block_hasher(2, sha256_cbor)(request)


def test_non_talker_stage_keeps_native_request_hasher():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        model_config=SimpleNamespace(model_stage="thinker"),
    )
    engine.request_block_hasher = original = object()
    engine._init_talker_request_hasher()
    assert engine.request_block_hasher is original


@pytest.mark.parametrize(("timeout_s", "expected_minutes"), [(900.0, 15), (901.0, 16), (60.0, 5)])
def test_handshake_wait_is_extended_but_never_shortened(monkeypatch, timeout_s, expected_minutes) -> None:
    import vllm.v1.engine.core as engine_core_module

    monkeypatch.setattr(engine_core_module, "HANDSHAKE_TIMEOUT_MINS", 5)
    _extend_handshake_timeout(timeout_s)
    assert engine_core_module.HANDSHAKE_TIMEOUT_MINS == expected_minutes
