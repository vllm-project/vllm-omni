import pytest
from pytest_mock import MockerFixture
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine, StageRuntimeInfo
from vllm_omni.engine.stage_pool import StagePool

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_engine_core_request(request_id: str = "req-1") -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[1, 1, 1],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def test_build_add_request_message_preserves_additional_information(mocker: MockerFixture):
    engine = object.__new__(AsyncOmniEngine)
    params = SamplingParams(max_tokens=8)
    engine.default_sampling_params_list = [params]
    engine.stage_metadata = [StageRuntimeInfo(final_output=False, final_output_type=None, stage_type="llm")]
    engine.supported_tasks = ("speech",)

    input_processor = mocker.Mock()
    input_processor.process_inputs.return_value = _make_engine_core_request()
    engine.input_processor = input_processor

    output_processor = mocker.Mock()
    engine.output_processors = [output_processor]

    prompt = {
        "prompt_token_ids": [1, 1, 1],
        "additional_information": {
            "text": ["hello world"],
            "speaker": ["vivian"],
        },
    }

    msg = engine._build_add_request_message(
        request_id="req-1",
        prompt=prompt,
        sampling_params_list=[params],
        final_stage_id=0,
        arrival_time=0.0,
    )

    request = msg.prompt
    assert isinstance(request, OmniEngineCoreRequest)
    assert request.external_req_id == "req-1"
    assert request.additional_information is not None
    assert request.additional_information.entries["text"].list_data == ["hello world"]
    assert request.additional_information.entries["speaker"].list_data == ["vivian"]
    output_processor.add_request.assert_not_called()


def test_build_add_request_message_with_resumable_streaming(mocker: MockerFixture):
    engine = object.__new__(AsyncOmniEngine)
    params = SamplingParams(max_tokens=8)
    engine.default_sampling_params_list = [params]
    engine.stage_metadata = [StageRuntimeInfo(final_output=False, final_output_type=None, stage_type="llm")]
    engine.supported_tasks = ("generate",)

    input_processor = mocker.Mock()
    input_processor.process_inputs.return_value = _make_engine_core_request()
    engine.input_processor = input_processor

    output_processor = mocker.Mock()
    engine.output_processors = [output_processor]

    msg = engine._build_add_request_message(
        request_id="req-stream",
        prompt={"prompt_token_ids": [1, 2, 3]},
        sampling_params_list=[params],
        final_stage_id=0,
        resumable=True,
        message_type="streaming_update",
    )

    assert msg.type == "streaming_update"
    input_processor.process_inputs.assert_called_once()
    assert input_processor.process_inputs.call_args.kwargs["resumable"] is True


class _FakeStageClient:
    stage_type = "llm"
    final_output = False


def test_build_add_request_message_scopes_mm_uuids_to_selected_stage0_replica(mocker: MockerFixture):
    engine = object.__new__(AsyncOmniEngine)
    params = SamplingParams(max_tokens=8)
    engine.model = "test-model"
    engine.default_sampling_params_list = [params]
    engine.stage_metadata = [StageRuntimeInfo(final_output=False, final_output_type=None, stage_type="llm")]
    engine.supported_tasks = ("generate",)
    engine.stage_pools = [StagePool(0, [_FakeStageClient(), _FakeStageClient()])]

    seen_uuids: list[str] = []

    def process_inputs(**kwargs):
        prompt = kwargs["prompt"]
        seen_uuids.append(prompt["multi_modal_uuids"]["image"][0])
        return _make_engine_core_request(kwargs["request_id"])

    input_processor = mocker.Mock()
    input_processor.process_inputs.side_effect = process_inputs
    engine.input_processor = input_processor

    for request_id in ("req-1", "req-2"):
        engine._build_add_request_message(
            request_id=request_id,
            prompt={
                "prompt": "describe",
                "multi_modal_data": {"image": "same-image"},
            },
            sampling_params_list=[params],
            final_stage_id=0,
        )

    assert seen_uuids[0].startswith("stage0:rep0:")
    assert seen_uuids[1].startswith("stage0:rep1:")
    assert seen_uuids[0].removeprefix("stage0:rep0:") == seen_uuids[1].removeprefix("stage0:rep1:")


def test_stage_pool_replica_count_falls_back_to_clients():
    class PoolWithoutLiveNumReplicas:
        clients = [object(), None, object()]

    assert AsyncOmniEngine._stage_pool_replica_count(PoolWithoutLiveNumReplicas()) == 2


def test_stage_pool_is_distributed_falls_back_to_hub():
    class PoolWithoutIsDistributed:
        _hub = object()

    assert AsyncOmniEngine._stage_pool_is_distributed(PoolWithoutIsDistributed()) is True


def test_build_add_request_message_releases_preselected_replica_on_preprocess_error(mocker: MockerFixture):
    engine = object.__new__(AsyncOmniEngine)
    params = SamplingParams(max_tokens=8)
    engine.model = "test-model"
    engine.default_sampling_params_list = [params]
    engine.stage_metadata = [StageRuntimeInfo(final_output=False, final_output_type=None, stage_type="llm")]
    engine.supported_tasks = ("generate",)
    stage_pool = StagePool(0, [_FakeStageClient(), _FakeStageClient()])
    engine.stage_pools = [stage_pool]

    input_processor = mocker.Mock()
    input_processor.process_inputs.side_effect = RuntimeError("boom")
    engine.input_processor = input_processor

    with pytest.raises(RuntimeError, match="boom"):
        engine._build_add_request_message(
            request_id="req-error",
            prompt={
                "prompt": "describe",
                "multi_modal_data": {"image": "same-image"},
            },
            sampling_params_list=[params],
            final_stage_id=0,
        )

    assert stage_pool.get_bound_replica_id("req-error") is None
