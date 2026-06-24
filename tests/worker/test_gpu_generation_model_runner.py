import pytest
import torch

from vllm_omni.worker.gpu_generation_model_runner import (
    ExecuteModelState,
    GPUGenerationModelRunner,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyInputBatch:
    def __init__(self):
        self.req_ids = ["req-1"]
        self.req_id_to_index = {"req-1": 0}
        self.num_reqs = 1
        self.vocab_size = 10


def _make_runner(multimodal_outputs, req_ids=None, req_id_to_index=None):
    req_ids = req_ids or ["req-1"]
    req_id_to_index = req_id_to_index or {"req-1": 0}
    runner = object.__new__(GPUGenerationModelRunner)
    runner.execute_model_state = ExecuteModelState(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        multimodal_outputs,
        None,
        req_ids,
        req_id_to_index,
    )
    runner.kv_connector_output = None
    runner.input_batch = _DummyInputBatch()
    runner.use_async_scheduling = False
    runner.device = torch.device("cpu")
    runner.supports_mm_inputs = False
    runner.speculative_config = None
    runner.routed_experts_initialized = False
    return runner


def test_sample_tokens_tensor_output():
    multimodal_outputs = torch.randn(1, 2, 3)
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.multimodal_outputs) == 1
    assert output.multimodal_outputs[0]["model_outputs"].shape == (2, 3)


def test_sample_tokens_list_output():
    multimodal_outputs = [torch.randn(2, 1)]
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.multimodal_outputs) == 1
    assert output.multimodal_outputs[0]["model_outputs"].shape == (2, 1)


def test_sample_tokens_list_allows_none_output():
    multimodal_outputs = [None]
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.multimodal_outputs) == 1
    assert output.multimodal_outputs[0]["model_outputs"] is None


def test_sample_tokens_dict_output():
    multimodal_outputs = {"audio": torch.randn(1, 4), "unused": None}
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.multimodal_outputs) == 1
    assert "audio" in output.multimodal_outputs[0]
    assert "unused" not in output.multimodal_outputs[0]
    assert output.multimodal_outputs[0]["audio"].shape == (1, 4)


def test_sample_tokens_uses_execute_snapshot_when_input_batch_mutates():
    multimodal_outputs = {
        "model_outputs": [torch.randn(1, 2)],
        "sr": [torch.tensor(24_000, dtype=torch.int32)],
    }
    runner = _make_runner(
        multimodal_outputs,
        req_ids=["req-1"],
        req_id_to_index={"req-1": 0},
    )
    runner.input_batch.req_ids = ["req-1", "req-2", "req-3"]
    runner.input_batch.req_id_to_index = {"req-1": 0, "req-2": 1, "req-3": 2}
    runner.input_batch.num_reqs = 3

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert output.req_ids == ["req-1"]
    assert output.req_id_to_index == {"req-1": 0}
    assert len(output.multimodal_outputs) == 1
    assert output.multimodal_outputs[0]["model_outputs"].shape == (1, 2)
    assert output.multimodal_outputs[0]["sr"].item() == 24_000
