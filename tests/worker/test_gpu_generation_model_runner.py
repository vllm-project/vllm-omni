import pytest
import torch

from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyInputBatch:
    def __init__(self, num_reqs=1):
        self.req_ids = [f"req-{i}" for i in range(num_reqs)]
        self.req_id_to_index = {rid: i for i, rid in enumerate(self.req_ids)}
        self.num_reqs = num_reqs
        self.vocab_size = 10


def _make_runner(multimodal_outputs, num_reqs=1):
    runner = object.__new__(GPUGenerationModelRunner)
    runner.execute_model_state = (
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
    )
    runner.kv_connector_output = None
    runner.input_batch = _DummyInputBatch(num_reqs)
    runner.use_async_scheduling = False
    runner.device = torch.device("cpu")
    runner.supports_mm_inputs = False
    runner.speculative_config = None
    return runner


def test_sample_tokens_tensor_output():
    multimodal_outputs = torch.randn(1, 2, 3)
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.pooler_output) == 1
    assert output.pooler_output[0]["model_outputs"].shape == (2, 3)


def test_sample_tokens_list_output():
    multimodal_outputs = [torch.randn(2, 1)]
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.pooler_output) == 1
    assert output.pooler_output[0]["model_outputs"].shape == (2, 1)


def test_sample_tokens_list_allows_none_output():
    multimodal_outputs = [None]
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.pooler_output) == 1
    assert output.pooler_output[0]["model_outputs"] is None


def test_sample_tokens_dict_output():
    multimodal_outputs = {"audio": torch.randn(1, 4), "unused": None}
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.pooler_output) == 1
    assert "audio" in output.pooler_output[0]
    assert "unused" not in output.pooler_output[0]
    assert output.pooler_output[0]["audio"].shape == (1, 4)


# ------------------------------------------------------------------
# Batched (num_reqs > 1) tests — these catch the contradictory
# assertion bug where shape[0]==1 AND shape[0]==num_reqs were both
# required simultaneously.
# ------------------------------------------------------------------


def test_sample_tokens_tensor_batched():
    """Tensor output with batch dim matching num_reqs must work."""
    num_reqs = 3
    multimodal_outputs = torch.randn(num_reqs, 4, 5)
    runner = _make_runner(multimodal_outputs, num_reqs=num_reqs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.pooler_output) == num_reqs
    for i in range(num_reqs):
        assert output.pooler_output[i]["model_outputs"].shape == (4, 5)


def test_sample_tokens_list_batched():
    """List output with one entry per request must work."""
    num_reqs = 3
    multimodal_outputs = [torch.randn(8) for _ in range(num_reqs)]
    runner = _make_runner(multimodal_outputs, num_reqs=num_reqs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.pooler_output) == num_reqs
    for i in range(num_reqs):
        assert output.pooler_output[i]["model_outputs"].shape == (8,)


def test_sample_tokens_dict_list_batched():
    """Dict output with per-request lists must work."""
    num_reqs = 2
    multimodal_outputs = {
        "model_outputs": [torch.randn(16) for _ in range(num_reqs)],
    }
    runner = _make_runner(multimodal_outputs, num_reqs=num_reqs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.pooler_output) == num_reqs
    for i in range(num_reqs):
        assert output.pooler_output[i]["model_outputs"].shape == (16,)


def test_sample_tokens_tensor_mismatched_batch_raises():
    """Tensor with wrong batch dim should raise AssertionError."""
    multimodal_outputs = torch.randn(2, 4)
    runner = _make_runner(multimodal_outputs, num_reqs=3)

    with pytest.raises(AssertionError, match="num_reqs"):
        GPUGenerationModelRunner.sample_tokens(runner)


def test_sample_tokens_list_mismatched_batch_raises():
    """List with wrong length should raise AssertionError."""
    multimodal_outputs = [torch.randn(4), torch.randn(4)]
    runner = _make_runner(multimodal_outputs, num_reqs=3)

    with pytest.raises(AssertionError, match="num_reqs"):
        GPUGenerationModelRunner.sample_tokens(runner)
