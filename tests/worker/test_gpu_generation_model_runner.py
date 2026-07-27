import contextlib
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CacheConfig
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

import vllm_omni.worker.gpu_generation_model_runner as gen_runner_module
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


def _make_runner(multimodal_outputs):
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
    )
    runner.kv_connector_output = None
    runner.input_batch = _DummyInputBatch()
    runner.use_async_scheduling = False
    runner.device = torch.device("cpu")
    runner.supports_mm_inputs = False
    runner.speculative_config = None
    runner.routed_experts_initialized = False
    runner._async_chunk = False
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


class _StubSchedulerOutput:
    def __init__(self, total_num_scheduled_tokens):
        self.total_num_scheduled_tokens = total_num_scheduled_tokens
        self.num_scheduled_tokens = {"req-1": total_num_scheduled_tokens}
        self.finished_req_ids = set()
        self.kv_connector_metadata = None


def _make_guard_runner():
    # Stubbed far enough that a span escaping the guard reaches the real
    # `_prepare_inputs`, i.e. fails the way the reported crash does.
    runner = object.__new__(GPUGenerationModelRunner)
    runner.execute_model_state = None
    runner.routed_experts_initialized = False
    runner.speculative_config = None
    runner.model_config = SimpleNamespace(async_chunk=False)
    runner.cache_config = CacheConfig()
    runner.input_batch = _DummyInputBatch()
    runner.model = object()
    runner.parallel_config = SimpleNamespace(
        distributed_executor_backend=None,
        data_parallel_size=1,
    )
    runner._dummy_run_calls = []
    runner._dummy_run = lambda n: runner._dummy_run_calls.append(n)
    runner._update_states = lambda scheduler_output: None
    runner.synchronize_input_prep = contextlib.nullcontext
    runner.attach_omni_connector_output = lambda result: result
    return runner


@pytest.mark.parametrize("total", [-1, -512, 0])
def test_execute_model_skips_non_positive_scheduled_span(monkeypatch, total):
    """#5196: a negative span is truthy, so it used to reach `_prepare_inputs`,
    whose `assert total_num_scheduled_tokens > 0` killed the stage EngineCore."""
    monkeypatch.setattr(gen_runner_module, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(gen_runner_module, "has_kv_transfer_group", lambda: False)
    runner = _make_guard_runner()

    output = GPUGenerationModelRunner.execute_model(runner, _StubSchedulerOutput(total))

    assert output is EMPTY_MODEL_RUNNER_OUTPUT
    assert runner._dummy_run_calls == []


@pytest.mark.parametrize("total", [-1, 0])
def test_execute_model_zero_tokens_runs_dp_dummy_run(monkeypatch, total):
    """A 0-token step under external_launcher + DP>1 must `_dummy_run(1)` so
    `coordinate_batch_across_dp` stays in sync (it used to be shadowed by an
    unconditional early return and never ran -> DP hang risk)."""
    monkeypatch.setattr(gen_runner_module, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(gen_runner_module, "has_kv_transfer_group", lambda: False)
    runner = _make_guard_runner()
    runner.parallel_config = SimpleNamespace(
        distributed_executor_backend="external_launcher",
        data_parallel_size=2,
    )

    output = GPUGenerationModelRunner.execute_model(runner, _StubSchedulerOutput(total))

    assert output is EMPTY_MODEL_RUNNER_OUTPUT
    assert runner._dummy_run_calls == [1]


def test_execute_model_zero_tokens_kv_connector_no_forward(monkeypatch):
    """With a KV-transfer group, a 0-token step must go through
    `kv_connector_no_forward` (also shadowed by the old early return)."""
    monkeypatch.setattr(gen_runner_module, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(gen_runner_module, "has_kv_transfer_group", lambda: True)
    runner = _make_guard_runner()
    sentinel = object()
    calls = []
    runner.kv_connector_no_forward = lambda scheduler_output, vllm_config: (
        calls.append(scheduler_output),
        sentinel,
    )[1]
    runner.vllm_config = object()

    output = GPUGenerationModelRunner.execute_model(runner, _StubSchedulerOutput(0))

    assert output is sentinel
    assert len(calls) == 1
