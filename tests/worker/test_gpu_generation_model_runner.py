# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import contextlib
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CacheConfig, CUDAGraphMode
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

import vllm_omni.worker.gpu_generation_model_runner as gen_runner_module
from vllm_omni.model_executor.models.indextts2 import runner as indextts_runner_module
from vllm_omni.model_executor.models.indextts2.runner import (
    IndexTTS2GenerationModelRunner,
)
from vllm_omni.worker.gpu_generation_model_runner import (
    ExecuteModelState,
    GPUGenerationModelRunner,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _StopExecutionError(Exception):
    """Halts execute_model at the model boundary."""


class _DummyInputBatch:
    def __init__(self, req_ids=("req-1",)):
        self.req_ids = list(req_ids)
        self.req_id_to_index = {req_id: index for index, req_id in enumerate(req_ids)}
        self.num_reqs = len(self.req_ids)
        self.vocab_size = 10


def _make_runner(multimodal_outputs, num_reqs=1, *, req_ids=None, runner_cls=GPUGenerationModelRunner):
    if req_ids is None:
        req_ids = tuple(f"req-{i + 1}" for i in range(num_reqs))
    runner = object.__new__(runner_cls)
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
    runner.input_batch = _DummyInputBatch(req_ids)
    runner.use_async_scheduling = False
    runner.device = torch.device("cpu")
    runner.supports_mm_inputs = False
    runner.speculative_config = None
    runner.routed_experts_initialized = False
    runner._async_chunk = False
    runner._generation_finished_req_ids = set()
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


def test_sample_tokens_dict_maps_only_completed_rows():
    completed_audio = torch.ones(4)
    sample_rate = torch.tensor(22050, dtype=torch.int32)
    runner = _make_runner(
        {
            "audio": [completed_audio, None],
            "sr": [sample_rate, None],
        },
        req_ids=("done", "pending"),
        runner_cls=IndexTTS2GenerationModelRunner,
    )
    runner._stepwise_output_req_ids = ["done", "pending"]
    runner._generation_finished_req_ids = {"done"}

    output = IndexTTS2GenerationModelRunner.sample_tokens(runner)

    assert output.multimodal_outputs is not None
    assert output.multimodal_outputs[1] is None
    torch.testing.assert_close(output.multimodal_outputs[0]["audio"], completed_audio)
    torch.testing.assert_close(output.multimodal_outputs[0]["sr"], sample_rate)
    assert output.generation_finished_req_ids == {"done"}


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
        self.scheduled_new_reqs = []
        self.stepwise_req_ids: list[str] = []
        self.scheduled_encoder_inputs = {}
        self.scheduled_spec_decode_tokens = {}
        self.num_common_prefix_blocks = 0


def _make_guard_runner(runner_cls=GPUGenerationModelRunner):
    # Stubbed far enough that a span escaping the guard reaches the real
    # `_prepare_inputs`, i.e. fails the way the reported crash does.
    runner = object.__new__(runner_cls)
    runner.execute_model_state = None
    runner.routed_experts_initialized = False
    runner.speculative_config = None
    runner.model_config = SimpleNamespace(async_chunk=False)
    runner.cache_config = CacheConfig()
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
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


@pytest.mark.parametrize("total_num_scheduled_tokens", [0, 1])
def test_execute_model_runs_stepwise_without_input_prep(
    monkeypatch,
    total_num_scheduled_tokens,
):
    monkeypatch.setattr(gen_runner_module, "has_kv_transfer_group", lambda: False)
    runner = _make_guard_runner(IndexTTS2GenerationModelRunner)
    runner.device = torch.device("cpu")
    runner.model_intermediate_buffer = {
        "req-a": {"value": "a"},
        "req-b": {"value": "b"},
    }
    runner._active_stepwise_req_ids = None
    runner.model = SimpleNamespace(
        requires_request_ids=True,
        take_finished_request_ids=lambda: {"req-b"},
    )
    captured = {}

    def model_forward(**kwargs):
        kwargs["model_intermediate_buffer"] = [
            runner.model_intermediate_buffer[request_id] for request_id in runner._active_stepwise_req_ids
        ]
        captured.update(kwargs)
        return {"audio": [None, torch.ones(4)]}

    runner._model_forward = model_forward
    runner.extract_multimodal_outputs = lambda outputs: (None, outputs)
    sampled_output = object()
    sample_calls = []

    def sample_tokens():
        sample_calls.append(True)
        return sampled_output

    runner.sample_tokens = sample_tokens
    runner._prepare_inputs = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("zero-token stepwise work reached _prepare_inputs")
    )
    scheduler_output = _StubSchedulerOutput(total_num_scheduled_tokens)
    scheduler_output.stepwise_req_ids = ["req-a", "req-b"]

    output = IndexTTS2GenerationModelRunner.execute_model(runner, scheduler_output)

    if total_num_scheduled_tokens == 0:
        assert output is sampled_output
        assert sample_calls == [True]
    else:
        assert output is None
        assert sample_calls == []
    assert captured["request_ids"] == ["req-a", "req-b"]
    assert captured["model_intermediate_buffer"] == [
        {"value": "a"},
        {"value": "b"},
    ]
    assert captured["input_ids"].numel() == 0
    assert runner._generation_finished_req_ids == {"req-b"}
    assert runner._stepwise_output_req_ids == ["req-a", "req-b"]


def test_stepwise_forward_preserves_native_kv_connector_lifecycle(monkeypatch):
    lifecycle = []
    kv_connector_output = object()

    @contextlib.contextmanager
    def tracked_context(name, value=None):
        lifecycle.append(f"{name}:enter")
        try:
            yield value
        finally:
            lifecycle.append(f"{name}:exit")

    monkeypatch.setattr(
        indextts_runner_module,
        "has_kv_transfer_group",
        lambda: True,
    )
    monkeypatch.setattr(
        indextts_runner_module,
        "set_forward_context",
        lambda *_args, **_kwargs: tracked_context("forward-context"),
    )
    runner = _make_guard_runner(IndexTTS2GenerationModelRunner)
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    runner.model_intermediate_buffer = {"req-a": {"value": "a"}}
    runner._active_stepwise_req_ids = None
    runner._sync_local_stage_payloads = lambda: None
    runner.maybe_get_kv_connector_output = lambda scheduler_output, defer_finalize: tracked_context(
        "kv-connector",
        kv_connector_output,
    )
    runner.model = SimpleNamespace(
        take_finished_request_ids=lambda: set(),
    )

    def model_forward(**_kwargs):
        lifecycle.append("model-forward")

    runner._model_forward = model_forward
    runner.extract_multimodal_outputs = lambda outputs: (None, outputs)
    sampled_output = object()

    def sample_tokens():
        assert runner.kv_connector_output is kv_connector_output
        lifecycle.append("sample-tokens")
        return sampled_output

    runner.sample_tokens = sample_tokens
    scheduler_output = _StubSchedulerOutput(0)

    output = runner._execute_stepwise_generation(
        scheduler_output,
        ["req-a"],
        None,
    )

    assert output is sampled_output
    assert lifecycle == [
        "forward-context:enter",
        "kv-connector:enter",
        "model-forward",
        "kv-connector:exit",
        "forward-context:exit",
        "sample-tokens",
    ]


def test_execute_model_flushes_finished_state_without_stepwise_work(monkeypatch):
    monkeypatch.setattr(gen_runner_module, "has_kv_transfer_group", lambda: False)
    runner = _make_guard_runner(IndexTTS2GenerationModelRunner)
    flush_calls = []
    runner.model = SimpleNamespace(
        flush_finished_requests=lambda: flush_calls.append(True),
    )
    runner.attach_omni_connector_output = lambda output: output
    scheduler_output = _StubSchedulerOutput(0)
    scheduler_output.stepwise_req_ids = []
    scheduler_output.finished_req_ids = {"done"}
    runner.model.on_requests_finished = lambda request_ids: None

    IndexTTS2GenerationModelRunner.execute_model(runner, scheduler_output)

    assert flush_calls == [True]


def test_execute_model_drops_cancelled_request_from_stepwise_work(monkeypatch):
    monkeypatch.setattr(gen_runner_module, "has_kv_transfer_group", lambda: False)
    runner = _make_guard_runner(IndexTTS2GenerationModelRunner)
    lifecycle_calls: list[tuple[str, set[str] | None]] = []
    runner.model = SimpleNamespace(
        requires_request_ids=True,
        on_requests_finished=lambda request_ids: lifecycle_calls.append(("finish", set(request_ids))),
        flush_finished_requests=lambda: lifecycle_calls.append(("flush", None)),
    )
    runner.attach_omni_connector_output = lambda output: output
    runner._execute_stepwise_generation = lambda *_args: (_ for _ in ()).throw(
        AssertionError("cancelled request reached stepwise forward")
    )
    scheduler_output = _StubSchedulerOutput(0)
    scheduler_output.num_scheduled_tokens = {}
    scheduler_output.stepwise_req_ids = ["cancelled"]
    scheduler_output.finished_req_ids = {"cancelled"}

    output = IndexTTS2GenerationModelRunner.execute_model(runner, scheduler_output)

    assert output is EMPTY_MODEL_RUNNER_OUTPUT
    assert lifecycle_calls == [
        ("finish", {"cancelled"}),
        ("flush", None),
    ]


def test_execute_model_reinitializes_resubmitted_stepwise_request(monkeypatch):
    monkeypatch.setattr(gen_runner_module, "has_kv_transfer_group", lambda: False)
    runner = _make_guard_runner(IndexTTS2GenerationModelRunner)
    lifecycle_calls: list[tuple[str, set[str] | None]] = []
    runner.model = SimpleNamespace(
        requires_request_ids=True,
        on_requests_finished=lambda request_ids: lifecycle_calls.append(("finish", set(request_ids))),
        flush_finished_requests=lambda: lifecycle_calls.append(("flush", None)),
    )
    runner.model_intermediate_buffer = {"same-id": {"generation": "new"}}
    captured_request_ids = []
    runner._execute_stepwise_generation = lambda _output, request_ids, _corrections: captured_request_ids.extend(
        request_ids
    )
    scheduler_output = _StubSchedulerOutput(1)
    scheduler_output.num_scheduled_tokens = {"same-id": 1}
    scheduler_output.stepwise_req_ids = ["same-id"]
    scheduler_output.finished_req_ids = {"same-id"}
    scheduler_output.scheduled_new_reqs = [SimpleNamespace(req_id="same-id")]

    output = IndexTTS2GenerationModelRunner.execute_model(runner, scheduler_output)

    assert output is None
    assert captured_request_ids == ["same-id"]
    assert lifecycle_calls == [
        ("finish", {"same-id"}),
        ("flush", None),
    ]


def test_sample_tokens_uses_explicit_stepwise_output_order():
    runner = _make_runner(
        {"audio": [torch.ones(2), None]},
        req_ids=("stale-input-batch",),
        runner_cls=IndexTTS2GenerationModelRunner,
    )
    runner._stepwise_output_req_ids = ["done", "pending"]

    output = IndexTTS2GenerationModelRunner.sample_tokens(runner)

    assert output.req_ids == ["done", "pending"]
    assert output.req_id_to_index == {"done": 0, "pending": 1}
    assert output.multimodal_outputs is not None
    assert output.multimodal_outputs[1] is None


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

    def kv_connector_no_forward(scheduler_output, vllm_config):
        calls.append(scheduler_output)
        return sentinel

    runner.kv_connector_no_forward = kv_connector_no_forward
    runner.vllm_config = object()

    output = GPUGenerationModelRunner.execute_model(runner, _StubSchedulerOutput(0))

    assert output is sentinel
    assert len(calls) == 1


def test_execute_model_zero_tokens_ec_producer_takes_encoder_path(monkeypatch):
    """AR lock-step: the EC-producer block sits ABOVE the zero-token block
    (matching GPUARModelRunner), so a 0-token step on an EC producer returns
    the empty ENCODER output. Before the C1 reorder, the unconditional early
    return won and produced EMPTY_MODEL_RUNNER_OUTPUT instead — this is the
    one cell whose behaviour the reorder changed."""
    monkeypatch.setattr(gen_runner_module, "has_ec_transfer", lambda: True)
    monkeypatch.setattr(gen_runner_module, "get_ec_transfer", lambda: SimpleNamespace(is_consumer=False))
    monkeypatch.setattr(gen_runner_module, "has_kv_transfer_group", lambda: False)
    sentinel = object()
    monkeypatch.setattr(gen_runner_module, "make_empty_encoder_model_runner_output", lambda so: sentinel)
    runner = _make_guard_runner()
    runner.encoder_cache = {}
    runner.maybe_get_ec_connector_output = lambda scheduler_output, encoder_cache: contextlib.nullcontext()
    encoder_calls = []
    runner._execute_mm_encoder = lambda so: encoder_calls.append(so)

    output = GPUGenerationModelRunner.execute_model(runner, _StubSchedulerOutput(0))

    assert output is sentinel
    assert len(encoder_calls) == 1
    assert runner._dummy_run_calls == []


class TestSampleTokensBatched:
    """RFC #5450 C5: `sample_tokens` must slice per request for num_reqs > 1.
    The tensor branch's asserts used to force num_reqs == 1 and the list
    branch built a length-1 payload misaligned with `req_ids`."""

    def test_tensor_output_sliced_per_request(self):
        multimodal_outputs = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
        runner = _make_runner(multimodal_outputs, num_reqs=2)

        output = GPUGenerationModelRunner.sample_tokens(runner)

        assert len(output.multimodal_outputs) == 2
        assert torch.equal(output.multimodal_outputs[0]["model_outputs"], multimodal_outputs[0])
        assert torch.equal(output.multimodal_outputs[1]["model_outputs"], multimodal_outputs[1])

    def test_list_output_one_entry_per_request(self):
        entries = [torch.zeros(2, 1), torch.ones(2, 1)]
        runner = _make_runner(list(entries), num_reqs=2)

        output = GPUGenerationModelRunner.sample_tokens(runner)

        assert len(output.multimodal_outputs) == 2
        assert torch.equal(output.multimodal_outputs[0]["model_outputs"], entries[0])
        assert torch.equal(output.multimodal_outputs[1]["model_outputs"], entries[1])

    def test_tensor_output_batch_mismatch_raises(self):
        runner = _make_runner(torch.randn(1, 2, 3), num_reqs=2)

        with pytest.raises(ValueError, match="leading dim 1 .* 2 requests"):
            GPUGenerationModelRunner.sample_tokens(runner)

    def test_list_output_batch_mismatch_raises(self):
        runner = _make_runner([torch.randn(2, 1)], num_reqs=2)

        with pytest.raises(ValueError, match="length 1 .* 2 requests"):
            GPUGenerationModelRunner.sample_tokens(runner)


@pytest.mark.parametrize("exact_shape, expected_len", [(True, 3), (False, 4)])
def test_exact_shape_models_receive_unpadded_input_ids(monkeypatch, exact_shape, expected_len):
    """#6712: the cudagraph dispatcher rounds a 3-token batch up to a capture
    size of 4, so ``_preprocess`` returns a 4-element ``input_ids`` while
    ``seq_token_counts`` stays ``[3]``. Models opting into
    ``requires_exact_input_shape`` must receive the trimmed view; everything
    else must keep the padded buffer.
    """
    monkeypatch.setattr(gen_runner_module, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(gen_runner_module, "has_kv_transfer_group", lambda: False)

    runner = _make_guard_runner()
    runner.model = SimpleNamespace(
        has_preprocess=True,
        requires_request_ids=False,
        requires_exact_input_shape=exact_shape,
    )
    runner.cascade_attn_enabled = False
    runner.num_prompt_logprobs = None
    runner.vllm_config = None
    runner.parallel_config = SimpleNamespace(
        distributed_executor_backend=None,
        data_parallel_size=1,
        use_ubatching=False,
        num_ubatches=1,
    )

    monkeypatch.setattr(GPUGenerationModelRunner, "_prepare_inputs", lambda self, so, nst: (None, None, 1))
    monkeypatch.setattr(
        GPUGenerationModelRunner,
        "_determine_batch_execution_and_padding",
        lambda self, **kw: (
            CUDAGraphMode.PIECEWISE,
            SimpleNamespace(num_tokens=4, num_reqs=1),
            False,
            None,
            None,
        ),
    )
    monkeypatch.setattr(GPUGenerationModelRunner, "_get_slot_mappings", lambda self, **kw: (None, None))
    monkeypatch.setattr(GPUGenerationModelRunner, "_build_attention_metadata", lambda self, **kw: (None, None))
    monkeypatch.setattr(
        GPUGenerationModelRunner,
        "_maybe_attach_attention_metadata_extensions",
        lambda self, **kw: None,
    )
    monkeypatch.setattr(gen_runner_module, "set_forward_context", lambda *a, **kw: contextlib.nullcontext())

    def _fake_preprocess(self, so, num_input_tokens, itensors=None):
        return (torch.zeros(num_input_tokens, dtype=torch.int32), None, None, None, {}, None)

    monkeypatch.setattr(GPUGenerationModelRunner, "_preprocess", _fake_preprocess)

    seen = {}

    def _capture(self, **kwargs):
        seen["input_ids"] = kwargs["input_ids"]
        seen["counts"] = kwargs["model_kwargs"]["seq_token_counts"]
        raise _StopExecutionError

    monkeypatch.setattr(GPUGenerationModelRunner, "_run_generation_model", _capture)

    with pytest.raises(_StopExecutionError):
        GPUGenerationModelRunner.execute_model(runner, _StubSchedulerOutput(3))

    assert seen["input_ids"].numel() == expected_len
    assert seen["counts"] == [3]
    if exact_shape:
        assert sum(seen["counts"]) == seen["input_ids"].numel()
