# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch, split_diffusion_output_by_request

pytestmark = [pytest.mark.diffusion]


@contextmanager
def _noop_forward_context(*args, **kwargs):
    del args, kwargs
    yield


class _DummyPipeline:
    supports_request_batch = True

    def __init__(self, output):
        self._output = output
        self.forward_calls = 0
        self.last_req = None

    def forward(self, req):
        self.last_req = req
        self.forward_calls += 1
        return [self._output]


class _SingleRequestBatchPipeline:
    supports_request_batch = False

    def __init__(self):
        self.last_req = None

    def forward(self, req):
        self.last_req = req
        return [DiffusionOutput(output=req.prompts[0])]


class _SingleRequestDiffusionOutputPipeline:
    supports_request_batch = False

    def __init__(self):
        self.last_req = None

    def forward(self, req):
        self.last_req = req
        return DiffusionOutput(output=req.prompts[0])


class _ChunkStepPipeline:
    device = torch.device("cpu")
    supports_step_execution = True

    def __init__(self, outputs):
        self._outputs = outputs
        self.prepare_calls = 0
        self.decode_calls = 0

    def prepare_encode(self, state):
        self.prepare_calls += 1
        state.prompt_embeds = torch.zeros(1, 1, 1)
        state.latents = torch.zeros(1, 1)
        state.timesteps = torch.tensor([1.0, 0.0])
        state.step_index = 0
        state.step_in_chunk = 0
        state.chunk_num_steps = 2
        state.total_chunks = len(self._outputs)
        return state

    def denoise_step(self, input_batch, states):
        del states
        return torch.ones_like(input_batch.latents)

    def step_scheduler(self, state, noise_pred):
        state.latents = noise_pred
        state.step_index += 1
        state.step_in_chunk += 1

    def post_decode(self, state):
        output = self._outputs[self.decode_calls]
        self.decode_calls += 1
        state.chunk_index += 1
        state.step_index = 0
        state.step_in_chunk = 0
        if not state.request_denoise_completed:
            state.latents = torch.zeros(1, 1)
        return output


def _make_request(skip_cache_refresh: bool = True):
    sampling_params = SimpleNamespace(
        generator=None,
        seed=None,
        generator_device=None,
        num_inference_steps=4,
    )
    return SimpleNamespace(
        request_id="req-test",
        prompt="a prompt",
        sampling_params=sampling_params,
        skip_cache_refresh=skip_cache_refresh,
        kv_sender_info=None,
    )


def _make_request_with_params(req_id: str, sampling_params):
    return SimpleNamespace(
        request_id=req_id,
        prompt=f"prompt-{req_id}",
        prompts=[f"prompt-{req_id}"],
        sampling_params=sampling_params,
        skip_cache_refresh=True,
    )


def _fake_platform_for_peak_memory():
    return SimpleNamespace(
        reset_peak_memory_stats=lambda: None,
        max_memory_reserved=lambda: 0,
        max_memory_allocated=lambda: 0,
    )


def _make_runner(cache_backend, cache_backend_name: str, enable_cache_dit_summary: bool = True):
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    runner.pipeline = _DummyPipeline(output=DiffusionOutput(output="ok"))
    runner.cache_backend = cache_backend
    runner.offload_backend = None
    runner.state_cache = {}
    runner.prompt_embed_cache = None
    runner.od_config = SimpleNamespace(
        cache_backend=cache_backend_name,
        enable_cache_dit_summary=enable_cache_dit_summary,
        parallel_config=SimpleNamespace(use_hsdp=False),
        streaming_output=False,
    )
    runner.kv_transfer_manager = SimpleNamespace(
        receive_kv_cache=lambda req, target_device=None: None,
        receive_multi_kv_cache=lambda req, cfg_kv_collect_func=None, target_device=None: None,
        receive_multi_kv_cache_distributed=lambda *a, **k: None,
    )
    runner._kv_prefetch_enabled = False
    return runner


def _make_compile_runner(*, use_hsdp: bool):
    runner = object.__new__(DiffusionModelRunner)
    runner.pipeline = SimpleNamespace(transformer=SimpleNamespace())
    runner.od_config = SimpleNamespace(parallel_config=SimpleNamespace(use_hsdp=use_hsdp))
    return runner


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize("use_hsdp", [False, True])
def test_compile_transformer_regionally_compiles_blocks(monkeypatch, use_hsdp):
    runner = _make_compile_runner(use_hsdp=use_hsdp)
    compile_calls = []

    def _regionally_compile(model, *args, **kwargs):
        compile_calls.append((model, args, kwargs))
        return model

    monkeypatch.setattr(model_runner_module, "regionally_compile", _regionally_compile)

    DiffusionModelRunner._compile_transformer(runner, "transformer")

    assert compile_calls == [
        (
            runner.pipeline.transformer,
            (),
            {"dynamic": True},
        )
    ]


@pytest.mark.core_model
@pytest.mark.cpu
def test_execute_stepwise_streaming_returns_chunks_at_boundaries(monkeypatch):
    """Step streaming returns empty step results until a chunk decode boundary."""
    chunks = [
        DiffusionOutput(output="chunk-0", finished=False, chunk_index=0, total_chunks=2),
        DiffusionOutput(output="chunk-1", finished=True, chunk_index=1, total_chunks=2),
    ]
    runner = _make_runner(cache_backend=None, cache_backend_name=None)
    runner.pipeline = _ChunkStepPipeline(chunks)
    runner.od_config.streaming_output = True
    runner.od_config.step_execution = True
    req = _make_request(skip_cache_refresh=True)
    req.request_id = "req"

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "max_memory_allocated", lambda: 0)
    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        scheduled_new_reqs=[SimpleNamespace(request_id="req", req=req)],
        scheduled_cached_reqs=SimpleNamespace(request_ids=[]),
    )

    first = DiffusionModelRunner.execute_stepwise(runner, scheduler_output)
    assert first.get_request_output("req").result is None

    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(request_ids=["req"]),
    )
    second = DiffusionModelRunner.execute_stepwise(runner, scheduler_output)
    assert second.get_request_output("req").result == chunks[0]
    assert second.get_request_output("req").finished is False

    DiffusionModelRunner.execute_stepwise(runner, scheduler_output)
    fourth = DiffusionModelRunner.execute_stepwise(runner, scheduler_output)
    assert fourth.get_request_output("req").result == chunks[1]
    assert fourth.get_request_output("req").finished is True


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_execute_model_skips_cache_summary_without_active_cache_backend(monkeypatch):
    """Guard cache diagnostics with runtime backend state to avoid stale-config crashes."""
    runner = _make_runner(cache_backend=None, cache_backend_name="cache_dit")
    req = _make_request(skip_cache_refresh=True)

    cache_summary_calls = []

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(
        model_runner_module,
        "cache_summary",
        lambda pipeline, details: cache_summary_calls.append((pipeline, details)),
    )

    output = DiffusionModelRunner.execute_model(runner, req)

    assert output.output == "ok"
    assert cache_summary_calls == []


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_execute_model_emits_cache_summary_with_active_cache_dit_backend(monkeypatch):
    class _EnabledCacheBackend:
        def is_enabled(self):
            return True

    runner = _make_runner(cache_backend=_EnabledCacheBackend(), cache_backend_name="cache_dit")
    req = _make_request(skip_cache_refresh=True)

    cache_summary_calls = []

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(
        model_runner_module,
        "cache_summary",
        lambda pipeline, details: cache_summary_calls.append((pipeline, details)),
    )

    output = DiffusionModelRunner.execute_model(runner, req)

    assert output.output == "ok"
    assert cache_summary_calls == [(runner.pipeline, True)]


@pytest.mark.core_model
@pytest.mark.cpu
def test_execute_model_passes_single_request_batch_to_non_admission_pipeline(monkeypatch):
    runner = _make_runner(cache_backend=None, cache_backend_name="none")
    runner.pipeline = _SingleRequestBatchPipeline()
    req = _make_request(skip_cache_refresh=True)

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

    output = DiffusionModelRunner.execute_model(runner, req)

    assert output.output == "a prompt"
    assert isinstance(runner.pipeline.last_req, DiffusionRequestBatch)
    assert runner.pipeline.last_req.num_reqs == 1


@pytest.mark.core_model
@pytest.mark.cpu
def test_execute_model_accepts_bare_diffusion_output_from_single_request_pipeline(monkeypatch):
    runner = _make_runner(cache_backend=None, cache_backend_name="none")
    runner.pipeline = _SingleRequestDiffusionOutputPipeline()
    req = _make_request(skip_cache_refresh=True)

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

    output = DiffusionModelRunner.execute_model(runner, req)

    assert output.output == "a prompt"
    assert isinstance(runner.pipeline.last_req, DiffusionRequestBatch)
    assert runner.pipeline.last_req.num_reqs == 1


class _BatchPipeline:
    """Pipeline returning a configurable list of outputs from forward()."""

    supports_request_batch = True

    def __init__(self, outputs):
        self._outputs = outputs
        self.last_batch = None

    def forward(self, batch):
        self.last_batch = batch
        return list(self._outputs)


class _SingleRequestPipeline:
    def forward(self, batch):
        del batch
        return [DiffusionOutput(output="single")]


class _BatchSingleOutputPipeline:
    supports_request_batch = True

    def forward(self, batch):
        return DiffusionOutput(output=batch.prompts[0])


def _make_batch_runner(pipeline):
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    runner.pipeline = pipeline
    runner.cache_backend = None
    runner.offload_backend = None
    runner.od_config = SimpleNamespace(
        cache_backend="none",
        enable_cache_dit_summary=False,
        parallel_config=SimpleNamespace(use_hsdp=False),
    )
    runner.kv_transfer_manager = SimpleNamespace(
        receive_multi_kv_cache_distributed=lambda req, cfg_kv_collect_func=None, target_device=None: None,
    )
    return runner


def _make_scheduler_output(num_reqs: int):
    reqs = [_make_request() for _ in range(num_reqs)]
    for i, req in enumerate(reqs):
        req.request_id = f"req-{i}"
    return SimpleNamespace(scheduled_new_reqs=[SimpleNamespace(req=req) for req in reqs])


@pytest.mark.core_model
@pytest.mark.cpu
def test_execute_model_batch_rejects_output_count_mismatch(monkeypatch):
    """A pipeline returning the wrong number of outputs must fail loudly,
    not silently drop requests or IndexError on the per-request mapping."""
    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module, "current_omni_platform", _fake_platform_for_peak_memory())
    # forward returns 1 output for 2 scheduled requests
    runner = _make_batch_runner(_BatchPipeline(outputs=[DiffusionOutput(output="only-one")]))
    sched = _make_scheduler_output(num_reqs=2)

    with pytest.raises(RuntimeError, match="returned 1 outputs for 2 requests"):
        DiffusionModelRunner.execute_model_batch(runner, sched, runner.od_config)


@pytest.mark.core_model
@pytest.mark.cpu
def test_execute_model_batch_routes_one_output_per_request(monkeypatch):
    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module, "current_omni_platform", _fake_platform_for_peak_memory())
    outs = [DiffusionOutput(output="a"), DiffusionOutput(output="b")]
    runner = _make_batch_runner(_BatchPipeline(outputs=outs))
    sched = _make_scheduler_output(num_reqs=2)

    result = DiffusionModelRunner.execute_model_batch(runner, sched, runner.od_config)

    assert len(result.runner_outputs) == 2
    assert result.runner_outputs[0].request_id == "req-0"
    assert result.runner_outputs[0].result.output == "a"
    assert result.runner_outputs[1].request_id == "req-1"
    assert result.runner_outputs[1].result.output == "b"


@pytest.mark.core_model
@pytest.mark.cpu
def test_execute_model_batch_preserves_per_request_sampling_and_seeds_generators(monkeypatch):
    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module, "current_omni_platform", _fake_platform_for_peak_memory())
    outputs = [DiffusionOutput(output="a"), DiffusionOutput(output="b")]
    pipeline = _BatchPipeline(outputs=outputs)
    runner = _make_batch_runner(pipeline)
    sched = _make_scheduler_output(num_reqs=2)
    sched.scheduled_new_reqs[0].req.sampling_params.seed = 111
    sched.scheduled_new_reqs[1].req.sampling_params.seed = 222

    DiffusionModelRunner.execute_model_batch(runner, sched, runner.od_config)

    assert pipeline.last_batch is not None
    assert [sp.seed for sp in pipeline.last_batch.sampling_params_list] == [111, 222]
    assert [sp.generator.initial_seed() for sp in pipeline.last_batch.sampling_params_list] == [111, 222]
    with pytest.raises(AssertionError, match="multiple requests"):
        _ = pipeline.last_batch.sampling_params


@pytest.mark.core_model
@pytest.mark.cpu
def test_execute_model_batch_rejects_pipeline_without_request_batch_support(monkeypatch):
    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module, "current_omni_platform", _fake_platform_for_peak_memory())
    runner = _make_batch_runner(_SingleRequestPipeline())
    sched = _make_scheduler_output(num_reqs=2)

    with pytest.raises(RuntimeError, match="does not support request-batch forward"):
        DiffusionModelRunner.execute_model_batch(runner, sched, runner.od_config)


@pytest.mark.core_model
@pytest.mark.cpu
def test_execute_model_batch_rejects_single_diffusion_output(monkeypatch):
    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module, "current_omni_platform", _fake_platform_for_peak_memory())
    runner = _make_batch_runner(_BatchSingleOutputPipeline())
    sched = _make_scheduler_output(num_reqs=1)

    with pytest.raises(RuntimeError, match="request-batch forward must return list\\[DiffusionOutput\\]"):
        DiffusionModelRunner.execute_model_batch(runner, sched, runner.od_config)


@pytest.mark.core_model
@pytest.mark.cpu
def test_execute_model_batch_uses_runner_output_helper(monkeypatch):
    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module, "current_omni_platform", _fake_platform_for_peak_memory())
    outs = [DiffusionOutput(output="a"), DiffusionOutput(output="b")]
    runner = _make_batch_runner(_BatchPipeline(outputs=outs))
    sched = _make_scheduler_output(num_reqs=2)
    helper_calls = []
    original_from_outputs = DiffusionModelRunner._runner_output_from_outputs

    def _recording_from_outputs(self, reqs, outputs):
        helper_calls.append(([req.request_id for req in reqs], [output.output for output in outputs]))
        return original_from_outputs(self, reqs, outputs)

    monkeypatch.setattr(DiffusionModelRunner, "_runner_output_from_outputs", _recording_from_outputs)

    result = DiffusionModelRunner.execute_model_batch(runner, sched, runner.od_config)

    assert [runner_output.result.output for runner_output in result.runner_outputs] == ["a", "b"]
    assert helper_calls == [(["req-0", "req-1"], ["a", "b"])]


@pytest.mark.core_model
@pytest.mark.cpu
def test_split_diffusion_output_by_request_slices_single_and_multi_request_outputs():
    reqs = [_make_request(), _make_request()]
    reqs[0].request_id = "req-0"
    reqs[1].request_id = "req-1"
    batch = DiffusionRequestBatch(requests=reqs)
    result = DiffusionOutput(output=["img-0a", "img-0b", "img-1a", "img-1b"], stage_durations={"decode": 1.0})

    outputs = split_diffusion_output_by_request(result, batch, num_outputs_per_prompt=2)

    assert [output.output for output in outputs] == [["img-0a", "img-0b"], ["img-1a", "img-1b"]]
    assert [output.stage_durations for output in outputs] == [{"decode": 1.0}, {"decode": 1.0}]

    single = split_diffusion_output_by_request(
        result, DiffusionRequestBatch(requests=reqs[:1]), num_outputs_per_prompt=2
    )

    assert single[0].output == ["img-0a", "img-0b"]


@pytest.mark.core_model
@pytest.mark.cpu
def test_split_diffusion_output_by_request_slices_tuple_outputs():
    reqs = [_make_request(), _make_request()]
    batch = DiffusionRequestBatch(requests=reqs)
    result = DiffusionOutput(
        output=(
            ["video-0", "video-1"],
            torch.tensor([10, 20]),
        )
    )

    outputs = split_diffusion_output_by_request(result, batch, num_outputs_per_prompt=1)

    assert outputs[0].output[0] == ["video-0"]
    assert torch.equal(outputs[0].output[1], torch.tensor([10]))
    assert outputs[1].output[0] == ["video-1"]
    assert torch.equal(outputs[1].output[1], torch.tensor([20]))


@pytest.mark.core_model
@pytest.mark.cpu
def test_load_model_clears_cache_backend_for_unsupported_pipeline(monkeypatch):
    class _DummyLoader:
        def __init__(self, load_config, od_config=None):
            del load_config, od_config

        def load_model(self, **kwargs):
            del kwargs
            return SimpleNamespace(transformer=torch.nn.Identity())

    class _DummyMemoryProfiler:
        consumed_memory = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    class _DummyCacheBackend:
        def __init__(self):
            self.enabled = False

        def enable(self, pipeline):
            del pipeline
            self.enabled = True

    dummy_cache_backend = _DummyCacheBackend()

    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    runner.pipeline = None
    runner.cache_backend = None
    runner.offload_backend = None
    runner.od_config = SimpleNamespace(
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        cache_backend="cache_dit",
        cache_config={},
        model_class_name="NextStep11Pipeline",
        enforce_eager=True,
        streaming_output=False,
    )

    monkeypatch.setattr(model_runner_module, "LoadConfig", lambda: object())
    monkeypatch.setattr(model_runner_module, "DiffusersPipelineLoader", _DummyLoader)
    monkeypatch.setattr(model_runner_module, "DeviceMemoryProfiler", _DummyMemoryProfiler)
    monkeypatch.setattr(model_runner_module, "get_offload_backend", lambda od_config, device: None)
    monkeypatch.setattr(
        model_runner_module, "get_cache_backend", lambda cache_backend, cache_config: dummy_cache_backend
    )

    DiffusionModelRunner.load_model(runner)

    assert runner.cache_backend is None
    assert runner.od_config.cache_backend is None
    assert dummy_cache_backend.enabled is False


@pytest.mark.core_model
@pytest.mark.cpu
def test_set_forward_context_enters_vllm_config_contexts(monkeypatch):
    """Ensure `with set_forward_context(...):` enters vllm's context managers internally and calls desired vllm functions."""
    import vllm.config.vllm as vllm_config_module
    import vllm.ir
    from vllm.config import CompilationConfig, DeviceConfig, VllmConfig

    from vllm_omni.diffusion.forward_context import (
        get_forward_context,
        is_forward_context_available,
        set_forward_context,
    )

    vllm_config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(),
    )
    calls = []

    @contextmanager
    def _set_current_vllm_config(cfg):
        calls.append(("set_current_vllm_config", cfg))
        yield
        calls.append(("set_current_vllm_config_exit", cfg))

    @contextmanager
    def _set_priority(*args, **kwargs):
        del args, kwargs
        calls.append(("ir_op_priority", None))
        yield
        calls.append(("ir_op_priority_exit", None))

    @contextmanager
    def _enable_torch_wrap(flag):
        calls.append(("enable_torch_wrap", flag))
        yield
        calls.append(("enable_torch_wrap_exit", flag))

    monkeypatch.setattr(vllm_config_module, "set_current_vllm_config", _set_current_vllm_config)
    monkeypatch.setattr(vllm_config.kernel_config.ir_op_priority, "set_priority", _set_priority)
    monkeypatch.setattr(vllm.ir, "enable_torch_wrap", _enable_torch_wrap)

    assert not is_forward_context_available()

    with set_forward_context(vllm_config=vllm_config):
        assert is_forward_context_available()
        assert get_forward_context().vllm_config is vllm_config

    assert not is_forward_context_available()
    assert calls == [
        ("set_current_vllm_config", vllm_config),
        ("ir_op_priority", None),
        ("enable_torch_wrap", vllm_config.compilation_config.ir_enable_torch_wrap),
        ("enable_torch_wrap_exit", vllm_config.compilation_config.ir_enable_torch_wrap),
        ("ir_op_priority_exit", None),
        ("set_current_vllm_config_exit", vllm_config),
    ]


@pytest.mark.core_model
@pytest.mark.cpu
def test_vllm_set_forward_context_implementation(monkeypatch):
    """Regression test: ensure that vLLM's set_forward_context implementation has changed."""
    import vllm.forward_context as vllm_forward_context
    import vllm.ir
    from vllm.config import CompilationConfig, DeviceConfig, VllmConfig

    ERROR_MESSAGE = (
        "If this test fails, it likely means that vLLM's set_forward_context (vllm/forward_context.py) implementation has changed. "
        "In this case, we should update our forward_context (vllm_omni/diffusion/forward_context.py) as well. "
        "We should at least confirm that the `try: with (<what's inside?>): yield` part does not miss any information "
        "(typically by calling the same or similar stuff as vLLM). See #3352 for an example. "
        "Then, update this test to reflect the new implementation, and also update test_set_forward_context_enters_vllm_config_contexts."
    )

    vllm_config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(),
    )
    calls = []

    @contextmanager
    def _set_priority():
        calls.append(("ir_op_priority", None))
        yield
        calls.append(("ir_op_priority_exit", None))

    @contextmanager
    def _enable_torch_wrap(flag):
        calls.append(("enable_torch_wrap", flag))
        yield
        calls.append(("enable_torch_wrap_exit", flag))

    def _set_additional_forward_context(**kwargs):
        calls.append(("set_additional_forward_context", tuple(sorted(kwargs.keys()))))
        return {}

    monkeypatch.setattr(vllm_config.kernel_config.ir_op_priority, "set_priority", _set_priority)
    monkeypatch.setattr(vllm.ir, "enable_torch_wrap", _enable_torch_wrap)
    monkeypatch.setattr(
        vllm_forward_context.current_platform,
        "set_additional_forward_context",
        _set_additional_forward_context,
    )

    assert not vllm_forward_context.is_forward_context_available(), ERROR_MESSAGE

    with vllm_forward_context.set_forward_context(None, vllm_config):
        assert vllm_forward_context.is_forward_context_available(), ERROR_MESSAGE
        assert vllm_forward_context.get_forward_context().attn_metadata is None, ERROR_MESSAGE

    assert not vllm_forward_context.is_forward_context_available(), ERROR_MESSAGE
    assert calls == [
        (
            "set_additional_forward_context",
            (
                "attn_metadata",
                "batch_descriptor",
                "cudagraph_runtime_mode",
                "dp_metadata",
                "num_tokens",
                "num_tokens_across_dp",
                "ubatch_slices",
                "vllm_config",
            ),
        ),
    ], ERROR_MESSAGE
