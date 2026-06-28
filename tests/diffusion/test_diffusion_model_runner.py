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

pytestmark = [pytest.mark.diffusion]


@contextmanager
def _noop_forward_context(*args, **kwargs):
    del args, kwargs
    yield


class _DummyPipeline:
    def __init__(self, output):
        self._output = output
        self.forward_calls = 0

    def forward(self, req):
        del req
        self.forward_calls += 1
        return self._output


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
        prompts=["a prompt"],
        sampling_params=sampling_params,
        skip_cache_refresh=skip_cache_refresh,
    )


def _make_runner(cache_backend, cache_backend_name: str, enable_cache_dit_summary: bool = True):
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    runner.pipeline = _DummyPipeline(output=SimpleNamespace(output="ok"))
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
