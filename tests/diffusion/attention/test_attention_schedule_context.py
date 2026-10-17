# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Contracts for request schedule binding, context restore, and layer selection."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm_omni.diffusion.attention.layer as layer_mod
import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
from tests.diffusion.attention.test_attention_schedule_candidates import (
    _fake_resolve,
    _FakeImpl,
    _make_config,
)
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.attention.schedule import (
    AttentionScheduleRange,
)
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import (
    AttentionConfig,
    AttentionScheduleConfig,
    AttentionSpec,
    DiffusionOutput,
    DiffusionRequestAbortedError,
    OmniDiffusionConfig,
)
from vllm_omni.diffusion.forward_context import (
    DenoiseProgressMixin,
    begin_scheduled_denoise,
    get_forward_context,
    is_forward_context_available,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput, NewRequestData
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def attention_env(monkeypatch):
    monkeypatch.setattr(layer_mod.SDPABackend, "get_impl_cls", staticmethod(lambda: _FakeImpl))
    monkeypatch.setattr(layer_mod, "build_parallel_attention_strategy", lambda **kwargs: NoParallelAttention())
    monkeypatch.setattr(layer_mod, "get_attn_backend_for_role", lambda **kwargs: _fake_resolve(**kwargs))

    def build(config, **kwargs):
        with set_current_diffusion_config(config):
            return Attention(num_heads=4, head_size=64, causal=False, softmax_scale=1.0, **kwargs)

    return SimpleNamespace(build=build)


def _service():
    return AttentionScheduleConfig(
        profiles={
            "sparse": AttentionConfig(default=AttentionSpec(backend="SDPA")),
            "dense": AttentionConfig(default=AttentionSpec(backend="FLASH_ATTN")),
        },
        default=[{"start": 0, "end": 2, "profile": "sparse"}],
    )


@pytest.fixture
def runner_platform(monkeypatch):
    platform = model_runner_module.current_omni_platform
    monkeypatch.setattr(platform, "is_available", lambda: False)
    monkeypatch.setattr(platform, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(platform, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(platform, "max_memory_allocated", lambda: 0)


def _runner_config(schedule):
    config = _make_config(schedule=schedule)
    config.parallel_config.use_hsdp = False
    config.cache_backend = None
    config.enable_cache_dit_summary = False
    config.streaming_output = False
    return config


def _make_runner(pipeline, config):
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = None
    runner.od_config = config
    runner.device = torch.device("cpu")
    runner.pipeline = pipeline
    runner.cache_backend = None
    runner.offload_backend = None
    runner.state_cache = {}
    runner.kv_transfer_manager = None
    return runner


def _request(request_id, schedule=None, steps=4):
    return OmniDiffusionRequest(
        prompt=f"prompt-{request_id}",
        request_id=request_id,
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=steps, attention_schedule=schedule),
    )


def _new_requests(*requests, finished=()):
    return DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=[NewRequestData(request_id=request.request_id, req=request) for request in requests],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(finished),
        num_running_reqs=len(requests),
        num_waiting_reqs=0,
    )


def _names(layer, selected):
    """Replace each recorded implementation with its profile name, or "base"."""
    names = {id(layer.attention): "base"}
    names.update({id(record.impl): name for name, record in layer._schedule_candidates.items()})
    return [(*entry[:-1], names[id(entry[-1])]) for entry in selected]


class _PublishingRequestPipeline(DenoiseProgressMixin):
    """Request mode: encode, publish every denoise step, then decode."""

    supports_request_batch = True

    def __init__(self, layer, *, fail_step=None, error=None, actual_steps=None):
        self.layer = layer
        self.actual_steps = actual_steps
        self.fail_step = fail_step
        self.error = error
        self.selected: list[tuple[Any, ...]] = []
        self.bound_schedules: list[Any] = []

    def _select(self, phase):
        self.selected.append((phase, self.layer.effective_attention()[0]))

    def forward(self, batch):
        self.bound_schedules.append(get_forward_context().attention_schedule)
        total = self.actual_steps
        if total is None:
            total = batch.requests[0].sampling_params.num_inference_steps
        self._select("encode")
        begin_scheduled_denoise(total)
        for step in range(total):
            self.record_denoise_step(step, total_steps=total)
            if step == self.fail_step:
                raise self.error
            self._select(step)
        self.record_denoise_step(None)
        self._select("decode")
        return [DiffusionOutput(output={"request_id": request.request_id}) for request in batch.requests]


@pytest.mark.parametrize(
    ("schedule", "bound", "expected"),
    [
        (
            [{"start": 1, "end": 3, "profile": "dense"}],
            (AttentionScheduleRange(start=1, end=3, profile="dense"),),
            ["base", "base", "dense", "dense", "base", "base"],
        ),
        ([], None, ["base"] * 6),
    ],
    ids=["replace", "disable"],
)
def test_request_runner_override_replaces_or_disables_default(
    attention_env, runner_platform, schedule, bound, expected
):
    config = _runner_config(_service())
    layer = attention_env.build(config)
    pipeline = _PublishingRequestPipeline(layer)
    runner = _make_runner(pipeline, config)

    DiffusionModelRunner.execute_model(runner, _request("req-a", schedule=schedule))

    assert pipeline.bound_schedules == [bound]
    assert [name for _phase, name in _names(layer, pipeline.selected)] == expected


@pytest.mark.parametrize(
    "error",
    [RuntimeError("denoise failed"), DiffusionRequestAbortedError("Request req-a aborted.")],
    ids=["exception", "cancel"],
)
def test_request_runner_failure_or_cancel_does_not_leak_into_next_request(attention_env, runner_platform, error):
    config = _runner_config(_service())
    layer = attention_env.build(config)
    failing = _PublishingRequestPipeline(layer, fail_step=1, error=error)
    runner = _make_runner(failing, config)
    request = _request("req-a", schedule=[{"start": 0, "end": 4, "profile": "dense"}])

    if isinstance(error, DiffusionRequestAbortedError):
        assert DiffusionModelRunner.execute_model(runner, request).aborted is True
    else:
        with pytest.raises(RuntimeError, match="denoise failed"):
            DiffusionModelRunner.execute_model(runner, request)
    assert _names(layer, failing.selected) == [("encode", "base"), (0, "dense")]
    assert is_forward_context_available() is False

    healthy = _PublishingRequestPipeline(layer)
    runner.pipeline = healthy
    DiffusionModelRunner.execute_model(runner, _request("req-b"))
    assert [name for _phase, name in _names(layer, healthy.selected)] == [
        "base",
        "sparse",
        "sparse",
        "base",
        "base",
        "base",
    ]


@pytest.mark.parametrize("batch_size", [1, 2])
def test_request_runner_returns_400_for_actual_schedule_bounds(attention_env, runner_platform, batch_size):
    config = _runner_config(_service())
    layer = attention_env.build(config)
    pipeline = _PublishingRequestPipeline(layer, actual_steps=4)
    runner = _make_runner(pipeline, config)
    requests = [
        _request(f"req-{i}", schedule=[{"start": 0, "end": 6, "profile": "dense"}], steps=8) for i in range(batch_size)
    ]

    if batch_size == 1:
        outputs = [runner.execute_model(requests[0])]
    else:
        result = runner.execute_model_batch(_new_requests(*requests), config)
        assert [item.request_id for item in result.runner_outputs] == [req.request_id for req in requests]
        outputs = [item.result for item in result.runner_outputs]

    for output in outputs:
        assert "exceeds total_steps" in output.error
        assert output.error_status_code == 400
        assert output.error_type == "invalid_attention_schedule"
    assert len(pipeline.bound_schedules) == 1
    assert len(pipeline.selected) == 1
    assert pipeline.selected[0][0] == "encode"
    assert is_forward_context_available() is False

    runner.pipeline = _PublishingRequestPipeline(layer)
    assert runner.execute_model(_request("healthy", schedule=[])).error is None
    assert is_forward_context_available() is False


class _StartupPublishingPipeline(DenoiseProgressMixin):
    """A loaded pipeline that publishes denoise progress."""


class _StartupSilentPipeline:
    """A loaded pipeline without a progress publisher."""


class _StartupCacheBackend:
    def enable(self, pipeline):
        del pipeline


class _StartupMemoryProfiler:
    consumed_memory = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


def _load_model(monkeypatch, pipeline, *, schedule, cache_backend=None, model_class_name="WanPipeline"):
    """Run the real load_model with the loader, memory profiler, offload and cache factory replaced."""

    class _Loader:
        def __init__(self, load_config, od_config=None):
            del load_config, od_config

        def load_model(self, **kwargs):
            del kwargs
            return pipeline

    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    runner.pipeline = None
    runner.cache_backend = None
    runner.offload_backend = None
    runner.od_config = OmniDiffusionConfig(
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        cache_backend=cache_backend,
        cache_config={},
        model_class_name=model_class_name,
        enforce_eager=True,
        streaming_output=False,
        diffusion_attention_schedule=schedule,
    )
    cache = _StartupCacheBackend() if cache_backend not in (None, "none") else None
    monkeypatch.setattr(model_runner_module, "LoadConfig", lambda: object())
    monkeypatch.setattr(model_runner_module, "DiffusersPipelineLoader", _Loader)
    monkeypatch.setattr(model_runner_module, "DeviceMemoryProfiler", _StartupMemoryProfiler)
    monkeypatch.setattr(model_runner_module, "enable_offload_backend", lambda od_config, pipe, device: (pipe, None))
    monkeypatch.setattr(model_runner_module, "get_cache_backend", lambda name, cache_config: cache)
    DiffusionModelRunner.load_model(runner)
    return runner


def _profiles_only_service():
    return AttentionScheduleConfig(profiles=_service().profiles, default=[])


def test_startup_rejects_profiles_on_pipeline_without_progress_publisher(monkeypatch):
    with pytest.raises(ValueError, match="_StartupSilentPipeline never publishes denoise progress"):
        _load_model(monkeypatch, _StartupSilentPipeline(), schedule=_profiles_only_service())


def test_startup_rejects_profiles_with_cache_backend_when_default_is_empty(monkeypatch):
    with pytest.raises(ValueError, match="cache_backend='tea_cache'"):
        _load_model(
            monkeypatch, _StartupPublishingPipeline(), schedule=_profiles_only_service(), cache_backend="tea_cache"
        )
