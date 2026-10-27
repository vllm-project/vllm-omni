# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU execution of actual integration definitions with external dependencies injected."""

import copy
import dataclasses
import importlib.util
import json
import logging
import os
import sys
import types
from abc import ABC, abstractmethod
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.diffusion.cache.aligncache_helpers import ROOT, definitions
from vllm_omni.diffusion.cache.aligncache.config import AlignCacheConfig, normalize_cache_config
from vllm_omni.diffusion.cache.aligncache.multicomponent import WanRuntime
from vllm_omni.errors import OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def shared_config(monkeypatch):
    env = definitions(
        "vllm_omni/diffusion/data.py",
        {"DiffusionCacheConfig"},
        {"dataclass": dataclasses.dataclass, "field": dataclasses.field, "fields": dataclasses.fields, "Any": Any},
    )
    cls = env["DiffusionCacheConfig"]
    module = types.ModuleType("vllm_omni.diffusion.data")
    module.__dict__["DiffusionCacheConfig"] = cls
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return cls


@pytest.fixture
def backend_module(monkeypatch, shared_config):
    base = definitions(
        "vllm_omni/diffusion/cache/base.py",
        {"CacheBackend"},
        {"ABC": ABC, "abstractmethod": abstractmethod, "nullcontext": nullcontext},
    )
    module = types.ModuleType("vllm_omni.diffusion.cache.base")
    module.__dict__["CacheBackend"] = base["CacheBackend"]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    name = "vllm_omni.diffusion.cache.aligncache._backend_contract_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "vllm_omni/diffusion/cache/aligncache/backend.py")
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def execution_config():
    return SimpleNamespace(enforce_eager=True, dtype=torch.bfloat16, parallel_config=SimpleNamespace(use_hsdp=False))


def request(**sampling):
    return SimpleNamespace(
        prompt="prompt",
        request_id="aligncache-test",
        kv_recv_ms=0,
        sampling_params=SimpleNamespace(
            num_outputs_per_prompt=1,
            lora_request=None,
            num_inference_steps=None,
            timesteps=None,
            sigmas=None,
            generator=object(),
            **sampling,
        ),
        skip_cache_refresh=False,
    )


def test_shared_config_roundtrip_explicit_foreign(shared_config):
    a, b = shared_config(), shared_config()
    a.align_cache["mode"] = "trace"
    assert not b.align_cache
    source = shared_config.from_dict({"align_cache": {"warmup_steps": 3}})
    for value in (source, copy.deepcopy(source), shared_config.from_dict(dataclasses.asdict(source))):
        assert AlignCacheConfig.parse(value).warmup_steps == 3
    for value in (
        shared_config.from_dict({"rel_l1_thresh": 0.2}),
        shared_config.from_dict({"unknown": 1}),
        shared_config(rel_l1_thresh=0.3),
    ):
        with pytest.raises(ValueError):
            AlignCacheConfig.parse(value)


def test_actual_diffusion_ingress_normalization(shared_config):
    import warnings

    normalize = definitions(
        "vllm_omni/diffusion/data.py",
        {"normalize_omni_diffusion_kwargs", "_move_diffusion_alias"},
        {"os": os, "torch": torch, "json": json, "warnings": warnings, "logger": logging.getLogger(__name__)},
    )["normalize_omni_diffusion_kwargs"]
    for value in (None, {"align_cache": {"mode": "trace"}}, '{"align_cache":{"mode":"trace"}}'):
        result = normalize({"cache_backend": "align_cache", "cache_config": value})["cache_config"]
        final_worker = shared_config.from_dict(result)
        assert AlignCacheConfig.parse(final_worker).mode == ("accelerate" if value is None else "trace")
    for value in ("{", "[]", "true", '"nested"'):
        with pytest.raises(ValueError):
            normalize({"cache_backend": "align_cache", "cache_config": value})
    assert "cache_config" not in normalize({"cache_backend": "tea_cache", "cache_config": "{"})


def test_install_refresh_disable_ownership(backend_module, monkeypatch):
    monkeypatch.setattr(backend_module, "validate_pipeline", lambda p: None)
    pipeline = SimpleNamespace(transformer=object(), od_config=execution_config())
    backend = backend_module.AlignCacheBackend()
    backend.enable(pipeline)
    installed = pipeline._aligncache_runtime
    backend.enable(pipeline)
    assert installed is pipeline._aligncache_runtime
    with backend.request_context(pipeline, request()):
        before = installed.components["transformer"].state
        with pytest.raises(RuntimeError, match="active"):
            backend.refresh(pipeline, 0)
        assert installed.components["transformer"].state is before
    for _ in range(2):
        backend.refresh(pipeline, 40)
        with backend.request_context(pipeline, request()):
            assert installed.components["transformer"].state.completed_pairs == 0
    backend.disable(pipeline)
    backend.disable(pipeline)
    assert not hasattr(pipeline, "_aligncache_runtime")
    backend.enable(pipeline)
    assert pipeline._aligncache_runtime is not installed
    pipeline.transformer = object()
    with pytest.raises(RuntimeError, match="ownership"):
        backend.enable(pipeline)
    foreign = object()
    pipeline._aligncache_runtime = foreign
    backend.disable(pipeline)
    assert pipeline._aligncache_runtime is foreign


def test_install_failure_rolls_back(backend_module, monkeypatch):
    monkeypatch.setattr(backend_module, "validate_pipeline", lambda p: None)

    class Pipeline:
        transformer = object()
        od_config = execution_config()

        def __setattr__(self, key, value):
            object.__setattr__(self, key, value)
            if key == "_aligncache_runtime":
                raise RuntimeError("injected installation failure")

    pipeline = Pipeline()
    backend = backend_module.AlignCacheBackend()
    with pytest.raises(RuntimeError, match="injected"):
        backend.enable(pipeline)
    assert not backend.is_enabled() and not hasattr(pipeline, "_aligncache_runtime")


@pytest.mark.parametrize(
    "field,value",
    [
        ("enforce_eager", False),
        ("enable_cpu_offload", True),
        ("enable_layerwise_offload", True),
        ("step_execution", True),
        ("dtype", torch.float16),
        ("lora_path", "weights"),
        ("quantization_config", "fp8"),
        ("enable_sleep_mode", True),
        ("num_gpus", 2),
        ("diffusion_offload_config", {"dit": "layerwise"}),
        ("enable_distributed_layerwise_offload", True),
        ("streaming_output", True),
        ("stage_role", "denoise"),
        ("diffusion_kv_mode", "paged_scheduler"),
        ("kv_offload_config", {"mode": "on"}),
    ],
)
@pytest.mark.parametrize("policy", ["wan21_feedback_v1", "wan22_feedback_v1", "qwen_flow_v1", "minimax_h3_flow_v1"])
def test_static_rejections(backend_module, field, value, policy):
    config = execution_config()
    setattr(config, field, value)
    with pytest.raises(ValueError, match=field):
        backend_module.validate_execution_config(config, policy=policy)


@pytest.mark.parametrize(
    "profile", ["wan21_reference_v1", "wan22_t2v_reference_v1", "qwen21_reference_v1", "minimax_h3_reference_v1"]
)
def test_kv_offload_rejected_before_installation_and_at_request_entry(backend_module, monkeypatch, profile):
    backend = backend_module.AlignCacheBackend({"align_cache": {"profile": profile}})
    pipeline = SimpleNamespace(transformer=object(), transformer_2=object(), od_config=execution_config())
    monkeypatch.setattr(backend, "_validate_pipeline", lambda p: None)
    pipeline.od_config.kv_offload_config = {"mode": "on"}
    with pytest.raises(ValueError, match="align_cache.kv_offload_config"):
        backend.enable(pipeline)
    assert not backend.enabled and not hasattr(pipeline, "_aligncache_runtime")
    pipeline.od_config.kv_offload_config = None
    backend.enable(pipeline)
    pipeline.od_config.kv_offload_config = {"mode": "on"}
    with pytest.raises(ValueError, match="align_cache.kv_offload_config"):
        with backend.request_context(pipeline, request()):
            pytest.fail("KV offload must not enter the AlignCache request")
    assert not backend.runtime.active
    pipeline.od_config.kv_offload_config = None
    with backend.request_context(pipeline, request()):
        assert backend.runtime.active


def fake_quant_config(name="fp8", checkpoint=False, components=None):
    config = SimpleNamespace(get_name=lambda: name, is_checkpoint_fp8_serialized=checkpoint)
    if components is not None:
        config.component_configs, config.default_config = components, None
    return config


def test_quantization_requires_explicit_fp8_opt_in(backend_module):
    config = execution_config()
    config.quantization_config = fake_quant_config()
    with pytest.raises(ValueError, match="quantization_config"):
        backend_module.validate_execution_config(config)
    backend_module.validate_execution_config(config, "fp8")
    components = {"transformer": fake_quant_config(), "text_encoder": None}
    config.quantization_config = fake_quant_config("component", components=components)
    backend_module.validate_execution_config(config, "fp8")
    for rejected in (
        fake_quant_config("int8"),
        fake_quant_config(checkpoint=True),
        fake_quant_config("component", components={"transformer": fake_quant_config("gguf")}),
    ):
        config.quantization_config = rejected
        with pytest.raises(ValueError, match="quantization_config"):
            backend_module.validate_execution_config(config, "fp8")
    config.quantization_config = None
    with pytest.raises(ValueError, match="requires the engine's online FP8"):
        backend_module.validate_execution_config(config, "fp8")


def test_backend_threads_quantization_opt_in_to_admission(backend_module, monkeypatch):
    cache_config = {"align_cache": {"profile": "qwen21_reference_v1", "quantization": "fp8"}}
    pipeline = SimpleNamespace(transformer=object(), od_config=execution_config())
    backend = backend_module.AlignCacheBackend(cache_config)
    monkeypatch.setattr(backend, "_validate_pipeline", lambda p: None)
    with pytest.raises(ValueError, match="requires the engine's online FP8"):
        backend.enable(pipeline)
    pipeline.od_config.quantization_config = fake_quant_config()
    backend.enable(pipeline)
    pipeline.od_config.quantization_config = fake_quant_config("int8")
    with pytest.raises(ValueError, match="quantization_config"):
        with backend.request_context(pipeline, request()):
            pytest.fail("invalid request entered")
    pipeline.od_config.quantization_config = fake_quant_config()
    with backend.request_context(pipeline, request()):
        assert backend.runtime.active
    default_backend = backend_module.AlignCacheBackend()
    with pytest.raises(ValueError, match="quantization_config"):
        default_backend.enable(SimpleNamespace(transformer=object(), od_config=pipeline.od_config))


@pytest.mark.parametrize(
    "field",
    [
        "pipeline_parallel_size",
        "tensor_parallel_size",
        "sequence_parallel_size",
        "data_parallel_size",
        "cfg_parallel_size",
        "ring_degree",
        "ulysses_degree",
        "vae_patch_parallel_size",
        "enable_expert_parallel",
        "use_hsdp",
    ],
)
def test_parallel_rejections(backend_module, field):
    config = execution_config()
    setattr(config.parallel_config, field, 2 if field != "use_hsdp" else True)
    with pytest.raises(ValueError, match=field):
        backend_module.validate_execution_config(config)


def test_actual_runner_scope_default_steps_and_failure(backend_module, monkeypatch):
    monkeypatch.setattr(backend_module, "validate_pipeline", lambda p: None)
    config = execution_config()
    config.cache_backend = "align_cache"
    config.enable_cache_dit_summary = False

    class Pipeline:
        _aligncache_runtime: WanRuntime
        transformer = object()
        od_config = config
        device = torch.device("cpu")

        def forward(self, req):
            assert self._aligncache_runtime.active
            return SimpleNamespace(peak_memory_mb=0)

    pipeline = Pipeline()
    backend = backend_module.AlignCacheBackend()
    backend.enable(pipeline)
    cls = definitions(
        "vllm_omni/diffusion/worker/diffusion_model_runner.py",
        {"DiffusionModelRunner"},
        {
            "torch": torch,
            "nullcontext": nullcontext,
            "set_forward_context": lambda **kw: nullcontext(),
            "record_function": lambda *a: nullcontext(),
            "logger": logging.getLogger(__name__),
            "current_omni_platform": SimpleNamespace(reset_peak_memory_stats=lambda: None),
            "resolve_offload_strategy": lambda c: "none",
            "OffloadStrategy": SimpleNamespace(DISTRIBUTED_LAYER_WISE="distributed"),
            "DiffusionRequestBatch": lambda requests: SimpleNamespace(requests=requests),
            "request_cancellation_scope": lambda *a, **kw: nullcontext(),
            "check_request_cancellation": lambda: None,
            "DiffusionRequestAbortedError": type("AbortError", (Exception,), {}),
            "_normalize_pipeline_outputs": lambda raw, **kwargs: [raw],
        },
        {"_execute_request_list", "_refresh_cache_for_requests"},
    )["DiffusionModelRunner"]
    runner = cls()
    runner.pipeline, runner.cache_backend, runner.od_config = pipeline, backend, config
    runner.device, runner.vllm_config, runner.prompt_embed_cache = torch.device("cpu"), None, None
    runner.kv_transfer_manager = SimpleNamespace(receive_multi_kv_cache_distributed=lambda *a, **kw: None)
    runner._prepare_request_for_forward = lambda *a, **kw: None
    runner._prepare_output_for_transport = lambda output, sampling: output
    runner._sample_peak_memory_mb = lambda: 0
    runner._maybe_send_stage_payload = lambda *args: None
    runner._runner_output_from_outputs = lambda reqs, outputs: outputs

    def execute(req):
        return runner._execute_request_list(
            [req], od_config=config, allow_single_output=True, require_request_batch_support=False, record_name="test"
        )

    for _ in range(2):
        execute(request())
        assert backend.runtime.components["transformer"].state is None
    error = RuntimeError("postprocessing error")

    def fail(*args):
        raise error

    runner._prepare_output_for_transport = fail
    with pytest.raises(RuntimeError) as caught:
        execute(request())
    assert caught.value is error and backend.runtime.components["transformer"].state is None
    with pytest.raises(ValueError, match="one request"):
        runner._refresh_cache_for_requests([request(), request()], od_config=config)
    req = request()
    req.skip_cache_refresh = True
    with pytest.raises(ValueError, match="skip_cache_refresh"):
        execute(req)


def test_stage_schema_cli_override_reaches_worker(shared_config):
    import re
    import warnings
    from enum import Enum

    parallel = definitions(
        "vllm_omni/diffusion/data.py",
        {"DiffusionParallelConfig"},
        {
            "dataclass": dataclasses.dataclass,
            "field": dataclasses.field,
            "fields": dataclasses.fields,
            "config": lambda cls: cls,
            "model_validator": lambda **kw: lambda f: f,
        },
    )["DiffusionParallelConfig"]
    sys.modules["vllm_omni.diffusion.data"].__dict__["DiffusionParallelConfig"] = parallel
    cfg = {"align_cache": {"mode": "trace", "predictor": {"beta": {"mid": 0.2}}}}
    env = definitions(
        "vllm_omni/config/stage_config.py",
        {
            "StageType",
            "StageConfig",
            "build_stage_runtime_overrides",
            "_apply_diffusion_parallel_runtime_overrides",
            "reconcile_diffusion_attention_overrides",
            "_get_recursively_merged_dict",
        },
        {
            "Enum": Enum,
            "dataclass": dataclasses.dataclass,
            "field": dataclasses.field,
            "fields": dataclasses.fields,
            "warnings": warnings,
            "create_config": lambda x: x,
            "_STAGE_OVERRIDE_PATTERN": re.compile(r"^stage_(\d+)_(.+)$"),
        },
    )
    override = env["build_stage_runtime_overrides"](
        0,
        {
            "cache_backend": "align_cache",
            "cache_config": cfg,
            "stage_0_cache_config": {"align_cache": {"mode": "force_exact"}},
        },
        internal_keys=set(),
    )
    actual_stage = env["StageConfig"](
        0,
        "diffusion",
        stage_type=env["StageType"].DIFFUSION,
        yaml_engine_args={"cache_backend": "align_cache", "cache_config": cfg},
        runtime_overrides=override,
    )
    worker_args = actual_stage.to_omegaconf()["engine_args"]
    worker = shared_config.from_dict(normalize_cache_config(worker_args["cache_config"]))
    assert AlignCacheConfig.parse(worker).mode == "force_exact"
    assert AlignCacheConfig.parse(worker).predictor.beta.mid == 0.2


def test_actual_environment_fallback(monkeypatch):
    import warnings

    normalize = definitions(
        "vllm_omni/diffusion/data.py",
        {"normalize_omni_diffusion_kwargs", "_move_diffusion_alias"},
        {"os": os, "torch": torch, "json": json, "warnings": warnings, "logger": logging.getLogger(__name__)},
    )["normalize_omni_diffusion_kwargs"]
    monkeypatch.setenv("DIFFUSION_CACHE_BACKEND", "ALIGN_CACHE")
    assert normalize({})["cache_backend"] == "align_cache"
    assert normalize({"cache_backend": "none"})["cache_backend"] == "none"
    monkeypatch.delenv("DIFFUSION_CACHE_BACKEND")
    monkeypatch.setenv("DIFFUSION_CACHE_ADAPTER", "align_cache")
    assert normalize({})["cache_backend"] == "align_cache"
    with pytest.raises(ValueError, match="JSON"):
        normalize({"cache_backend": "align_cache", "cache_config": "{"})


def test_selector_none_existing_and_align(backend_module, shared_config, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.cache.aligncache.backend", backend_module)
    selector = definitions(
        "vllm_omni/diffusion/cache/selector.py",
        {"get_cache_backend"},
        {
            "DiffusionCacheConfig": shared_config,
            "CacheDiTBackend": lambda c: ("cache_dit", c),
            "TeaCacheBackend": lambda c: ("tea_cache", c),
            "MagCacheBackend": lambda c: ("mag_cache", c),
        },
    )["get_cache_backend"]
    for name in (None, "none"):
        assert selector(name, None) is None
    for name in ("cache_dit", "tea_cache", "mag_cache"):
        assert selector(name, {})[0] == name
    values: tuple[object, ...] = (None, {}, shared_config())
    for value in values:
        assert isinstance(selector("align_cache", value), backend_module.AlignCacheBackend)
    with pytest.raises(ValueError, match="align_cache"):
        selector("unknown", None)


def test_wan21_adapter_structural_allowlist(monkeypatch):
    from vllm_omni.diffusion.cache.aligncache.adapters.wan21 import validate_pipeline

    class Config(dict):
        __getattr__ = dict.__getitem__

    class Scheduler:
        config = Config(
            prediction_type="flow_prediction",
            num_train_timesteps=1000,
            solver_order=2,
            solver_type="bh2",
            predict_x0=True,
            use_dynamic_shifting=False,
            thresholding=False,
            lower_order_final=True,
            timestep_spacing="linspace",
            steps_offset=0,
            final_sigmas_type="zero",
            shift=3.0,
        )
        solver_p = None
        disable_corrector = []

    class Transformer:
        _forward_hooks: dict[int, object] = {}

        # Native transformer uses attributes, not a dict/FrozenDict.
        config = SimpleNamespace(
            in_channels=16,
            out_channels=16,
            num_layers=30,
            num_attention_heads=12,
            attention_head_dim=128,
            text_dim=4096,
            ffn_dim=8960,
            freq_dim=256,
            image_dim=None,
            added_kv_proj_dim=None,
            patch_size=(1, 2, 2),
        )

        def modules(self):
            return iter([self])

        def parameters(self):
            return iter([SimpleNamespace(dtype=torch.bfloat16, device=torch.device("cuda:0"))])

    class Pipeline:
        transformer = Transformer()
        transformer_2 = None
        has_transformer_2 = False
        expand_timesteps = False
        is_dmd = False
        scheduler = Scheduler()

    for name, attr, value in (
        ("vllm_omni.diffusion.models.schedulers", "FlowUniPCMultistepScheduler", Scheduler),
        ("vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2", "Wan22Pipeline", Pipeline),
        ("vllm_omni.diffusion.models.wan2_2.wan2_2_transformer", "WanTransformer3DModel", Transformer),
    ):
        module = types.ModuleType(name)
        setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, name, module)
    pipeline = Pipeline()
    validate_pipeline(pipeline)

    class I2V(Pipeline):
        pass

    with pytest.raises(ValueError, match="native WanPipeline"):
        validate_pipeline(I2V())
    pipeline.expand_timesteps = True
    with pytest.raises(ValueError, match="native WanPipeline"):
        validate_pipeline(pipeline)
    pipeline.expand_timesteps = False
    pipeline.transformer_2 = Transformer()
    with pytest.raises(ValueError, match="one native WanTransformer3DModel"):
        validate_pipeline(pipeline)
    pipeline.transformer_2 = None
    pipeline.transformer.config.in_channels = 36
    with pytest.raises(ValueError, match="in_channels"):
        validate_pipeline(pipeline)
    pipeline.transformer.config.in_channels = 16
    pipeline.transformer._forward_hooks = {1: lambda x: x}
    with pytest.raises(ValueError, match="hooks"):
        validate_pipeline(pipeline)
    pipeline.transformer._forward_hooks = {}
    pipeline.scheduler.config["solver_order"] = 3
    with pytest.raises(ValueError, match="solver_order"):
        validate_pipeline(pipeline)


def test_worker_rejects_lora_before_activation():
    cls = definitions(
        "vllm_omni/diffusion/worker/diffusion_worker.py",
        {"DiffusionWorker"},
        {"OmniClientError": OmniClientError},
        {"execute_model"},
    )["DiffusionWorker"]
    worker = cls()
    worker.model_runner = object()
    worker.lora_manager = SimpleNamespace(set_request_adapter=lambda *args: pytest.fail("weights must not change"))
    req = request()
    req.sampling_params.lora_request = object()
    with pytest.raises(ValueError, match="LoRA"):
        worker.execute_model(req, SimpleNamespace(cache_backend="align_cache"))


def test_active_weight_update_rejected_before_loading():
    cls = definitions(
        "vllm_omni/diffusion/worker/diffusion_model_runner.py", {"DiffusionModelRunner"}, {}, {"load_weights"}
    )["DiffusionModelRunner"]
    runner = cls()
    runner.pipeline = SimpleNamespace(
        _aligncache_runtime=SimpleNamespace(active=True),
        load_weights=lambda weights: pytest.fail("weights must not change"),
    )
    with pytest.raises(RuntimeError, match="active"):
        runner.load_weights([])


def test_backend_invalid_request_and_default_context(backend_module, monkeypatch):
    monkeypatch.setattr(backend_module, "validate_pipeline", lambda p: None)
    pipeline = SimpleNamespace(transformer=object(), od_config=execution_config())
    backend = backend_module.AlignCacheBackend()
    backend.enable(pipeline)
    for field, value in (("num_outputs_per_prompt", 2), ("lora_request", object())):
        req = request()
        setattr(req.sampling_params, field, value)
        with pytest.raises(ValueError):
            with backend.request_context(pipeline, req):
                pytest.fail("invalid scope entered")
        assert backend.runtime.components["transformer"].state is None
    req = request()
    req.prompt = None
    with pytest.raises(ValueError):
        with backend.request_context(pipeline, req):
            pass


@pytest.mark.parametrize("invalid", [{"enforce_eager": False}, {"kv_offload_config": {"mode": "on"}}])
def test_precheck_precedes_loading_offload_compile(backend_module, monkeypatch, invalid):
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.cache.aligncache.backend", backend_module)
    cls = definitions(
        "vllm_omni/diffusion/worker/diffusion_model_runner.py", {"DiffusionModelRunner"}, {}, {"load_model"}
    )["DiffusionModelRunner"]
    runner = cls()
    runner.od_config = execution_config()
    runner.od_config.cache_backend = "align_cache"
    runner.od_config.cache_config = None
    for name, value in invalid.items():
        setattr(runner.od_config, name, value)
    # No loader/offload/compile symbols supplied: attempting those would fail the test.
    with pytest.raises(ValueError, match=next(iter(invalid))):
        runner.load_model()


def test_wan21_schedule_actual_length_and_dynamic_admission():
    from vllm_omni.diffusion.cache.aligncache.multicomponent import WanRuntime

    runtime = WanRuntime(AlignCacheConfig())
    timesteps = torch.tensor([999.0, 600.0, 300.0])
    pipeline = SimpleNamespace(
        transformer=object(),
        transformer_2=None,
        expand_timesteps=False,
        scheduler=SimpleNamespace(
            timesteps=timesteps,
            sigmas=torch.ones(4),
            num_inference_steps=3,
            config={"prediction_type": "flow_prediction"},
        ),
    )
    latent = SimpleNamespace(ndim=5, shape=(1, 16, 1, 2, 2), device=torch.device("cuda:0"))
    cond, uncond = torch.ones(1, 2), torch.zeros(1, 2)

    def bind(timesteps=timesteps, negative=uncond, guidance=5.0, guidance_2=5.0):
        runtime.bind_schedule(pipeline, timesteps, latent, cond, negative, guidance, guidance_2, 875.0, torch.bfloat16)

    with runtime.request_context():
        bind()
        assert runtime.components["transformer"].plan.evaluations == 3
        assert runtime.routes == ("transformer",) * 3
        assert runtime.scheduler_identity[:2] == (id(pipeline.scheduler), id(timesteps))
    for guidance, negative, batch in ((1, uncond, 1), (5, None, 1), (5, uncond, 2)):
        latent.shape = (batch, 16, 1, 2, 2)
        with runtime.request_context(), pytest.raises(ValueError):
            bind(negative=negative, guidance=guidance, guidance_2=guidance)
    latent.shape = (1, 16, 1, 2, 2)
    with runtime.request_context(), pytest.raises(ValueError, match="one guidance scale"):
        bind(guidance_2=3.0)
    pipeline.transformer_2 = object()
    with runtime.request_context(), pytest.raises(ValueError, match="one transformer"):
        bind()
    pipeline.transformer_2 = None
    with runtime.request_context(), pytest.raises(ValueError, match="time grid"):
        bind(timesteps=timesteps.clone())


def test_request_revalidates_capabilities_after_install(backend_module, monkeypatch):
    checks = []

    def validate(pipeline):
        checks.append(pipeline)
        if getattr(pipeline, "foreign_hook", False):
            raise ValueError("foreign hook")

    monkeypatch.setattr(backend_module, "validate_pipeline", validate)
    pipeline = SimpleNamespace(transformer=object(), od_config=execution_config())
    backend = backend_module.AlignCacheBackend()
    backend.enable(pipeline)
    pipeline.od_config.enforce_eager = False
    with pytest.raises(ValueError, match="enforce_eager"):
        with backend.request_context(pipeline, request()):
            pytest.fail("invalid request entered")
    pipeline.od_config.enforce_eager = True
    pipeline.foreign_hook = True
    with pytest.raises(ValueError, match="foreign hook"):
        with backend.request_context(pipeline, request()):
            pytest.fail("invalid request entered")
    assert backend.runtime.components["transformer"].state is None and not backend.runtime.active
    pipeline.foreign_hook = False
    with backend.request_context(pipeline, request()):
        assert backend.runtime.active
    assert len(checks) == 3


def test_wan22_install_preflight_and_low_component_ownership(backend_module, monkeypatch):
    backend = backend_module.AlignCacheBackend({"align_cache": {"profile": "wan22_t2v_reference_v1"}})
    pipeline = SimpleNamespace(transformer=object(), transformer_2=object(), od_config=execution_config())

    def reject_second(p):
        assert p.transformer is pipeline.transformer
        raise ValueError("second component rejected")

    monkeypatch.setattr(backend, "_validate_pipeline", reject_second)
    with pytest.raises(ValueError, match="second component"):
        backend.enable(pipeline)
    assert not hasattr(pipeline, "_aligncache_runtime") and not backend.enabled
    monkeypatch.setattr(backend, "_validate_pipeline", lambda p: None)
    backend.enable(pipeline)
    installed = pipeline._aligncache_runtime
    backend.enable(pipeline)
    assert pipeline._aligncache_runtime is installed
    pipeline.transformer_2 = object()
    with pytest.raises(RuntimeError, match="ownership"):
        backend.refresh(pipeline)
    backend.disable(pipeline)
    assert not hasattr(pipeline, "_aligncache_runtime")
