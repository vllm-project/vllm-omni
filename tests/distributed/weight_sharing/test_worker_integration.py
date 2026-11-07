# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real backend control/loader seam with CPU storage primitives, not CUDA evidence."""

import importlib
import os
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import msgspec
import pytest
import torch

from tests.distributed.weight_sharing.helpers import FakeCodec, load_class_methods, make_weights

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def memory_worker(weight_sharing, monkeypatch, mocker):
    """Execute the production device-init method with CPU device/NCCL doubles."""
    runtime = importlib.import_module(f"{weight_sharing.__name__}.runtime")
    worker_module = importlib.import_module(f"{weight_sharing.__name__}.worker_session")
    monkeypatch.setitem(sys.modules, "vllm_omni.distributed.weight_sharing.worker_session", worker_module)
    gib = 1024**3
    plan = runtime.SharingPlan(
        0,
        1,
        ("GPU-0",),
        (runtime.DeviceBudget(76 * gib, 4 * gib, 2 * gib, gib),),
        "model",
        "revision",
        "layout",
    )
    bootstrap = runtime.SharingRuntime(plan, None).bootstrap("consumer", (b"export",))
    config = SimpleNamespace(
        kernel_config=SimpleNamespace(ir_op_priority=None),
        cache_config=SimpleNamespace(gpu_memory_utilization=0.9, kv_cache_memory_bytes=gib),
    )
    snapshot = SimpleNamespace(total_memory=80 * gib, free_memory=24 * gib)

    def ordinary_request_memory(snapshot, cache):
        requested = int(snapshot.total_memory * cache.gpu_memory_utilization)
        if snapshot.free_memory < requested:
            raise ValueError("ordinary stage requires 90% free memory")
        return requested

    ordinary = mocker.Mock(side_effect=ordinary_request_memory)
    kv_modes = SimpleNamespace(DENSE_LEGACY="dense", PAGED_SCHEDULER="paged")
    namespace = dict(
        os=os,
        gc=mocker.Mock(),
        logger=mocker.Mock(),
        current_omni_platform=mocker.Mock(),
        create_diffusion_vllm_config=mocker.Mock(return_value=config),
        _resolve_ir_op_priority=mocker.Mock(return_value=[]),
        set_forward_context=lambda **kwargs: nullcontext(),
        set_current_vllm_config=lambda cfg: nullcontext(),
        init_distributed_environment=mocker.Mock(),
        initialize_model_parallel=mocker.Mock(),
        _setup_diffusion_worker_proc_title_and_log_prefix=mocker.Mock(),
        init_workspace_manager=mocker.Mock(),
        MemorySnapshot=mocker.Mock(return_value=snapshot),
        request_memory=ordinary,
        format_gib=str,
        DiffusionKVCacheMode=kv_modes,
    )
    cls = load_class_methods(
        "vllm_omni/diffusion/worker/diffusion_worker.py", "DiffusionWorker", ["init_device"], namespace
    )
    worker = cls()
    worker.rank = worker.local_rank = 0
    worker.distributed_init_method = "tcp://localhost:12345"
    worker.set_proc_title = False
    worker.od_config = SimpleNamespace(
        num_gpus=1,
        parallel_config=mocker.Mock(),
        diffusion_kv_mode=kv_modes.PAGED_SCHEDULER,
        additional_config={runtime.CONFIG_KEY: msgspec.to_builtins(bootstrap)},
    )
    for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE"):
        monkeypatch.setenv(key, "test")
    return worker, config, snapshot, ordinary


def test_shared_consumer_init_does_not_request_whole_gpu_ratio(memory_worker):
    worker, config, snapshot, ordinary = memory_worker
    assert snapshot.free_memory < snapshot.total_memory * config.cache_config.gpu_memory_utilization
    worker.init_device()
    assert worker.requested_memory == 20 * 1024**3  # Free space less the group reserve; KV stays fixed.
    ordinary.assert_not_called()


@pytest.mark.parametrize("free_gib", [24, 76])
def test_ordinary_consumer_keeps_existing_startup_memory_check(memory_worker, free_gib):
    worker, config, snapshot, ordinary = memory_worker
    worker.od_config.additional_config.clear()
    snapshot.free_memory = free_gib * 1024**3
    if free_gib < 72:
        with pytest.raises(ValueError, match="90% free"):
            worker.init_device()
    else:
        worker.init_device()
        assert worker.requested_memory == 72 * 1024**3
    ordinary.assert_called_once_with(snapshot, config.cache_config)


@pytest.mark.parametrize("bad", ["free_memory", "kv_budget", "missing_kv", "bootstrap"])
def test_shared_consumer_init_rejects_invalid_or_insufficient_budget(memory_worker, bad):
    worker, config, snapshot, ordinary = memory_worker
    if bad == "free_memory":
        snapshot.free_memory = 4 * 1024**3  # Less than fixed KV + group reserve.
    elif bad == "kv_budget":
        config.cache_config.kv_cache_memory_bytes *= 2
    elif bad == "missing_kv":
        config.cache_config.kv_cache_memory_bytes = None
    else:
        key = next(iter(worker.od_config.additional_config))
        worker.od_config.additional_config[key] = {}
    with pytest.raises(ValueError):
        worker.init_device()
    ordinary.assert_not_called()


@pytest.mark.parametrize("role,kv_bytes", [("provider", 128), ("consumer", 64)])
def test_shared_stage_memory_envelope_preserves_fixed_kv(weight_sharing, role, kv_bytes):
    runtime = importlib.import_module(f"{weight_sharing.__name__}.runtime")
    worker = importlib.import_module(f"{weight_sharing.__name__}.worker_session")
    budget = runtime.DeviceBudget(900, 100, 128, 64)
    args = dict(role=role, rank=0, kv_cache_bytes=kv_bytes, free_bytes=300)
    assert worker.sharing_memory_request(budget, **args) == 200
    assert worker.sharing_memory_request(None, **args) is None
    with pytest.raises(ValueError, match="fixed KV"):
        worker.sharing_memory_request(budget, **(args | {"kv_cache_bytes": kv_bytes + 1}))
    with pytest.raises(ValueError, match="insufficient free memory"):
        worker.sharing_memory_request(budget, **(args | {"free_bytes": 100 + kv_bytes - 1}))


@pytest.mark.parametrize("manager_mode", ["none", "default", "close_error"])
def test_loader_handles_are_reused_and_detached_before_release(weight_sharing, monkeypatch, mocker, manager_mode):
    runtime = importlib.import_module(f"{weight_sharing.__name__}.runtime")
    worker = importlib.import_module(f"{weight_sharing.__name__}.worker_session")
    api = importlib.import_module(f"{weight_sharing.__name__}.contracts")
    platform = ModuleType("vllm_omni.platforms")
    events = []
    memory = dict(peak=512, reserved_peak=512, resident=512)
    setattr(
        platform,
        "current_omni_platform",
        SimpleNamespace(
            memory_reserved=lambda device: memory["resident"],
            get_free_memory=lambda: 1024,
            synchronize=lambda: events.append("sync"),
        ),
    )
    monkeypatch.setitem(sys.modules, "vllm_omni.platforms", platform)
    monkeypatch.setattr(torch.accelerator, "max_memory_allocated", lambda device: memory["peak"])
    monkeypatch.setattr(torch.accelerator, "max_memory_reserved", lambda device: memory["reserved_peak"])
    codec = FakeCodec(api)
    plan = runtime.SharingPlan(
        0, 1, ("GPU-0",), (runtime.DeviceBudget(2048, 64, 64, 64),), "model", "revision", "layout"
    )
    group = runtime.SharingRuntime(plan, None)
    config = SimpleNamespace(additional_config={runtime.CONFIG_KEY: msgspec.to_builtins(group.bootstrap("provider"))})
    worker.configure_worker(config, role="provider", rank=0)
    assert config.weight_sharing.model_id == "model"
    assert config.weight_sharing.checkpoint_revision == "revision"
    context = worker.build_session_context(config.weight_sharing, 0)
    bindings, spec = make_weights(api, two_storages=True)
    provider = weight_sharing.TorchCudaIpcBackend(codec=codec)
    exported = provider.prepare_export(bindings, spec, context)
    ar = SimpleNamespace(_weight_share_backend=provider, _weight_share_session=exported)
    worker.install_runner_session(ar, config, device_index=0, detach=lambda: None)
    assert ar.weight_sharing.backend is provider and ar.weight_sharing.handle is exported
    claim = worker.control_runner(ar, "claim", ())
    assert claim["private_bytes"] == 512 - provider.session_status(exported).shared_nbytes
    payload = msgspec.convert(claim, type=runtime.RankResult).payload
    config = SimpleNamespace(
        additional_config={runtime.CONFIG_KEY: msgspec.to_builtins(group.bootstrap("consumer", (payload,)))}
    )
    worker.configure_worker(config, role="consumer", rank=0)
    consumer = weight_sharing.TorchCudaIpcBackend(codec=codec)
    imported = consumer.open_import(config.weight_sharing.payload, spec, context)
    dit = SimpleNamespace(
        _weight_share_backend=consumer,
        pipeline=SimpleNamespace(_weight_share_import=imported, weights=dict(imported)),
        state_cache={"request": object()},
        vllm_config=SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context={})),
        release_captured_graphs=lambda: events.append("graphs"),
    )

    # Exercise the worker's actual callback registration, including its default
    # no-adapter manager, which owns the pipeline independently of the runner.
    class RetainingManager:
        def __init__(self, pipeline, **kwargs):
            self.pipeline = pipeline

        def close(self):
            events.append("lora_close")
            # The real manager also keeps self.pipeline after close().
            if manager_mode == "close_error":
                raise RuntimeError("LoRA cleanup failed")

    monkeypatch.setitem(sys.modules, "vllm_omni.distributed.weight_sharing.worker_session", worker)
    cls = load_class_methods(
        "vllm_omni/diffusion/worker/diffusion_worker.py",
        "DiffusionWorker",
        ["load_model", "init_lora_manager"],
        dict(
            nullcontext=nullcontext,
            set_forward_context=lambda **kwargs: nullcontext(),
            set_current_vllm_config=lambda cfg: nullcontext(),
            current_omni_platform=mocker.Mock(current_device=lambda: 0),
            gc=mocker.Mock(),
            get_process_gpu_memory=lambda device: None,
            DiffusionLoRAManager=RetainingManager,
            LoRABackend=SimpleNamespace(PEFT="peft"),
        ),
    )
    owner = cls()
    owner.model_runner = dit
    owner.lora_manager = None
    owner.device = "cpu"
    owner.local_rank = 0
    owner.vllm_config = dit.vllm_config
    owner._maybe_get_memory_pool_context = nullcontext
    owner.od_config = config
    config.lora_path = None
    config.lora_backend = "peft"
    config.lora_scale = 1.0
    config.max_cpu_loras = 1
    config.dtype = torch.bfloat16
    config.lora_application = "runtime"

    def load_model(*, initialize_lora, **kwargs):
        if manager_mode != "none":
            initialize_lora()

    dit.load_model = load_model
    owner.load_model()
    if manager_mode != "none":
        assert owner.lora_manager is not None
        assert owner.lora_manager.pipeline is dit.pipeline
    status = worker.control_runner(dit, "status", ())
    assert status["private_bytes"] == 512  # Never subtract imported bytes.
    memory.update(reserved_peak=768, resident=600)
    status = worker.control_runner(dit, "status", ())
    assert status["private_bytes"] == 768 and status["resident_bytes"] == 600
    # Profiling may reset PyTorch's counters and empty cached allocations.
    memory.update(peak=256, reserved_peak=256, resident=256)
    status = worker.control_runner(dit, "status", ())
    assert status["private_bytes"] == 768 and status["resident_bytes"] == 256
    provider.confirm_import(imported.receipt)
    # Teardown needs receipts, not memory telemetry from a failing device.
    dit.weight_sharing.memory_usage = mocker.Mock(side_effect=AssertionError("memory query"))
    if manager_mode == "close_error":
        with pytest.raises(RuntimeError, match="LoRA cleanup failed"):
            worker.control_runner(dit, "close", ())
        assert owner.lora_manager is not None
        assert owner.lora_manager.pipeline is dit.pipeline
        assert consumer.session_status(imported).state == api.SessionState.POISONED
        assert provider.release_export(exported).status == "HELD"
        return
    external = imported["first"]
    result = worker.control_runner(dit, "close", ())
    assert result["state"] == api.SessionState.RELEASING.value
    assert dit.pipeline is None and not dit.state_cache
    assert owner.lora_manager is None
    expected_events = (["lora_close"] if manager_mode != "none" else []) + ["sync", "graphs"]
    assert events == expected_events
    del external
    result = msgspec.convert(worker.control_runner(dit, "close", ()), type=runtime.RankResult)
    assert result.state == api.SessionState.CLOSED
    provider.confirm_release(result.receipt)
    assert provider.release_export(exported).status == "CLOSED"
    assert events == expected_events


def test_failed_loader_backend_can_be_poisoned_without_pipeline(weight_sharing):
    worker = importlib.import_module(f"{weight_sharing.__name__}.worker_session")
    observed: list[str] = []
    runner = SimpleNamespace(
        _weight_share_backend=SimpleNamespace(mark_failed=observed.append), _weight_share_session_id="partial-session"
    )
    worker.control_runner(runner, "fail", ())
    assert observed == ["partial-session"]
    with pytest.raises(RuntimeError, match="did not install"):
        worker.control_runner(runner, "claim", ())
