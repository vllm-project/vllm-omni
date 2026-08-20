# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

import vllm_omni.diffusion.worker.diffusion_worker as diffusion_worker_module
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _Backend:
    @classmethod
    def indexes_kv_by_block_stride(cls) -> bool:
        return True


def _attention(*, enabled: bool) -> Attention:
    attention = Attention.__new__(Attention)
    nn.Module.__init__(attention)
    attention.paged_kv_cache_role = "primary" if enabled else None
    attention.paged_kv_cache_dtype = torch.bfloat16
    attention.num_kv_heads = 2
    attention.head_size = 8
    attention.causal = False
    attention.attn_backend = _Backend
    return attention


def _runner(attention: Attention) -> DiffusionModelRunner:
    runner = object.__new__(DiffusionModelRunner)
    runner.od_config = SimpleNamespace(diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=4),
        model_config=SimpleNamespace(dtype=torch.float16),
    )
    pipeline = nn.Module()
    pipeline.image_attention = attention
    runner.pipeline = pipeline
    return runner


def test_runner_discovers_native_spec_from_loaded_attention() -> None:
    specs = _runner(_attention(enabled=True)).get_kv_cache_spec()

    assert set(specs) == {"image_attention"}
    spec = specs["image_attention"]
    assert isinstance(spec, FullAttentionSpec)
    assert spec.block_size == 4
    assert spec.num_kv_heads == 2
    assert spec.head_size == 8
    assert spec.dtype is torch.bfloat16
    assert spec.indexes_kv_by_block_stride is True
    assert spec.non_causal is True


def test_runner_rejects_paged_mode_without_cache_enabled_attention() -> None:
    runner = _runner(_attention(enabled=False))

    with pytest.raises(RuntimeError, match="no cache-enabled Attention"):
        runner.get_kv_cache_spec()


def test_runner_retains_matching_rank_local_config() -> None:
    runner = _runner(_attention(enabled=True))
    spec = runner.get_kv_cache_spec()["image_attention"]
    config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[KVCacheTensor(size=spec.page_size_bytes * 8, shared_by=["image_attention"])],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["image_attention"], kv_cache_spec=spec)],
    )

    runner.set_kv_cache_config(config)

    assert runner.kv_cache_config is config


def test_runner_rejects_rank_local_config_for_different_layers() -> None:
    runner = _runner(_attention(enabled=True))
    spec = runner.get_kv_cache_spec()["image_attention"]
    config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[KVCacheTensor(size=spec.page_size_bytes * 8, shared_by=["other_attention"])],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["other_attention"], kv_cache_spec=spec)],
    )

    with pytest.raises(ValueError, match="layer mismatch"):
        runner.set_kv_cache_config(config)


def test_worker_selects_its_rank_local_config() -> None:
    worker = object.__new__(DiffusionWorker)
    worker.rank = 1
    worker.od_config = SimpleNamespace(num_gpus=2)
    worker.model_runner = SimpleNamespace(set_kv_cache_config=lambda config: setattr(worker, "installed", config))
    configs = [object(), object()]

    worker.set_kv_cache_configs(configs)

    assert worker.installed is configs[1]
    with pytest.raises(ValueError, match="rank count mismatch"):
        worker.set_kv_cache_configs(configs[:1])


def test_worker_honors_explicit_kv_memory_budget(monkeypatch) -> None:
    worker = object.__new__(DiffusionWorker)
    worker.rank = 0
    worker.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(kv_cache_memory_bytes=4096))
    worker.init_snapshot = SimpleNamespace(free_memory=16384)
    worker.requested_memory = 8192
    worker.model_runner = SimpleNamespace(profile_run=Mock(), model_memory_usage=1024)
    monkeypatch.setattr(diffusion_worker_module, "_all_gather_rank_values", lambda value: [value])
    profile_request = object()

    assert worker.determine_available_kv_memory(profile_request) == [4096]
    worker.model_runner.profile_run.assert_called_once_with(profile_request)


def test_worker_treats_zero_kv_memory_budget_as_profiled_auto_sizing(monkeypatch) -> None:
    worker = object.__new__(DiffusionWorker)
    worker.rank = 0
    worker.init_snapshot = SimpleNamespace(free_memory=900)
    worker.requested_memory = 750
    worker.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=0,
            gpu_memory_utilization=0.75,
        )
    )
    worker.model_runner = SimpleNamespace(profile_run=Mock(), model_memory_usage=100)
    monkeypatch.setattr(diffusion_worker_module, "_all_gather_rank_values", lambda value: [value])
    profile_result = SimpleNamespace(
        non_kv_cache_memory=400,
        after_profile=SimpleNamespace(free_memory=500),
    )
    monkeypatch.setattr(
        diffusion_worker_module,
        "memory_profiling",
        lambda snapshot, weights_memory: nullcontext(profile_result),
    )
    profile_request = object()

    assert worker.determine_available_kv_memory(profile_request) == [350]
    worker.model_runner.profile_run.assert_called_once_with(profile_request)


@pytest.mark.parametrize("non_kv_cache_memory", [750, 800])
def test_worker_rejects_non_positive_profiled_kv_memory(monkeypatch, non_kv_cache_memory: int) -> None:
    worker = object.__new__(DiffusionWorker)
    worker.rank = 0
    worker.init_snapshot = SimpleNamespace(free_memory=900)
    worker.requested_memory = 750
    worker.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=None,
            gpu_memory_utilization=0.75,
        )
    )
    worker.model_runner = SimpleNamespace(profile_run=Mock(), model_memory_usage=100)
    monkeypatch.setattr(diffusion_worker_module, "_all_gather_rank_values", lambda value: [value])
    profile_result = SimpleNamespace(
        non_kv_cache_memory=non_kv_cache_memory,
        after_profile=SimpleNamespace(free_memory=100),
    )
    monkeypatch.setattr(
        diffusion_worker_module,
        "memory_profiling",
        lambda snapshot, weights_memory: nullcontext(profile_result),
    )
    profile_request = object()

    with pytest.raises(
        RuntimeError,
        match=(
            "rank 0: RuntimeError: No memory remains for Diffusion KV cache after profiling: "
            f"requested_memory=750 bytes, non_kv_cache_memory={non_kv_cache_memory} bytes"
        ),
    ):
        worker.determine_available_kv_memory(profile_request)

    worker.model_runner.profile_run.assert_called_once_with(profile_request)


def test_worker_profiles_activation_headroom_instead_of_current_residency(monkeypatch) -> None:
    worker = object.__new__(DiffusionWorker)
    worker.rank = 0
    worker.init_snapshot = SimpleNamespace(free_memory=1200)
    worker.requested_memory = 1000
    worker.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=None,
            gpu_memory_utilization=0.8,
        )
    )
    worker.model_runner = SimpleNamespace(profile_run=Mock(), model_memory_usage=250)
    monkeypatch.setattr(diffusion_worker_module, "_all_gather_rank_values", lambda value: [value])
    profile_result = SimpleNamespace(
        non_kv_cache_memory=650,
        after_profile=SimpleNamespace(free_memory=550),
    )
    profiling_calls = []

    def profile(snapshot, *, weights_memory):
        profiling_calls.append((snapshot, weights_memory))
        return nullcontext(profile_result)

    monkeypatch.setattr(
        diffusion_worker_module,
        "memory_profiling",
        profile,
    )
    profile_request = object()

    assert worker.determine_available_kv_memory(profile_request) == [350]
    assert profiling_calls == [(worker.init_snapshot, 250)]
    worker.model_runner.profile_run.assert_called_once_with(profile_request)


def test_worker_rejects_profile_without_preload_snapshot(monkeypatch) -> None:
    worker = object.__new__(DiffusionWorker)
    worker.init_snapshot = None
    worker.requested_memory = None
    worker.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(kv_cache_memory_bytes=None))
    worker.model_runner = SimpleNamespace()
    monkeypatch.setattr(diffusion_worker_module, "_all_gather_rank_values", lambda value: [value])

    with pytest.raises(RuntimeError, match="snapshot was not captured before model loading"):
        worker.determine_available_kv_memory(object())


def test_rank_probe_gathers_local_failure_before_raising(monkeypatch) -> None:
    gathered = []

    def gather(local_result):
        gathered.append(local_result)
        return [local_result, (True, "peer-result")]

    def fail_local_probe():
        raise ValueError("local failure")

    monkeypatch.setattr(diffusion_worker_module, "_all_gather_rank_values", gather)

    with pytest.raises(RuntimeError, match="rank 0: ValueError: local failure"):
        diffusion_worker_module._run_and_gather_rank_values(
            "test probe",
            fail_local_probe,
        )

    assert gathered == [(False, "ValueError: local failure")]
