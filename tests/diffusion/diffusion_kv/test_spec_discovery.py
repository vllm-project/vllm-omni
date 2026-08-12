# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

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
    worker.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(kv_cache_memory_bytes=4096))
    monkeypatch.setattr(diffusion_worker_module, "_all_gather_rank_values", lambda value: [value])

    assert worker.determine_available_kv_memory() == [4096]


def test_worker_treats_zero_kv_memory_budget_as_unset(monkeypatch) -> None:
    worker = object.__new__(DiffusionWorker)
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=0,
            gpu_memory_utilization=0.75,
        )
    )
    monkeypatch.setattr(diffusion_worker_module, "_all_gather_rank_values", lambda value: [value])
    monkeypatch.setattr(
        diffusion_worker_module.current_omni_platform,
        "get_device_memory",
        lambda _device: (200, 1000),
    )
    monkeypatch.setattr(diffusion_worker_module, "get_process_gpu_memory", lambda _rank: 400)

    assert worker.determine_available_kv_memory() == [350]


def test_worker_derives_kv_budget_from_process_residency(monkeypatch) -> None:
    worker = object.__new__(DiffusionWorker)
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=None,
            gpu_memory_utilization=0.75,
        )
    )
    monkeypatch.setattr(diffusion_worker_module, "_all_gather_rank_values", lambda value: [value])
    monkeypatch.setattr(
        diffusion_worker_module.current_omni_platform,
        "get_device_memory",
        lambda _device: (200, 1000),
    )
    monkeypatch.setattr(diffusion_worker_module, "get_process_gpu_memory", lambda _rank: 400)

    assert worker.determine_available_kv_memory() == [350]


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
