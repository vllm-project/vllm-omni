# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

import vllm_omni.diffusion.diffusion_kv.initialization as initialization
import vllm_omni.diffusion.vllm_config as diffusion_vllm_config
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.sched.request_scheduler import RequestScheduler
from vllm_omni.quantization import build_quant_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _Executor:
    def __init__(self) -> None:
        self.specs = [{"layer0": object()}, {"layer0": object()}]
        self.memory = [1024, 1024]
        self.configs = None
        self.resolved_max_model_len = None
        self.profile_requests = None

    def get_kv_cache_specs(self):
        return self.specs

    def determine_available_kv_memory(self, profile_requests):
        self.profile_requests = profile_requests
        return self.memory

    def set_kv_cache_configs(self, configs, resolved_max_model_len) -> None:
        self.configs = configs
        self.resolved_max_model_len = resolved_max_model_len


def _od_config(**overrides):
    values = dict(
        model="test-diffusion",
        diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
        dtype=torch.bfloat16,
        max_model_len=64,
        quantization_config=None,
        tf_model_config=None,
        enforce_eager=True,
        is_moe=False,
        additional_config={},
        profiler_config=None,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
        ),
        gpu_memory_utilization=0.9,
        kv_cache_memory_bytes=None,
        max_num_seqs=1,
        max_num_batched_tokens=64,
        num_gpus=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_control_plane_builds_worker_and_scheduler_configs(monkeypatch) -> None:
    executor = _Executor()
    od_config = _od_config(num_gpus=2, max_model_len=-1)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=256),
        max_in_flight_tokens=128,
    )
    worker_configs = [object(), object()]
    scheduler_kv_cache_config = object()
    profile_requests = [object()]
    monkeypatch.setattr(initialization, "create_diffusion_vllm_config", lambda *args, **kwargs: vllm_config)
    build = Mock(return_value=(worker_configs, scheduler_kv_cache_config))
    monkeypatch.setattr(
        initialization,
        "build_native_kv_cache_configs",
        build,
    )
    monkeypatch.setattr(initialization, "resolve_kv_cache_block_sizes", lambda *args: (16, 16))

    result = initialization.initialize_diffusion_kv_control_plane(
        executor,
        od_config,
        profile_requests=profile_requests,
        device=torch.device("cpu"),
    )

    build.assert_called_once_with(vllm_config, executor.specs, executor.memory)
    assert result == (scheduler_kv_cache_config, 16, 16, vllm_config)
    assert executor.configs == worker_configs
    assert executor.resolved_max_model_len == vllm_config.model_config.max_model_len
    assert executor.profile_requests is profile_requests


def test_dense_mode_skips_worker_kv_initialization(monkeypatch) -> None:
    executor = _Executor()
    od_config = _od_config(diffusion_kv_mode=DiffusionKVCacheMode.DENSE_LEGACY)
    get_device = Mock(side_effect=AssertionError("dense mode must not resolve a KV device"))
    monkeypatch.setattr(initialization.current_omni_platform, "get_torch_device", get_device)

    result = initialization.initialize_diffusion_kv_control_plane(executor, od_config)

    assert result is None
    assert executor.configs is None
    get_device.assert_not_called()


def test_control_plane_rejects_rank_count_mismatch(monkeypatch) -> None:
    executor = _Executor()
    executor.memory = [1024]
    od_config = _od_config(num_gpus=2)
    monkeypatch.setattr(initialization, "create_diffusion_vllm_config", lambda *args, **kwargs: object())

    with pytest.raises(ValueError, match="rank count mismatch"):
        initialization.initialize_diffusion_kv_control_plane(
            executor,
            od_config,
            profile_requests=[object()],
            device=torch.device("cpu"),
        )


def test_paged_control_plane_requires_prepared_profile_request() -> None:
    with pytest.raises(ValueError, match="prepared memory profile requests"):
        initialization.initialize_diffusion_kv_control_plane(
            _Executor(),
            _od_config(),
            device=torch.device("cpu"),
        )


def test_native_config_builder_produces_matching_worker_and_scheduler_pools() -> None:
    od_config = _od_config()
    vllm_config = diffusion_vllm_config.create_diffusion_vllm_config(torch.device("cpu"), od_config)
    spec = FullAttentionSpec(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
        non_causal=True,
    )
    num_blocks = 32

    worker_configs, scheduler_kv_cache_config = initialization.build_native_kv_cache_configs(
        vllm_config,
        [{"layer0": spec}],
        [spec.page_size_bytes * num_blocks],
    )

    assert len(worker_configs) == 1
    assert worker_configs[0].num_blocks == num_blocks
    assert scheduler_kv_cache_config.num_blocks == num_blocks
    assert worker_configs[0].kv_cache_groups == scheduler_kv_cache_config.kv_cache_groups


def test_native_control_plane_config_initializes_scheduler() -> None:
    od_config = _od_config()
    vllm_config = diffusion_vllm_config.create_diffusion_vllm_config(torch.device("cpu"), od_config)
    spec = FullAttentionSpec(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
        non_causal=True,
    )
    _, scheduler_kv_cache_config = initialization.build_native_kv_cache_configs(
        vllm_config,
        [{"layer0": spec}],
        [spec.page_size_bytes * 32],
    )

    scheduler = RequestScheduler()
    scheduler.initialize(
        od_config,
        kv_cache_config=scheduler_kv_cache_config,
        scheduler_block_size=spec.block_size,
        hash_block_size=spec.block_size,
        kv_vllm_config=vllm_config,
    )

    assert scheduler._diffusion_kv_manager is not None
    scheduler.close()


def test_dense_config_keeps_new_paged_sizing_inputs_inactive() -> None:
    od_config = _od_config(
        diffusion_kv_mode=DiffusionKVCacheMode.DENSE_LEGACY,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=4,
            data_parallel_size=1,
            enable_expert_parallel=False,
        ),
        gpu_memory_utilization=0.5,
        kv_cache_memory_bytes=4096,
        max_num_batched_tokens=64,
    )
    vllm_config = diffusion_vllm_config.create_base_diffusion_vllm_config(torch.device("cpu"), od_config)
    original_pp_size = vllm_config.parallel_config.pipeline_parallel_size
    original_memory_utilization = vllm_config.cache_config.gpu_memory_utilization
    original_memory_bytes = vllm_config.cache_config.kv_cache_memory_bytes
    original_token_budget = vllm_config.scheduler_config.max_num_batched_tokens

    diffusion_vllm_config.configure_diffusion_vllm_config(vllm_config, od_config)

    assert vllm_config.parallel_config.pipeline_parallel_size == original_pp_size
    assert vllm_config.cache_config.gpu_memory_utilization == original_memory_utilization
    assert vllm_config.cache_config.kv_cache_memory_bytes == original_memory_bytes
    assert vllm_config.scheduler_config.max_num_batched_tokens == original_token_budget


def test_paged_config_forwards_gpu_memory_utilization_to_native_cache_config() -> None:
    vllm_config = diffusion_vllm_config.create_diffusion_vllm_config(
        torch.device("cpu"),
        _od_config(gpu_memory_utilization=0.42),
    )

    assert vllm_config.cache_config.gpu_memory_utilization == 0.42


def test_diffusion_vllm_model_config_supplies_dtype_for_quant_methods() -> None:
    quantization_config = build_quant_config(
        {
            "quant_method": "modelopt",
            "quant_algo": "FP8",
            "ignore": [],
        }
    )
    model_config = diffusion_vllm_config._make_diffusion_vllm_model_config(
        _od_config(
            diffusion_kv_mode=DiffusionKVCacheMode.DENSE_LEGACY,
            quantization_config=quantization_config,
        )
    )

    assert model_config.dtype is torch.bfloat16
    assert model_config.quantization == "modelopt"
    assert model_config.quantization_config is quantization_config
    assert model_config.is_quantized
    assert model_config.is_moe is False
    assert model_config.is_nvfp4_quantized() is False
    assert model_config.original_max_model_len == model_config.max_model_len

    unquantized_model_config = diffusion_vllm_config._make_diffusion_vllm_model_config(
        _od_config(
            diffusion_kv_mode=DiffusionKVCacheMode.DENSE_LEGACY,
            quantization_config=None,
        )
    )
    assert not unquantized_model_config.is_quantized


def test_minus_one_max_model_len_uses_model_limit() -> None:
    od_config = SimpleNamespace(
        max_model_len=-1,
        tf_model_config=SimpleNamespace(max_position_embeddings=4096),
        diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
    )

    assert diffusion_vllm_config.resolve_diffusion_max_model_len(od_config) == 4096

    model_config = diffusion_vllm_config._make_diffusion_vllm_model_config(
        _od_config(
            max_model_len=-1,
            tf_model_config=SimpleNamespace(max_position_embeddings=4096),
        )
    )
    assert model_config.max_model_len == 4096
    assert model_config.original_max_model_len == -1


def test_minus_one_max_model_len_is_auto_fitted_by_native_cache_sizing() -> None:
    vllm_config = diffusion_vllm_config.create_diffusion_vllm_config(
        torch.device("cpu"),
        _od_config(
            max_model_len=-1,
            tf_model_config=SimpleNamespace(max_position_embeddings=4096),
        ),
    )
    spec = FullAttentionSpec(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
        non_causal=True,
    )
    model_limit = vllm_config.model_config.max_model_len

    initialization.build_native_kv_cache_configs(
        vllm_config,
        [{"layer0": spec}],
        [spec.page_size_bytes * 16],
    )

    assert vllm_config.model_config.original_max_model_len == -1
    # Sixteen physical pages provide ``16 * block_size`` tokens.  The native
    # backend chooses the block geometry, so keep this assertion backend
    # independent instead of assuming the generic 16-token block size.
    expected_max_model_len = min(model_limit, spec.block_size * 16)
    assert vllm_config.model_config.max_model_len == expected_max_model_len
