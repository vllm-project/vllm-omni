# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Test-only worker for reproducible full-payload Talker cache qualification."""

from vllm.config import CacheConfig, ModelConfig, SchedulerConfig
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_omni.core.prefix_cache.group_view import (
    check_prefix_cache_kv_groups,
    check_prefix_cache_kv_transfer,
    check_prefix_cache_token_accounting,
    check_qwen3_omni_talker_request_local_scope,
    is_qwen3_omni_talker_model,
    stage_prefix_cache_config,
)
from vllm_omni.core.prefix_cache.interface import PrefixCacheConfig
from vllm_omni.worker import gpu_model_runner
from vllm_omni.worker.gpu_ar_worker import GPUARWorker


def qualification_prefix_config(
    *,
    kv_cache_config: KVCacheConfig,
    cache_config: CacheConfig,
    kv_transfer_config: object,
    scheduler_config: SchedulerConfig,
    model_config: ModelConfig,
    is_pooling_model: bool,
    speculative_config: object = None,
) -> PrefixCacheConfig | None:
    """Bypass only the public Talker support refusal; retain safety checks."""
    if not is_qwen3_omni_talker_model(model_config):
        return stage_prefix_cache_config(
            kv_cache_config=kv_cache_config,
            cache_config=cache_config,
            kv_transfer_config=kv_transfer_config,
            scheduler_config=scheduler_config,
            model_config=model_config,
            is_pooling_model=is_pooling_model,
            speculative_config=speculative_config,
        )
    if not cache_config.enable_prefix_caching or is_pooling_model:
        return None
    check_qwen3_omni_talker_request_local_scope(
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        model_config=model_config,
        speculative_config=speculative_config,
        kv_transfer_config=kv_transfer_config,
    )
    check_prefix_cache_kv_transfer(kv_transfer_config)
    check_prefix_cache_token_accounting(cache_config, speculative_config)
    check_prefix_cache_kv_groups(kv_cache_config.kv_cache_groups)
    return PrefixCacheConfig.from_vllm_config(
        num_blocks=kv_cache_config.num_blocks,
        block_size=cache_config.block_size,
        scheduler_config=scheduler_config,
        model_config=model_config,
    )


class TalkerQualificationGPUARWorker(GPUARWorker):
    """Opt in with an explicit worker_cls; never selected by production config."""

    def init_device(self) -> None:
        if not is_qwen3_omni_talker_model(self.model_config):
            raise ValueError("Talker qualification worker requires a Qwen3-Omni Talker stage")
        # Each stage worker is a separate process. The production runner still
        # checks TP/PP/CP and scheduler scope before constructing its cache.
        gpu_model_runner.stage_prefix_cache_config = qualification_prefix_config
        super().init_device()
