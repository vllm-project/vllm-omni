# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Checked-in qualification launch stays test-only and retains refusals."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm.config import CacheConfig, SchedulerConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

from tests.model_executor.models.qwen3_omni.qualification_worker import (
    TalkerQualificationGPUARWorker,
    qualification_prefix_config,
)
from vllm_omni.config.stage_config import load_deploy_config
from vllm_omni.core.prefix_cache.group_view import stage_prefix_cache_config
from vllm_omni.core.prefix_cache.interface import OmniPrefixCacheUnmatchError
from vllm_omni.worker import gpu_model_runner
from vllm_omni.worker.gpu_ar_worker import GPUARWorker

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _arguments() -> dict[str, Any]:
    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.bfloat16)
    return dict(
        kv_cache_config=KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer"], kv_cache_spec=spec)],
        ),
        cache_config=CacheConfig(enable_prefix_caching=True, block_size=16),
        kv_transfer_config=None,
        scheduler_config=SchedulerConfig(
            max_model_len=128, is_encoder_decoder=False, max_num_seqs=2, max_num_batched_tokens=64
        ),
        model_config=SimpleNamespace(
            model_stage="talker", model_arch="Qwen3OmniMoeForConditionalGeneration", async_chunk=False
        ),
        is_pooling_model=False,
        speculative_config=None,
    )


def test_qualification_bypasses_only_public_support_refusal():
    arguments = _arguments()
    with pytest.raises(OmniPrefixCacheUnmatchError, match="not yet publicly supported"):
        stage_prefix_cache_config(**arguments)
    qualified = qualification_prefix_config(**arguments)
    assert qualified is not None
    assert qualified.num_blocks == 2
    assert qualified.block_size == 16
    assert qualified.staging_capacity_tokens == 64
    arguments["kv_transfer_config"] = SimpleNamespace(kv_role="kv_producer")
    with pytest.raises(OmniPrefixCacheUnmatchError, match="external KV"):
        qualification_prefix_config(**arguments)


@pytest.mark.parametrize("mode", ["async_chunk", "speculative", "hybrid"])
def test_qualification_keeps_scope_and_layout_checks(mode):
    arguments = _arguments()
    if mode == "async_chunk":
        arguments["model_config"].async_chunk = True
    elif mode == "speculative":
        arguments["speculative_config"] = SimpleNamespace(method="ngram")
    else:
        arguments["kv_cache_config"].kv_cache_groups *= 2
    with pytest.raises(OmniPrefixCacheUnmatchError):
        qualification_prefix_config(**arguments)


def test_explicit_worker_installs_fixture_before_device_init(monkeypatch, mocker):
    worker = object.__new__(TalkerQualificationGPUARWorker)
    worker.model_config = _arguments()["model_config"]
    monkeypatch.setattr(gpu_model_runner, "stage_prefix_cache_config", stage_prefix_cache_config)
    native_init = mocker.patch.object(GPUARWorker, "init_device")
    worker.init_device()
    assert gpu_model_runner.stage_prefix_cache_config is qualification_prefix_config
    native_init.assert_called_once_with()


def test_checked_in_launch_profile_resolves_worker_and_scope():
    config = load_deploy_config(Path(__file__).with_name("talker_qualification.yaml"))
    assert config.async_chunk is False
    assert len(config.stages) == 3
    talker = config.stages[1]
    assert talker.stage_id == 1
    assert talker.devices == "2"
    assert talker.engine_extras["worker_cls"].endswith(".TalkerQualificationGPUARWorker")
    assert talker.engine_extras["enable_prefix_caching"] is True
    assert talker.async_scheduling is False
