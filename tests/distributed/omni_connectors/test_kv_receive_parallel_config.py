# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for receive-side parallel config plumbing."""

import pytest

from vllm_omni.config.omni_config import (
    OmniStageDiffusionParallelConfig,
    OmniStageParallelConfig,
)
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    OmniKVCacheConfig,
    OmniKVTransferManager,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_factories_preserve_unified_parallel_config_objects():
    parallel_config = OmniStageDiffusionParallelConfig(
        cfg_parallel_size=2,
        ulysses_degree=2,
    )
    od_config = _Config(
        omni_kv_config={"rank_mapping": {"from_tp": 2, "to_tp": 4}},
        parallel_config=parallel_config,
    )

    manager = OmniKVTransferManager.from_od_config(od_config)

    assert manager.config.parallel_config is parallel_config

    vllm_parallel_config = OmniStageParallelConfig(pipeline_parallel_size=2)
    vllm_config = _Config(parallel_config=vllm_parallel_config)
    model_config = _Config(omni_kv_config={"rank_mapping": {"from_tp": 1, "to_tp": 1}})

    manager = OmniKVTransferManager.from_vllm_config(vllm_config, model_config)

    assert manager.config.parallel_config is vllm_parallel_config


def test_receive_parallel_sizes_use_unified_diffusion_parallel_config():
    parallel_config = OmniStageDiffusionParallelConfig(
        cfg_parallel_size=2,
        ring_degree=2,
        ulysses_degree=2,
        pipeline_parallel_size=3,
        data_parallel_size=2,
    )
    manager = OmniKVTransferManager(
        OmniKVCacheConfig(
            parallel_config=parallel_config,
        )
    )

    assert manager._configured_parallel_int("cfg_parallel_size") == 2
    assert manager._configured_parallel_int("sequence_parallel_size") == 4
    assert manager._configured_parallel_int("ring_degree") == 2
    assert manager._configured_parallel_int("ulysses_degree") == 2
    assert manager._configured_parallel_int("pipeline_parallel_size") == 3
    assert manager._configured_parallel_int("data_parallel_size") == 2


def test_receive_parallel_sizes_support_common_parallel_config():
    parallel_config = OmniStageParallelConfig(pipeline_parallel_size=2)
    manager = OmniKVTransferManager(OmniKVCacheConfig(parallel_config=parallel_config))

    assert manager._configured_parallel_int("pipeline_parallel_size") == 2
    assert manager._configured_parallel_int("cfg_parallel_size", default=1) == 1
