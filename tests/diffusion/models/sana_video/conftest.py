# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.diffusion.models.sana_video import transformer_sana_video


@pytest.fixture(autouse=True)
def sp_world_1(monkeypatch):
    """L1 runs single-process without a sequence parallel group; keep the
    transformer on its SP1 path unless a test mocks a larger SP world."""
    monkeypatch.setattr(transformer_sana_video, "get_sequence_parallel_world_size", lambda: 1)


@pytest.fixture(scope="module")
def _distributed_group():
    """Share one TP=1 group across tests that instantiate parallel layers."""
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.utils.network_utils import get_file_store_init_method

    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=get_file_store_init_method(),
    )
    try:
        initialize_model_parallel()
        yield
    finally:
        cleanup_dist_env_and_memory()


@pytest.fixture
def _cpu_parallel_layers(monkeypatch, _distributed_group):
    """Keep the CPU kernel override isolated to each model test."""
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda *_args, **_kwargs: default_unquantized_gemm,
    )
