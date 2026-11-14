# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.utils import default_unquantized_gemm

from tests.helpers.runtime import assign_ephemeral_master_port


@pytest.fixture
def single_rank_parallel(monkeypatch):
    """Initialize a one-process tensor-parallel group for vLLM's parallel linear layers.

    Mirrors tests/diffusion/models/kandinsky6/test_transformer_kandinsky6.py.
    """
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda *_args, **_kwargs: default_unquantized_gemm,
    )
    assign_ephemeral_master_port()
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://", backend="gloo")
    initialize_model_parallel(backend="gloo")
    yield
    cleanup_dist_env_and_memory()
