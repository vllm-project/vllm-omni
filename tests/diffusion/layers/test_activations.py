# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the shared ColumnParallelGELU layer."""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True, scope="module")
def _init_distributed():
    """Initialize the minimal distributed environment required by
    ColumnParallelLinear (tensor-parallel group must exist)."""
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="env://",
        backend="gloo",
    )
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


class TestColumnParallelGELU:
    """Verify ColumnParallelGELU behaves as ColumnParallelLinear + F.gelu."""

    def test_forward_shape_tanh(self) -> None:
        """Forward with approximate='tanh' produces correct output shape."""
        from vllm_omni.diffusion.layers.activations import ColumnParallelGELU

        layer = ColumnParallelGELU(16, 32, approximate="tanh")
        x = torch.randn(2, 4, 16)
        assert layer(x).shape == (2, 4, 32)

    def test_forward_shape_none(self) -> None:
        """Forward with approximate='none' produces correct output shape."""
        from vllm_omni.diffusion.layers.activations import ColumnParallelGELU

        layer = ColumnParallelGELU(16, 64, approximate="none")
        x = torch.randn(2, 8, 16)
        assert layer(x).shape == (2, 8, 64)

    def test_proj_attribute_exists(self) -> None:
        """The inner linear is accessible as .proj for weight loading."""
        from vllm_omni.diffusion.layers.activations import ColumnParallelGELU

        layer = ColumnParallelGELU(16, 32, approximate="tanh")
        assert hasattr(layer, "proj")
        keys = list(layer.state_dict())
        assert any("proj" in k for k in keys), f"No proj key in {keys}"

    def test_prefix_passthrough(self) -> None:
        """prefix is forwarded to ColumnParallelLinear unchanged."""
        from vllm_omni.diffusion.layers.activations import ColumnParallelGELU

        layer = ColumnParallelGELU(16, 32, approximate="tanh", prefix="blocks.0.ff.net.0.proj")
        assert layer.proj.prefix == "blocks.0.ff.net.0.proj"

    def test_no_prefix(self) -> None:
        """Default prefix is empty string."""
        from vllm_omni.diffusion.layers.activations import ColumnParallelGELU

        layer = ColumnParallelGELU(16, 32, approximate="tanh")
        assert layer.proj.prefix == ""
