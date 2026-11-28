# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest


@pytest.fixture(params=[1, 2], ids=["tp1", "tp2"])
def mock_tp(monkeypatch, request):
    """Allow standalone linear-layer tests without a distributed process group, as rank 0 of a TP group."""
    world_size = request.param
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", lambda: world_size)
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: world_size)


@pytest.fixture(autouse=True)
def force_default_gemm(monkeypatch, request):
    """Force CPU-compatible GEMM dispatch for tests using CPU tensors."""
    if request.node.get_closest_marker("cpu") is None:
        return

    from vllm.model_executor.layers.utils import default_unquantized_gemm

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda *_args, **_kwargs: default_unquantized_gemm,
    )
