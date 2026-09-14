# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest


@pytest.fixture
def mock_tp1(monkeypatch):
    """Allow standalone linear-layer tests without a distributed process group."""
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)
