# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest


@pytest.fixture(autouse=True)
def _single_rank_tensor_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide the TP metadata required by vLLM parallel linear layers."""
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear

    from vllm_omni.diffusion.models.abot_world import transformer

    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(transformer, "get_tensor_model_parallel_world_size", lambda: 1)
