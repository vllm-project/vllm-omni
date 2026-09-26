# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest


@pytest.fixture(autouse=True)
def _single_rank_tensor_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide the TP metadata required by vLLM parallel linear layers."""
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear

    from vllm_omni.diffusion.attention import layer
    from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
    from vllm_omni.diffusion.models.abot_world import abot_world_transformer

    monkeypatch.setattr(layer, "get_attn_backend_for_role", lambda **kwargs: (SDPABackend, None))
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(abot_world_transformer, "get_tensor_model_parallel_world_size", lambda: 1)
