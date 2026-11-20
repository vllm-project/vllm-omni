# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from vllm.platforms import current_platform

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import (
    HiggsAudioV3TalkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA graphs")
@pytest.mark.parametrize("batch", [1, 3])
def test_mlp_graph_first_call_and_replay_compute_current_inputs(monkeypatch, batch):
    model = object.__new__(HiggsAudioV3TalkerForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model._mlp_graphs = {}
    model._mlp_graph_disabled = set()
    monkeypatch.setattr(current_platform, "get_global_graph_pool", lambda: None)

    hidden = torch.arange(batch * 8, device="cuda", dtype=torch.float32).reshape(batch, 8)
    residual = torch.full_like(hidden, 2)
    # Capture records the operations without executing them. NaN-filled
    # outputs make returning a newly captured graph before replay observable.
    normed = torch.full_like(hidden, float("nan"))
    residual_out = torch.full_like(hidden, float("nan"))
    mlp_out = torch.full_like(hidden, float("nan"))

    def norm(h, r):
        torch.add(h, r, out=residual_out)
        torch.mul(residual_out, 0.5, out=normed)
        return normed, residual_out

    def mlp(h):
        return torch.mul(h, 3, out=mlp_out)

    layer = SimpleNamespace(post_attention_layernorm=norm, mlp=mlp)
    for offset in (0, 10):
        current = hidden + offset
        expected_residual = current + residual
        actual = model._run_qwen3_mlp_graph(0, layer, current, residual)
        assert actual is not None
        torch.testing.assert_close(actual[0], expected_residual * 1.5, rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected_residual, rtol=0, atol=0)
        assert len(model._mlp_graphs) == 1
