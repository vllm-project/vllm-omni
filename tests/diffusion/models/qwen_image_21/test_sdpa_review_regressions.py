# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.attention.backends import sdpa
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.qwen_image_21 import npu_attention

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _inputs(heads=1):
    q = torch.zeros(1, 2, heads, 1)
    k = torch.zeros(1, 2, 1, 1)
    v = torch.tensor([1.0, 9.0]).reshape(1, 2, 1, 1)
    return q, k, v


def test_shared_sdpa_does_not_use_qwen_fusion(monkeypatch):
    fusion = Mock(side_effect=AssertionError("model-specific fusion reached shared SDPA"))
    monkeypatch.setattr(npu_attention, "_npu_fusion_attention", fusion)
    impl = sdpa.SDPAImpl(1, 1, 1.0, causal=True)
    out = impl.forward_npu(*_inputs())
    torch.testing.assert_close(out.flatten(), torch.tensor([1.0, 5.0]))
    fusion.assert_not_called()


@pytest.mark.parametrize("case", ["supported", "causal", "gqa", "masked"])
def test_qwen_automatic_fusion_dispatch(monkeypatch, case):
    monkeypatch.delenv("QWEN21_FUSION_PRECISE", raising=False)
    with set_current_diffusion_config(OmniDiffusionConfig(enforce_eager=True)):
        impl = npu_attention.QwenImage21SDPAImpl(2 if case == "gqa" else 1, 1, 1.0, causal=case == "causal")
    fusion = Mock(return_value=torch.empty(0))
    monkeypatch.setattr(npu_attention, "_npu_fusion_attention", fusion)
    metadata = AttentionMetadata(attn_mask=torch.tensor([[True, False]])) if case == "masked" else None
    q, k, v = _inputs(2 if case == "gqa" else 1)
    with torch.inference_mode():
        out = impl.forward_npu(q.bfloat16(), k.bfloat16(), v.bfloat16(), metadata)
    expected_fusion = case == "supported"
    if expected_fusion:
        assert out is fusion.return_value
        assert fusion.call_args.args[-1] == 0
    else:
        fusion.assert_not_called()
        expected = [1.0, 5.0] if case == "causal" else ([1.0, 1.0] if case == "masked" else [5.0] * 4)
        torch.testing.assert_close(out.flatten(), torch.tensor(expected, dtype=torch.bfloat16))
