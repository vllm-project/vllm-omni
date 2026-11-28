# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Pin FIA output normalization inherited from the measured H3 checkpoint."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.diffusion.models.minimax_h3.test_minimax_h3_vae_npu_fia import _FakeDecoder, _NPUTensor
from vllm_omni.diffusion.models.minimax_h3 import vae_npu_fia

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fia_preserves_checkpoint_nonfinite_normalization(monkeypatch, dtype):
    raw_output = torch.arange(24, dtype=dtype).reshape(1, 3, 2, 4)
    raw_output.flatten()[:3] = torch.tensor([float("nan"), float("inf"), -float("inf")], dtype=dtype)
    operator = Mock(return_value=(raw_output, None))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_fused_infer_attention_score=operator))
    device = Mock(spec=torch.device, type="npu")
    decoder = _FakeDecoder()
    vae_npu_fia.install_vae_npu_fia(decoder, device, Mock(), enabled=True)
    query = _NPUTensor(torch.zeros(1, 2, 3, 4, dtype=dtype), device)

    actual = decoder.transformer_blocks[0].attn.perform_attention(query, query, query, {})

    expected = torch.arange(24, dtype=dtype).reshape(1, 3, 2, 4)
    expected.flatten()[:3] = torch.tensor([0, torch.finfo(dtype).max, torch.finfo(dtype).min], dtype=dtype)
    assert torch.equal(actual, expected.transpose(1, 2))
    assert actual.dtype == dtype
    operator.assert_called_once()
    assert decoder._h3_npu_fia_stats == {"calls": 1, "fallbacks": 0}
