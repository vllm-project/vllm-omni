# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.diffusion.layers import rope as rope_module
from vllm_omni.diffusion.layers.rope import RotaryEmbedding

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_rotary_embedding_npu_mindie_accepts_3d_input(monkeypatch: pytest.MonkeyPatch) -> None:
    rotary = RotaryEmbedding(is_neox_style=True)
    rotary.has_mindie = True

    seen: dict[str, torch.Size] = {}

    def fake_mindie(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        interleaved: bool = False,
        half_head_dim: bool = True,
    ) -> torch.Tensor:
        seen["x_shape"] = x.shape
        seen["cos_shape"] = cos.shape
        seen["sin_shape"] = sin.shape
        assert not interleaved
        assert half_head_dim
        return x + 1

    monkeypatch.setattr(rope_module, "apply_rotary_emb_mindiesd", fake_mindie)

    x = torch.zeros(8, 4, 64)
    cos = torch.ones(8, 32)
    sin = torch.zeros(8, 32)

    out = rotary.forward_npu(x, cos, sin)

    assert seen["x_shape"] == torch.Size([1, 8, 4, 64])
    assert seen["cos_shape"] == torch.Size([8, 32])
    assert seen["sin_shape"] == torch.Size([8, 32])
    assert out.shape == x.shape
    torch.testing.assert_close(out, torch.ones_like(x))
