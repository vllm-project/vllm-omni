# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MAGI-2's opt-in Ascend clipped-SwiGLU7, driven through the model's own entry point."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch_npu")

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.layers.swiglu7 import SwiGLU7
from vllm_omni.diffusion.models.magi2 import common_layers

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


def reference(x, alpha=1.702, limit=7.0, out_dtype=None):
    gate = x[..., ::2].float().clamp(max=limit)
    up = x[..., 1::2].float().clamp(min=-limit, max=limit)
    return (gate * torch.sigmoid(alpha * gate) * (up + 1)).to(x.dtype if out_dtype is None else out_dtype)


@hardware_test(res={"npu": "A2"}, num_cards=1)
def test_npu_fused_swiglu7_is_opt_in(monkeypatch):
    """The switch must reach MAGI-2's activation and must not reach the shared operator."""

    import torch_npu

    x = torch.tensor([[9, -9, 1, 4, -2, 8, 0, 0]] * 32, device="npu", dtype=torch.bfloat16)
    calls = []
    kernel = torch_npu.npu_clipped_swiglu

    def record(tensor, **kwargs):
        calls.append(kwargs["interleaved"])
        return kernel(tensor, **kwargs)

    monkeypatch.setattr(torch_npu, "npu_clipped_swiglu", record)

    monkeypatch.delenv("MAGI2_FUSED_SWIGLU7", raising=False)
    torch.testing.assert_close(common_layers.swiglu7(x), reference(x), rtol=0, atol=0)
    assert calls == []

    monkeypatch.setenv("MAGI2_FUSED_SWIGLU7", "1")
    torch.testing.assert_close(common_layers.swiglu7(x), reference(x), rtol=8e-3, atol=2e-3)
    assert calls == [True], "the dense and shared-expert activation must reach the fused kernel"

    torch.testing.assert_close(SwiGLU7()(x), reference(x), rtol=0, atol=0)
    assert calls == [True], "the shared operator must stay on the native formula"

    # A wider output dtype and a non-contiguous input stay native with the switch on.
    torch.testing.assert_close(
        common_layers.swiglu7(x, out_dtype=torch.float32), reference(x, out_dtype=torch.float32), rtol=0, atol=0
    )
    torch.testing.assert_close(common_layers.swiglu7(x[:, ::2]), reference(x[:, ::2]), rtol=0, atol=0)
    assert calls == [True]
