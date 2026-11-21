# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The MLP takes the fused fc1 + SwiGLU kernel only for a bare unquantized fc1, never for a LoRA-wrapped one."""

import pytest
import torch
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

import vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks as blocks_mod
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3MLP

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _Linear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quant_method = UnquantizedLinearMethod()
        self.bias = None
        self.weight = torch.nn.Parameter(torch.zeros(256, 64), requires_grad=False)


class _LoRAWrapped(BaseLayerWithLoRA):
    def __init__(self, base_layer: torch.nn.Module) -> None:
        torch.nn.Module.__init__(self)
        self.base_layer = base_layer


def _mlp(fc1: torch.nn.Module) -> MiniMaxH3MLP:
    mlp = MiniMaxH3MLP.__new__(MiniMaxH3MLP)
    torch.nn.Module.__init__(mlp)
    mlp.fc1 = fc1
    return mlp


@pytest.fixture(autouse=True)
def _kernel_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(blocks_mod, "fc1_swiglu_supported", lambda x, weight: True)


def test_bare_unquantized_fc1_takes_the_fused_kernel() -> None:
    assert _mlp(_Linear())._fuse_fc1_swiglu(torch.zeros(2, 64))


def test_lora_wrapped_fc1_keeps_its_own_forward() -> None:
    assert not _mlp(_LoRAWrapped(_Linear()))._fuse_fc1_swiglu(torch.zeros(2, 64))
