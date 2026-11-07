# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib

import pytest
import torch
from pytest_mock import MockerFixture

from vllm_omni.diffusion.forward_context import ForwardContext
from vllm_omni.diffusion.layers import fused_qk_norm_rope

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("model_name", ["flux2", "flux2_klein"])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_model_forward_gates_packed_rope_on_actual_head_dim(
    model_name: str, head_dim: int, mocker: MockerFixture
) -> None:
    """Partial RoPE must keep the model outside the CUDA RMSNorm fusion contract."""
    module = importlib.import_module(f"vllm_omni.diffusion.models.{model_name}.{model_name}_transformer")
    model = module.Flux2Transformer2DModel(
        in_channels=4,
        num_layers=0,
        num_single_layers=0,
        num_attention_heads=1,
        attention_head_dim=head_dim,
        joint_attention_dim=4,
        timestep_guidance_channels=16,
        axes_dims_rope=(32, 32, 32, 32),
    ).to(dtype=torch.bfloat16)
    mocker.patch.object(module, "get_forward_context", return_value=ForwardContext())
    # Exercise the real geometry gate on CPU instead of rejecting the device first.
    mocker.patch.object(fused_qk_norm_rope, "fused_qk_norm_rope_available", return_value=True)
    pack = mocker.spy(module, "pack_qk_norm_rope_table")

    model(
        hidden_states=torch.zeros(1, 3, 4, dtype=torch.bfloat16),
        encoder_hidden_states=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
        timestep=torch.ones(1, dtype=torch.bfloat16),
        img_ids=torch.zeros(3, 4),
        txt_ids=torch.zeros(2, 4),
        return_dict=False,
    )

    pack.assert_called_once()
    if head_dim == 256:
        assert pack.spy_return is None
    else:
        assert pack.spy_return is not None
        assert pack.spy_return.shape == (5, 128)
