# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib

import pytest
import torch
from pytest_mock import MockerFixture

from vllm_omni.diffusion.forward_context import ForwardContext
from vllm_omni.diffusion.layers import fused_qk_norm_rope

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _model(module, head_dim=128):
    return module.Flux2Transformer2DModel(
        in_channels=4,
        num_layers=0,
        num_single_layers=0,
        num_attention_heads=1,
        attention_head_dim=head_dim,
        joint_attention_dim=4,
        timestep_guidance_channels=16,
        axes_dims_rope=(32, 32, 32, 32),
    ).to(dtype=torch.bfloat16)


def _inputs(batch_size=1):
    return dict(
        hidden_states=torch.zeros(batch_size, 3, 4, dtype=torch.bfloat16),
        encoder_hidden_states=torch.zeros(batch_size, 2, 4, dtype=torch.bfloat16),
        timestep=torch.ones(batch_size, dtype=torch.bfloat16),
        img_ids=torch.zeros(3, 4),
        txt_ids=torch.zeros(2, 4),
        return_dict=False,
    )


@pytest.mark.parametrize("model_name", ["flux2", "flux2_klein"])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_model_forward_gates_packed_rope_on_actual_head_dim(
    model_name: str, head_dim: int, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partial RoPE must keep the model outside the CUDA RMSNorm fusion contract."""
    monkeypatch.delenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", raising=False)
    module = importlib.import_module(f"vllm_omni.diffusion.models.{model_name}.{model_name}_transformer")
    model = _model(module, head_dim)
    mocker.patch.object(module, "get_forward_context", return_value=ForwardContext())
    # Exercise the real geometry gate on CPU instead of rejecting the device first.
    mocker.patch.object(fused_qk_norm_rope, "fused_qk_norm_rope_available", return_value=True)
    pack = mocker.spy(module, "pack_qk_norm_rope_table")

    model(**_inputs())

    pack.assert_called_once()
    if head_dim == 256:
        assert pack.spy_return is None
    else:
        assert pack.spy_return is not None
        assert pack.spy_return.shape == (5, 128)


class _CaptureBlock(torch.nn.Module):
    """Observe model-to-block table hand-off without running CUDA attention."""

    def __init__(self):
        super().__init__()
        self.attention_kwargs = None

    def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
        self.attention_kwargs = kwargs["joint_attention_kwargs"]
        if encoder_hidden_states is not None:
            return encoder_hidden_states, hidden_states
        return hidden_states


@pytest.mark.parametrize("model_name", ["flux2", "flux2_klein"])
@pytest.mark.parametrize(
    "batch_size,min_tokens,expect_packed",
    [
        (1, None, True),
        (1, "", True),
        (1, "0", True),
        (1, "5", True),
        (1, "6", False),
        (2, "6", True),
        (2, "10", True),
        (2, "11", False),
    ],
)
def test_model_forward_honors_joint_token_threshold(
    model_name: str,
    batch_size: int,
    min_tokens: str | None,
    expect_packed: bool,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate on B*(text+image) tokens and never pass a stale table to blocks."""
    env = "VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS"
    if min_tokens is None:
        monkeypatch.delenv(env, raising=False)
    else:
        monkeypatch.setenv(env, min_tokens)
    module = importlib.import_module(f"vllm_omni.diffusion.models.{model_name}.{model_name}_transformer")
    model = _model(module)
    double_block, single_block = _CaptureBlock(), _CaptureBlock()
    model.transformer_blocks.append(double_block)
    model.single_transformer_blocks.append(single_block)
    mocker.patch.object(module, "get_forward_context", return_value=ForwardContext())
    # Keep the real packer/geometry checks; only bypass the CUDA device check.
    mocker.patch.object(fused_qk_norm_rope, "fused_qk_norm_rope_available", return_value=True)
    pack = mocker.spy(module, "pack_qk_norm_rope_table")
    stale_table = torch.empty(1)
    caller_kwargs = {fused_qk_norm_rope.QK_NORM_ROPE_TABLE_KEY: stale_table, "unrelated": "retained"}

    model(**_inputs(batch_size), joint_attention_kwargs=caller_kwargs)

    assert caller_kwargs[fused_qk_norm_rope.QK_NORM_ROPE_TABLE_KEY] is stale_table
    assert double_block.attention_kwargs is single_block.attention_kwargs
    block_kwargs = double_block.attention_kwargs
    assert block_kwargs is not None
    assert block_kwargs is not caller_kwargs
    assert block_kwargs["unrelated"] == "retained"
    if expect_packed:
        pack.assert_called_once()
        table = block_kwargs[fused_qk_norm_rope.QK_NORM_ROPE_TABLE_KEY]
        assert table is pack.spy_return
        assert table.shape == (batch_size * 5, 128)
    else:
        pack.assert_not_called()
        assert fused_qk_norm_rope.QK_NORM_ROPE_TABLE_KEY not in block_kwargs
