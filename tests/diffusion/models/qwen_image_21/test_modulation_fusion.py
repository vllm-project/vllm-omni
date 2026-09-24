# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.qwen_image_21 import qwen_image_21_transformer as transformer
from vllm_omni.diffusion.models.qwen_image_21.ops.modulation import (
    apply_gated_residual,
    apply_modulation,
    select_modulation_rows,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

CHANNELS = 4096


def _single_rank(monkeypatch):
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )


def _eager_modulate(x, params, mask):
    return x * (1 + select_modulation_rows(params, mask))


def _eager_gated_residual(residual, sublayer, params, mask):
    return residual + select_modulation_rows(params, mask).tanh() * sublayer


@pytest.mark.parametrize(
    "device",
    [pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=[pytest.mark.cuda, pytest.mark.gpu])],
)
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("seq,prefix", [(512, 0), (256, 64), (32, 30)])
def test_fused_modulation_matches_eager_pointwise(device, batch, seq, prefix):
    torch.manual_seed(0)
    channels = 1024 if device == "cpu" else CHANNELS
    # scale and gate arrive as two non-contiguous halves of one `[rows, 2 * channels]` tensor.
    packed = torch.randn(batch + 1, 2 * channels, device=device, dtype=torch.bfloat16)
    packed[-1] = packed[-1] * 8.0 + 4.0
    mask = torch.zeros(seq, dtype=torch.bool, device=device)
    mask[prefix:] = True
    x = (torch.randn(batch, seq, channels, device=device) * 2.0).to(torch.bfloat16)
    sublayer = (torch.randn(batch, seq, channels, device=device) * 2.0).to(torch.bfloat16)
    residual = (torch.randn(batch, seq, channels, device=device) * 2.0).to(torch.bfloat16)

    for params in (packed[:, :channels], packed[:, channels:]):
        assert not params.is_contiguous()
        torch.testing.assert_close(
            apply_modulation(x, params, mask), _eager_modulate(x, params, mask), rtol=0, atol=0
        )
        torch.testing.assert_close(
            apply_gated_residual(residual, sublayer, params, mask),
            _eager_gated_residual(residual, sublayer, params, mask),
            rtol=0,
            atol=0,
        )

    # Without `causal_condition` there is no t=0 row and every sample reads its own row.
    own = packed[:batch, :channels].contiguous()
    torch.testing.assert_close(apply_modulation(x, own, None), x * (1 + own.unsqueeze(1)), rtol=0, atol=0)
    torch.testing.assert_close(
        apply_gated_residual(residual, sublayer, own, None),
        residual + own.unsqueeze(1).tanh() * sublayer,
        rtol=0,
        atol=0,
    )


@pytest.mark.cuda
@pytest.mark.gpu
def test_t_zero_row_only_reaches_unmasked_tokens():
    torch.manual_seed(0)
    batch, seq = 2, 64
    params = torch.randn(batch + 1, CHANNELS, device="cuda", dtype=torch.bfloat16)
    # Make the t=0 row unmistakable, then check the split point exactly.
    params[-1] = 100.0
    x = torch.ones(batch, seq, CHANNELS, device="cuda", dtype=torch.bfloat16)
    mask = torch.zeros(seq, dtype=torch.bool, device="cuda")
    mask[10:] = True
    out = apply_modulation(x, params, mask)
    torch.testing.assert_close(out[:, :10], torch.full_like(out[:, :10], 101.0), rtol=0, atol=0)
    for row in range(batch):
        torch.testing.assert_close(
            out[row, 10:], x[row, 10:] * (1 + params[row].unsqueeze(0)), rtol=0, atol=0
        )


@pytest.mark.cuda
@pytest.mark.gpu
def test_fused_modulation_falls_back_for_ineligible_inputs():
    torch.manual_seed(0)
    batch, seq, channels = 2, 32, 1024
    params = torch.randn(batch + 1, channels, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(seq, dtype=torch.bool, device="cuda")
    x = (torch.randn(batch, seq, channels, device="cuda") * 2.0).to(torch.bfloat16)
    # FP16 is not eligible and must stay on the eager chain.
    half = x.half()
    torch.testing.assert_close(
        apply_modulation(half, params.half(), mask), _eager_modulate(half, params.half(), mask), rtol=0, atol=0
    )
    # A parameter count that matches neither mode is rejected rather than misread.
    with pytest.raises(ValueError):
        apply_modulation(x, torch.randn(batch + 3, channels, device="cuda", dtype=torch.bfloat16), mask)


@pytest.mark.cuda
@pytest.mark.gpu
def test_block_forward_is_unchanged_by_modulation_fusion(monkeypatch):
    _single_rank(monkeypatch)
    torch.manual_seed(0)
    block = transformer.QwenImage21TransformerBlock(
        dim=512, num_attention_heads=4, attention_head_dim=128, mlp_ratio=3, eps=1e-6
    ).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.05)

    recorded = {}

    class _Recorder(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, hidden, freqs, **kwargs):
            recorded["input"] = hidden
            return self.inner(hidden, freqs, **kwargs)

    block.attn = _Recorder(block.attn)
    batch, seq = 2, 48
    hidden = (torch.randn(batch, seq, 512, device="cuda") * 2.0).to(torch.bfloat16)
    modulation = (torch.randn(batch + 1, 4 * 512, device="cuda") * 0.2).to(torch.bfloat16)
    freqs = torch.polar(
        torch.ones(seq, 64, device="cuda"),
        torch.rand(seq, 64, device="cuda") * 6.0,
    )
    mask = torch.zeros(seq, dtype=torch.bool, device="cuda")
    mask[16:] = True

    block(hidden_states=hidden, modulation=modulation, freqs=freqs, target_token_mask=mask)
    fused_attn_input = recorded["input"]

    mod1, mod2 = modulation.chunk(2, dim=-1)
    want_attn_input = _eager_modulate(block.img_norm1(hidden), mod1[:, :512], mask)
    torch.testing.assert_close(fused_attn_input, want_attn_input, rtol=0, atol=0)
