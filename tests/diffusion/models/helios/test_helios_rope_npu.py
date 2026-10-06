# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.layers import rope as rope_module
from vllm_omni.diffusion.models.helios import helios_transformer as helios_module
from vllm_omni.diffusion.models.helios.helios_transformer import HeliosRotaryEmbedding

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

# bfloat16 keeps eight significand bits, so one unit in the last place of an
# element is 2**-7 of its magnitude.
BF16_ULP = 2.0**-7


def _reference(hidden_states: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_1, x_2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
    return (
        torch.stack(
            (
                x_1 * cos[..., 0::2] - x_2 * sin[..., 1::2],
                x_1 * sin[..., 1::2] + x_2 * cos[..., 0::2],
            ),
            dim=-1,
        )
        .flatten(-2, -1)
        .type_as(hidden_states)
    )


@hardware_test(res={"npu": "A3"}, num_cards=1)
def test_helios_rope_npu_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the fused NPU kernel's arithmetic at Helios's TP1 stage-0 shape."""
    torch.manual_seed(1101)
    hidden_states = torch.randn(1, 2940, 40, 128, dtype=torch.bfloat16, device="npu")
    angles = torch.randn(1, 2940, 64, dtype=torch.float32, device="npu")
    freqs_cis = torch.cat(
        (
            angles.cos().repeat_interleave(2, dim=-1),
            angles.sin().repeat_interleave(2, dim=-1),
        ),
        dim=-1,
    )
    rope = HeliosRotaryEmbedding()

    assert helios_module._is_validated_ascend_a3_device(hidden_states.device)
    assert rope._can_use_npu_impl(hidden_states, freqs_cis)

    # Count the fused call rather than assume it. Without this, a dispatch
    # change elsewhere would silently leave the comparisons below checking the
    # portable implementation against itself.
    fused_calls = 0
    fused = rope_module.apply_rotary_emb_torch_npu

    def counting_fused(*args, **kwargs):
        nonlocal fused_calls
        fused_calls += 1
        return fused(*args, **kwargs)

    monkeypatch.setattr(rope_module, "apply_rotary_emb_torch_npu", counting_fused)
    actual = rope(hidden_states, freqs_cis)
    assert fused_calls == 1
    assert actual.shape == hidden_states.shape
    assert actual.dtype == hidden_states.dtype

    # ``npu_rotary_mul`` takes the frequency table at the activation dtype and
    # accumulates in fp32. Given the same bfloat16 table, the portable
    # implementation reproduces it exactly, which pins the fused arithmetic
    # with no tolerance at all.
    torch.testing.assert_close(
        actual,
        _reference(hidden_states, freqs_cis.bfloat16().float()),
        atol=0,
        rtol=0,
    )

    # Against the full-precision table that rounding is the whole difference.
    # Bound it in absolute terms: elements where the two rotary terms nearly
    # cancel differ by more than one step of their own magnitude, so a relative
    # bound would say nothing about the error that matters.
    expected = _reference(hidden_states, freqs_cis)
    scale = expected.float().abs().amax()
    assert (actual.float() - expected.float()).abs().amax() <= BF16_ULP * scale
