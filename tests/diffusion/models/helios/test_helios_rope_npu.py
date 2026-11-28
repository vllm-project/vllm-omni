# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.layers import rope as rope_module
from vllm_omni.diffusion.models.helios import helios_transformer as helios_module
from vllm_omni.diffusion.models.helios.helios_transformer import HeliosRotaryEmbedding

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


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


# Helios reaches this layer at four sequence lengths over a pyramid
# generation. 2940 is the TP1 stage-0 shape; 11776 is the largest.
@hardware_test(res={"npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("seq_len", [2940, 11776])
def test_helios_rope_npu_matches_reference(seq_len: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """The fused NPU kernel must reproduce the portable path exactly."""
    torch.manual_seed(1101)
    hidden_states = torch.randn(1, seq_len, 40, 128, dtype=torch.bfloat16, device="npu")
    angles = torch.randn(1, seq_len, 64, dtype=torch.float32, device="npu")
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
    # change elsewhere would silently leave the comparison below checking the
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

    # ``npu_rotary_mul`` consumes the frequency table at the activation dtype,
    # so the adapter rotates in fp32 to keep this fp32 table exact. The result
    # is then bit-identical to the portable implementation, which pins the
    # fused arithmetic with no tolerance at all.
    torch.testing.assert_close(actual, _reference(hidden_states, freqs_cis), atol=0, rtol=0)
