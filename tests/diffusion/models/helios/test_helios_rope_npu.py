# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
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


@hardware_test(res={"npu": "A3"}, num_cards=1)
def test_helios_rope_npu_matches_reference() -> None:
    """Exercise MindIE RoPE at Helios's TP1 stage-0 shape and mixed dtypes."""
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

    if not rope.impl.has_mindie:
        pytest.skip("MindIE is not installed")
    assert helios_module._is_validated_ascend_a3_device(hidden_states.device)
    assert rope._can_use_npu_impl(hidden_states, freqs_cis)
    actual = rope(hidden_states, freqs_cis)
    expected = _reference(hidden_states, freqs_cis)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
