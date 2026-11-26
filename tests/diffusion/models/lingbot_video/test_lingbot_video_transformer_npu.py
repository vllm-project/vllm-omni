# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""NPU regression tests for LingBot-Video RoPE indexing.

The complex64 RoPE table used to be gathered with advanced indexing, which
lowers to ``aclnnIndex`` on Ascend NPU and rejects complex64 self tensors
(error 161002). Position ids were also built as int32, which ``aclnnMaxDim``
rejects on NPU. These tests run on NPU, compare the RoPE gather against a CPU
reference, and exercise a tiny transformer forward end to end.
"""

from __future__ import annotations

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.lingbot_video.lingbot_video_transformer import (
    LingBotVideoRotaryEmbedding,
    make_joint_position_ids,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

DEVICE = "npu"


def _require_npu() -> None:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU is not available")


def _reference_rope(rope: LingBotVideoRotaryEmbedding, position_ids: torch.Tensor) -> torch.Tensor:
    """CPU reference using the same complex64 tables and advanced indexing."""
    freqs_cis = [freqs.cpu() for freqs in rope.freqs_cis]
    pos = position_ids.cpu()
    return torch.cat([freqs_cis[i][pos[:, i]] for i in range(len(rope.axes_dims))], dim=-1)


@hardware_test(res={"npu": "A3"}, num_cards=1)
def test_rope_gather_matches_cpu_reference() -> None:
    _require_npu()
    rope = LingBotVideoRotaryEmbedding((4, 4, 8), (32, 32, 32), 10000.0)
    position_ids = make_joint_position_ids(3, 1, 2, 2, torch.device(DEVICE))

    # aclnnMaxDim rejects int32 on NPU; position ids must be int64.
    assert position_ids.dtype == torch.int64

    actual = rope(position_ids)
    expected = _reference_rope(rope, position_ids)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@hardware_test(res={"npu": "A3"}, num_cards=1)
def test_tiny_transformer_forward() -> None:
    _require_npu()
    from vllm_omni.diffusion.models.lingbot_video import LingBotVideoTransformer3DModel

    model = (
        LingBotVideoTransformer3DModel(
            patch_size=(1, 1, 1),
            in_channels=2,
            out_channels=2,
            hidden_size=16,
            num_attention_heads=1,
            depth=0,
            intermediate_size=32,
            text_dim=8,
            freq_dim=8,
            axes_dims=(4, 4, 8),
            axes_lens=(32, 32, 32),
        )
        .to(DEVICE)
        .eval()
    )

    hidden_states = torch.randn(1, 2, 1, 2, 2, device=DEVICE)
    timestep = torch.tensor([300.0], device=DEVICE)
    encoder_hidden_states = torch.randn(1, 3, 8, device=DEVICE)
    encoder_attention_mask = torch.ones(1, 3, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        output = model(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )[0]

    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()