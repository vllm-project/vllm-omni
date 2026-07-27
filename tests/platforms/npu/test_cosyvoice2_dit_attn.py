# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for CosyVoice2 DiT Ascend patches."""

import pytest
import torch

from vllm_omni.platforms.npu.models import cosyvoice2_dit_attn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_fused_residual_matches_unfused_formula() -> None:
    x = torch.randn(2, 5, 8)
    gate = torch.randn(2, 1, 8)
    branch = torch.randn_like(x)

    expected = x + gate * branch
    actual = cosyvoice2_dit_attn._fused_residual(x, gate, branch)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("value", ["0", "false", "OFF", "no"])
def test_fused_gated_residual_switch_disables_known_false_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("VLLM_OMNI_MINICPMO_FUSED_GATED_RESIDUAL", value)
    assert not cosyvoice2_dit_attn._fused_gated_residual_enabled()
