# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.npu]


@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires Ascend NPU")
def test_qwen3_vl_npu_mlp_uses_projection_module() -> None:
    from vllm_omni.platforms.npu.models.minimax_h3 import (
        _forward_minimax_h3_qwen3vl_text_mlp_npu,
    )

    x = Mock()
    gate_up = Mock()
    activated = Mock()
    output = Mock()
    mlp = SimpleNamespace(
        gate_up_proj=Mock(return_value=gate_up),
        down_proj=Mock(return_value=output),
    )

    with patch(
        "vllm_omni.platforms.npu.models.minimax_h3.npu_swiglu_from_packed",
        return_value=activated,
    ) as swiglu:
        result = _forward_minimax_h3_qwen3vl_text_mlp_npu(mlp, x)

    mlp.gate_up_proj.assert_called_once_with(x)
    swiglu.assert_called_once_with(gate_up)
    mlp.down_proj.assert_called_once_with(activated)
    assert result is output


@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires Ascend NPU")
@pytest.mark.parametrize("shape", [(1, 16, 256), (2, 1, 256)])
def test_qwen3_vl_npu_swiglu_matches_split_reference(
    shape: tuple[int, int, int],
) -> None:
    from vllm_omni.platforms.npu.models.minimax_h3 import npu_swiglu_from_packed

    packed = torch.randn(*shape, device="npu", dtype=torch.bfloat16)
    gate, up = packed.chunk(2, dim=-1)

    torch.testing.assert_close(
        npu_swiglu_from_packed(packed),
        F.silu(gate) * up,
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires Ascend NPU")
def test_qwen3_vl_npu_swiglu_matches_two_projection_reference() -> None:
    from vllm_omni.platforms.npu.models.minimax_h3 import npu_swiglu_from_packed

    x = torch.randn(1, 16, 128, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(256, 128, device="npu", dtype=torch.bfloat16)
    packed = F.linear(x, weight)
    gate = F.linear(x, weight[:128])
    up = F.linear(x, weight[128:])

    torch.testing.assert_close(
        npu_swiglu_from_packed(packed),
        F.silu(gate) * up,
        atol=2e-2,
        rtol=2e-2,
    )
