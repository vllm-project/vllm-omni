# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The packed causal conv kernel must match the padded cuDNN position embedding."""

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA")]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("lengths", [[1], [31, 30, 32], [500, 7, 129, 64]])
@torch.inference_mode()
def test_packed_causal_conv_mish_matches_padded_cudnn(lengths):
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_conv import (
        pack_conv_weight,
        packed_causal_conv_mish,
    )
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_dit import (
        gather_rows,
        pack_rows,
        scatter_rows,
    )

    torch.manual_seed(0)
    channels, groups, taps = 256, 16, 31
    conv = torch.nn.Conv1d(channels, channels, taps, groups=groups).cuda().to(torch.bfloat16)
    rows = pack_rows(lengths, torch.device("cuda"))
    x = torch.randn(rows.total, channels, device="cuda", dtype=torch.bfloat16)

    padded = scatter_rows(x.unsqueeze(0), rows, rows.width).permute(0, 2, 1)
    reference = F.mish(conv(F.pad(padded, (taps - 1, 0))))
    reference = gather_rows(reference.permute(0, 2, 1), rows)[0]
    output = packed_causal_conv_mish(x, pack_conv_weight(conv), conv.bias, rows.positions)

    # Rows never mix and match the left-zero-padded convolution up to bf16 rounding.
    torch.testing.assert_close(output.float(), reference.float(), atol=0.04, rtol=0.02)
