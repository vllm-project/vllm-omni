# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.common.ming.dit_blocks import FeedForward

pytestmark = [pytest.mark.core_model, pytest.mark.tts]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("approximate", ["tanh", "none"])
@torch.inference_mode()
def test_feed_forward_fused_gelu_matches_module_path(approximate):
    torch.manual_seed(0)
    module = FeedForward(64, mult=4, approximate=approximate).cuda().eval()
    x = torch.randn(4, 3, 64, device="cuda")
    expected = module(x)
    module.fused_gelu = True
    # Only the tanh GELU has a GEMM epilogue; others keep the module path.
    torch.testing.assert_close(module(x), expected, atol=1e-4, rtol=1e-4)
