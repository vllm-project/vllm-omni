# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_omni.diffusion.distributed.qkv_a2a import eligible, qkv_fwd_batched

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_fake_dispatch_has_three_independent_correct_outputs():
    with FakeTensorMode():
        q = torch.empty(2, 17, 32, 128, device="cuda", dtype=torch.bfloat16)
        outputs = qkv_fwd_batched(q, q, q, "fake-group", 4)
        assert len(outputs) == 3
        assert all(x.shape == (2, 68, 8, 128) and x.dtype == q.dtype for x in outputs)
        assert outputs[0] is not outputs[1] and outputs[1] is not outputs[2]
        assert eligible(q, q, q, 4)
        assert not eligible(q, q[:, :, :16], q[:, :, :16], 4)
        assert not eligible(q, q, q, 3)


def test_cpu_dispatch_is_ineligible():
    q = torch.empty(1, 16, 32, 128, dtype=torch.bfloat16)
    assert not eligible(q, q, q, 4)
