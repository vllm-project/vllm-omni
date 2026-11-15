# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.hy_nomas import fused_rope
from vllm_omni.diffusion.models.hy_nomas.layers import apply_rotary, rotary_tables
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="NVIDIA CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "transposed", "strided"])
@pytest.mark.parametrize("time_offset", [0, 16, 736])
@torch.no_grad()
def test_fused_rope_matches_fp32_reference(monkeypatch, dtype, layout, time_offset):
    torch.manual_seed(41)
    # Multiple batches/heads and a non-block-aligned element count exercise
    # indexing and the final masked Triton block (2 * 27 * 3 * 128).
    if layout == "transposed":
        x = torch.randn(2, 3, 27, 128, device="cuda", dtype=dtype).transpose(1, 2)
    elif layout == "strided":
        x = torch.randn(2, 27, 3, 258, device="cuda", dtype=dtype)[..., 1:257:2]
    else:
        x = torch.randn(2, 27, 3, 128, device="cuda", dtype=dtype)
    assert x.is_contiguous() == (layout == "contiguous")
    tables = rotary_tables((3, 3, 3), (32, 48, 48), 256.0, time_offset=time_offset, device=x.device)
    assert all(table.dtype == torch.float32 and table.device == x.device for table in tables)

    monkeypatch.setenv("VLLM_OMNI_HY_NOMAS_B3_ROPE_FUSION", "0")
    expected = apply_rotary(x, tables)
    actual = fused_rope.try_apply_rotary(x, tables)
    assert isinstance(actual, torch.Tensor), "The fused kernel must execute rather than fall back"
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))

    # Also verify the public opt-in path invokes the kernel, not eager fallback.
    original = fused_rope.try_apply_rotary
    calls = []

    def checked_rotary(source, reference_tables):
        result = original(source, reference_tables)
        assert isinstance(result, torch.Tensor)
        calls.append(result)
        return result

    monkeypatch.setattr(fused_rope, "try_apply_rotary", checked_rotary)
    monkeypatch.setenv("VLLM_OMNI_HY_NOMAS_B3_ROPE_FUSION", "1")
    dispatched = apply_rotary(x, tables)
    assert len(calls) == 1 and dispatched is calls[0]
    assert torch.equal(dispatched.view(torch.int16), expected.view(torch.int16))
