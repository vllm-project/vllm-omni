# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends.utils import rocm_h3_fmha as native
from vllm_omni.diffusion.attention.backends.utils.rocm_h3_bounds import packed_cu_seqlens

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


@hardware_test(res={"rocm": "MI350"}, num_cards=1)
@torch.inference_mode()
def test_grouped_fmha_on_strided_receive():
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("ROCm is required")
    device = torch.device("cuda", torch.accelerator.current_device_index())
    if torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] != "gfx950":
        pytest.skip("requires gfx950")
    if native._code_object() is None:
        pytest.skip("installed AITER has no qualified gfx950 grouped kernel")
    from aiter import flash_attn_varlen_func

    cut, rows, heads, dim = 37749, 37760, 14, 128
    generator = torch.Generator(device=device).manual_seed(17)
    packet = torch.randn((1, rows, 3, heads, dim), generator=generator, dtype=torch.bfloat16, device=device)
    q, k, v = packet.unbind(2)
    assert q.stride(1) == 3 * heads * dim
    cu = packed_cu_seqlens((0, cut, rows), device)

    result = native.try_grouped_fmha(q, k, v, cu, cu, max_seqlen=cut)
    assert result is not None and result.is_contiguous()
    dense = [x.squeeze(0).contiguous() for x in (q, k, v)]
    ref = flash_attn_varlen_func(*dense, cu, cu, cut, cut, softmax_scale=dim**-0.5, causal=False)
    assert torch.equal(result.squeeze(0).view(torch.int16), ref.view(torch.int16))
