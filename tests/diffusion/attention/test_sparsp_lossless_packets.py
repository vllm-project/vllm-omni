# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.attention.sparsp.kernels import kv_block_bytes, pack_kv

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": ["L4", "H100"]})]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sparsp_packs_original_bits_from_strided_kv(dtype):
    device = torch.device("cuda")
    key_bits = torch.arange(1 << 16, device=device, dtype=torch.int32).to(torch.int16).reshape(1, 256, 4, 64)
    value_bits = key_bits.flip((1, 3))
    key = key_bits.view(dtype).transpose(1, 2)
    value = value_bits.view(dtype).transpose(1, 2)
    heads = torch.tensor([0, 0, 2, 3], dtype=torch.int32, device=device)
    blocks = torch.tensor([0, 2, 1, 3], dtype=torch.int32, device=device)
    wire = torch.empty(heads.numel() * kv_block_bytes(64), dtype=torch.uint8, device=device)

    pack_kv(key, value, heads, blocks, wire)

    tokens = blocks[:, None] * 64 + torch.arange(64, device=device)
    expected = torch.stack((key_bits[0, tokens, heads[:, None]], value_bits[0, tokens, heads[:, None]]), dim=1)
    assert torch.equal(wire.view(torch.int16).reshape_as(expected), expected)
