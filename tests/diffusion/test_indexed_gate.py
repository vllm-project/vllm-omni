# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.layers.indexed_modulation import bf16_indexed_gate_add_
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cuda]


@pytest.mark.parametrize("batch,rows,heads,dim,tail", [(2, 17, 3, 32, 5), (1, 2998, 56, 128, 270)])
def test_indexed_gate_preserves_bf16_rounding_and_coarse_tail(batch, rows, heads, dim, tail):
    if not current_omni_platform.is_cuda():
        pytest.skip("CUDA BF16 arithmetic required")
    device = current_omni_platform.get_torch_device()
    generator = torch.Generator(device=device).manual_seed(27)
    bundle = torch.randn(batch, rows + tail, heads, dim, device=device, dtype=torch.bfloat16, generator=generator)
    gate = torch.randn(batch, rows, heads, dim, device=device, dtype=torch.bfloat16, generator=generator)
    indices = torch.arange(rows, device=device, dtype=torch.int32) % tail
    indices[::7] = -1
    before = bundle.clone()
    valid = indices >= 0
    expected = before[:, :rows].clone()
    expected[:, valid] += before[:, rows:].index_select(1, indices[valid].long()) * gate[:, valid]
    actual = bf16_indexed_gate_add_(bundle, gate, indices)
    assert torch.equal(actual, expected)
    assert torch.equal(bundle[:, rows:], before[:, rows:])
