# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed QAD sharding against an unsliced FlashInfer projection."""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization import NVFP4Projection, _Projection

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": ["B200", "RTX5090"]})]


@pytest.mark.parametrize("axis,swap", [(0, False), (0, True), (1, False)])
@torch.inference_mode()
def test_nvfp4_parallel_shards_match_unsliced_projection(axis, swap):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("requires Blackwell NVFP4")
    flashinfer = pytest.importorskip("flashinfer")
    torch.manual_seed(43)
    n, k = 512, 256
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(17, k, device="cuda", dtype=torch.bfloat16)
    global_scale = torch.tensor([32.0], device="cuda")
    input_scale = torch.tensor([2.0], device="cuda")
    packed, scales = flashinfer.nvfp4_quantize(
        weight, global_scale, sfLayout=flashinfer.SfLayout.layout_128x4, enable_pdl=False
    )
    a, a_scales = flashinfer.nvfp4_quantize(x, input_scale, sfLayout=flashinfer.SfLayout.layout_128x4, enable_pdl=False)
    expected = flashinfer.mm_fp4(
        a,
        packed.T,
        a_scales,
        scales.T,
        (global_scale * input_scale).reciprocal(),
        out_dtype=torch.bfloat16,
        block_size=16,
        use_8x4_sf_layout=False,
        backend="cutlass",
        enable_pdl=False,
    )
    spec = NVFP4Projection("test", n, k, {"weight": "w", "scales": "s", "alpha": "a", "input_scale": "x"}, swap)
    outputs = []
    for rank in range(2):
        layer = _Projection(spec, n // 2 if axis == 0 else n, k // 2 if axis == 1 else k, rank)
        for field, value in {
            "weight": packed.cpu(),
            "scales": scales.cpu(),
            "alpha": global_scale.reciprocal().cpu(),
            "input_scale": input_scale.cpu(),
        }.items():
            parameter = getattr(layer, field)
            parameter.weight_loader(parameter, value)
        layer = layer.cuda()
        outputs.append(layer(x if axis == 0 else x.chunk(2, dim=-1)[rank]))
    if axis == 1:
        actual = sum(output.float() for output in outputs).to(torch.bfloat16)
    elif swap:
        halves = [output.chunk(2, dim=-1) for output in outputs]
        actual = torch.cat([halves[rank][half] for half in (0, 1) for rank in (0, 1)], dim=-1)
        expected = torch.cat(expected.chunk(2, dim=-1)[::-1], dim=-1)
    else:
        actual = torch.cat(outputs, dim=-1)
    # Row-parallel accumulation rounds each shard once before the reduction.
    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative_l2.item() < 0.004
