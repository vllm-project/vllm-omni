# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.qwen_image_21.ops.gated_norm import apply_gated_norm_modulation
from vllm_omni.diffusion.models.qwen_image_21.ops.modulation import select_modulation_rows

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cuda, pytest.mark.gpu]


@pytest.mark.parametrize("batch,seq", [(1, 37), (2, 256), (1, 4097)])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("prepared", [False, True])
def test_compiled_gated_norm_precision(batch, seq, masked, prepared):
    torch.manual_seed(7945)
    residual = torch.randn(batch, seq, 4096, device="cuda", dtype=torch.bfloat16)
    sublayer = torch.randn_like(residual)
    params = torch.randn(batch + int(masked), 16384, device="cuda", dtype=torch.bfloat16)
    gate, scale = params[:, 4096:8192], params[:, 8192:12288]
    mask = torch.arange(seq, device="cuda") >= seq // 3 if masked else None
    if prepared:
        gate, scale = gate.tanh(), 1 + scale
    selected_gate = select_modulation_rows(gate, mask)
    selected_scale = select_modulation_rows(scale, mask)
    hidden = residual + (selected_gate if prepared else selected_gate.tanh()) * sublayer
    expected = F.layer_norm(hidden, (4096,), eps=1e-6) * (selected_scale if prepared else 1 + selected_scale)
    norm = torch.nn.LayerNorm(4096, elementwise_affine=False, eps=1e-6)
    args = (residual, sublayer, gate, scale, mask, norm, prepared)

    # Eager retains the original operations exactly; compiled normalization has
    # an FP32 norm/scale epilogue, bounded by BF16 precision rather than bit equality.
    eager = apply_gated_norm_modulation(*args)
    torch.testing.assert_close(eager, (hidden, expected), rtol=0, atol=0)
    compiled = torch.compile(apply_gated_norm_modulation, fullgraph=True, dynamic=True)
    actual_hidden, actual_norm = compiled(*args)
    torch.testing.assert_close(actual_hidden, hidden, rtol=0, atol=0)
    torch.testing.assert_close(actual_norm, expected, rtol=torch.finfo(torch.bfloat16).eps, atol=2e-5)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = compiled(*args)
    graph.replay()
    torch.testing.assert_close(captured, (actual_hidden, actual_norm), rtol=0, atol=0)
    previous_hidden = captured[0].clone()
    residual.add_(0.25)
    gate.mul_(0.75)
    scale.add_(0.25)
    if mask is not None:
        mask.logical_not_()
    changed = compiled(*args)
    graph.replay()
    assert not torch.equal(captured[0], previous_hidden)
    torch.testing.assert_close(captured, changed, rtol=0, atol=0)
