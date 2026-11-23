# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused NPU top-k mask and Gumbel argmax, retaining ATen log semantics."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _select(
    scores,
    noise,
    thresholds,
    output,
    width: tl.constexpr,
    block: tl.constexpr,
    threshold_stride: tl.constexpr,
    use_threshold: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.arange(0, block)
    x = tl.load(scores + row * width + idx, idx < width, other=-float("inf")).to(tl.float32)
    if use_threshold:
        threshold = tl.load(thresholds + row * threshold_stride)
        x = tl.where(x < threshold, -float("inf"), x)
    z = tl.load(noise + row * width + idx, idx < width, other=0)
    value = tl.where(idx < width, x - z, -float("inf"))
    maximum = tl.max(value, 0)
    first = tl.min(tl.where((idx < width) & (value == maximum), idx, block), 0)
    first_nan = tl.min(tl.where((idx < width) & (value != value), idx, block), 0)
    tl.store(output + row, tl.where(first_nan < block, first_nan, first))


def sample_topk_gumbel(
    logits: torch.Tensor, uniforms: torch.Tensor, top_k: int, inv_temperature: float
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape != uniforms.shape or logits.device != uniforms.device:
        raise ValueError("Gumbel uniforms must match the 2D logits on the same device")
    if not 0 < logits.shape[-1] <= 4096 or not 0 <= top_k <= logits.shape[-1]:
        raise ValueError("NPU sampling requires vocabulary 1..4096 and a valid top_k")
    scaled = (logits * inv_temperature).contiguous()
    # Read the strided last column directly: materializing it adds a device
    # copy at every residual codebook step. With no filtering, no buffer is needed.
    use_threshold = 0 < top_k < logits.shape[-1]
    thresholds = scaled.topk(top_k, dim=-1).values[:, -1] if use_threshold else scaled
    # Do not substitute CDF sampling or approximate Triton logs: consume the
    # same per-request uniforms and the same ATen log result as the reference.
    noise = torch.log(-torch.log(uniforms)).contiguous()
    output = torch.empty((logits.shape[0], 1), device=logits.device, dtype=torch.long)
    _select[(logits.shape[0],)](
        scaled,
        noise,
        thresholds,
        output,
        logits.shape[-1],
        triton.next_power_of_2(logits.shape[-1]),
        thresholds.stride(0),
        use_threshold,
        enable_fp_fusion=False,
    )
    return output
