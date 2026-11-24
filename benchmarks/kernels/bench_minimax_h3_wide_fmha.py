# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Replay the complete H3 attention operator on four gfx950 GPUs.

torchrun --nproc-per-node=4 benchmarks/kernels/bench_minimax_h3_wide_fmha.py wide_first.hsaco
"""

import argparse
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.distributed.packed_qkv import packed_output_all_to_all, packed_qkv_all_to_all
from vllm_omni.diffusion.models.minimax_h3.attention.wide_fmha import MiniMaxH3WideFlashImpl
from vllm_omni.diffusion.models.minimax_h3.rocm_wide_fmha import _CUTS, _workspaces


@torch.inference_mode()
def compare(code_object: str) -> None:
    if dist.get_world_size() != 4:
        raise ValueError("Launch with four MI350X ranks")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    generator = torch.Generator(device=device).manual_seed(625 + dist.get_rank())
    inputs = tuple(
        torch.randn((1, 9440, 56, 128), device=device, dtype=torch.bfloat16, generator=generator) for _ in range(3)
    )
    options = dict(num_heads=14, head_size=128, softmax_scale=128**-0.5, num_kv_heads=14)
    baseline, candidate = FlashAttentionImpl(**options), MiniMaxH3WideFlashImpl(**options)
    if baseline._kernel_variant != "aiter":
        raise RuntimeError("The selected ROCm FlashAttention provider must be AITER")
    group = dist.group.WORLD

    def run(impl, metadata):
        q, k, v = packed_qkv_all_to_all(group, *inputs)
        return packed_output_all_to_all(group, impl.forward(q, k, v, metadata))

    times: dict[str, list[float]] = {"baseline": [], "candidate": []}
    for cut in sorted(_CUTS):
        cu = torch.tensor([0, cut, 37_760], device=device, dtype=torch.int32)
        extra = dict(cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=cut, max_seqlen_k=cut, valid_kv_length=cut)
        arms = (
            ("baseline", baseline, AttentionMetadata(extra=extra)),
            ("candidate", candidate, AttentionMetadata(extra={**extra, "rocm_h3_wide_fmha_path": code_object})),
        )
        torch.testing.assert_close(run(*arms[0][1:]), run(*arms[1][1:]), rtol=0.02, atol=0.005)
        if not any(key[0] == device and cut in workspace.tables for key, workspace in _workspaces.items()):
            raise RuntimeError(f"Experimental kernel fell back on rank {dist.get_rank()} for cut {cut}")
        for iteration in range(6):
            for name, impl, metadata in arms if iteration % 2 == 0 else reversed(arms):
                dist.barrier(device_ids=[device.index])
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                run(impl, metadata)
                end.record()
                end.synchronize()
                elapsed = torch.tensor(start.elapsed_time(end), device=device)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                times[name].append(elapsed.item())

    if dist.get_rank() == 0:
        for name, measurements in times.items():
            print(
                f"{name} median slowest-rank time: {statistics.median(measurements):.3f} ms ({len(measurements)} calls)"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hsaco", type=Path, help="gfx950 code object built from wide_first_gfx950.s")
    args = parser.parse_args()
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        compare(str(args.hsaco.resolve(strict=True)))
    finally:
        dist.destroy_process_group()
