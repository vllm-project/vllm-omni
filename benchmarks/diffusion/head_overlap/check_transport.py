# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Two-rank attention, rank-row permutation and workspace reuse on synthetic QKV."""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from vllm_omni.diffusion.attention.parallel.head_overlap.ce_plan import prepare_plan


def attention(q, k, v, metadata=None):
    return (
        F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        .transpose(1, 2)
        .contiguous()
    )


def main():
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    rank, world = dist.get_rank(), dist.get_world_size()
    comm = torch.cuda.Stream()
    try:
        with torch.inference_mode():
            for seq in (127, 256, 127):
                for joint_len in (0, 19):
                    torch.manual_seed(13)
                    joint = (
                        tuple(
                            torch.randn(1, length, 8 * world, 64, device="cuda", dtype=torch.bfloat16)
                            for length in (0, joint_len, joint_len)
                        )
                        if joint_len
                        else None
                    )
                    torch.manual_seed(101 + rank + seq)
                    inputs = torch.randn(1, seq, 3, 8 * world, 64, device="cuda", dtype=torch.bfloat16).unbind(2)
                    full = []
                    for i, tensor in enumerate(inputs):
                        peers = [torch.empty_like(tensor, memory_format=torch.contiguous_format) for _ in range(world)]
                        dist.all_gather(peers, tensor.contiguous())
                        if joint is not None:
                            peers.insert(0, joint[i])
                        full.append(torch.cat(peers, dim=1))
                    expected = attention(*full)[:, rank * seq : (rank + 1) * seq]
                    for widths, tail in (((8,), 0), ((3, 5), 0), ((1, 7), seq), ((1, 7), seq // 2)):
                        plan = prepare_plan(
                            inputs,
                            joint,
                            group,
                            widths,
                            comm,
                            attention,
                            None,
                            query_tail_rows=tail,
                        )
                        actual = plan.run()
                        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-3)
            print(f"Rank {rank} passed 24 distributed attention comparisons", flush=True)
    finally:
        torch.accelerator.synchronize()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
