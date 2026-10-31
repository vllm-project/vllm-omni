# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Measure whether packing K and V into one Ring P2P transfer is worth it.

``RingComm`` already batches the separate K and V ``send_recv`` calls into a
single ``batch_isend_irecv``, so packing them trades two NCCL ops per hop for
one, plus one copy to build the packed buffer.

Result on 4x H100 (9450 tokens/rank, 40 heads, head_dim 128, bfloat16): packing
is ~1.9x on an isolated rotation, and nearly all of that comes from the single
larger transfer rather than from owning the receive buffer. Even so, packing
was NOT adopted in ``ring_flash_attn_forward``: the full ring call measured
slightly slower with it (12.28 -> 12.43 ms), because the forward issues the
transfer, computes the current block's attention, and only then waits, so the
transfer is already hidden behind compute. Making hidden work faster buys
nothing, while the packing copy lands on the critical path. Keep this script as
the evidence for that decision, and re-check it if the ring loop ever stops
overlapping comm with compute.

Three strategies move the same bytes around the ring:

  * split        -- what ships today: one ``send_recv`` for K and one for V,
                    both committed together, each receiving into a fresh buffer.
  * split+buf    -- the same two ops, receiving into caller-owned buffers,
                    which isolates allocation cost from op count.
  * packed+reuse -- one ``send_recv`` of a ``[2, ...]`` buffer into a
                    caller-owned ping-pong buffer, including the packing copy.
                    Reusing a receive buffer across hops would also let a hop's
                    NCCL write race the previous hop's compute read.
  * packed+alloc -- one ``send_recv`` of a ``[2, ...]`` buffer into a fresh
                    buffer per hop, so nothing is reused.

    cd projs/learn/vllm-omni
    torchrun --standalone --nproc-per-node=4 \
        tests/diffusion/attention/bench_ring_comm_packed.py \
        --sequence=9450 --heads=40 --head-dim=128
"""

import argparse
import os
import statistics

import torch
import torch.distributed as dist


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=positive_int, default=9450)
    parser.add_argument("--heads", type=positive_int, default=40)
    parser.add_argument("--head-dim", type=positive_int, default=128)
    parser.add_argument("--warmup", type=positive_int, default=5)
    parser.add_argument("--rounds", type=positive_int, default=10)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    from vllm_omni.diffusion.distributed.comm import RingComm

    dist.init_process_group("nccl")
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        device = torch.device("cuda", local_rank)
        shape = (1, args.sequence, args.heads, args.head_dim)
        k = torch.randn(shape, dtype=torch.bfloat16, device=device)
        v = torch.randn(shape, dtype=torch.bfloat16, device=device)

        k_bufs = [torch.empty_like(k) for _ in range(2)]
        v_bufs = [torch.empty_like(v) for _ in range(2)]
        kv_bufs = [torch.empty((2, *shape), dtype=k.dtype, device=device) for _ in range(2)]

        def split():
            """One full ring rotation, allocating a recv buffer per hop."""
            cur_k, cur_v = k, v
            for step in range(world_size - 1):
                comm = RingComm(dist.group.WORLD)
                next_k = comm.send_recv(cur_k)
                next_v = comm.send_recv(cur_v)
                comm.commit()
                comm.wait()
                cur_k, cur_v = next_k, next_v
            return cur_k, cur_v

        def split_buffered():
            """Same op count, receiving into caller-owned ping-pong buffers."""
            cur_k, cur_v = k, v
            for step in range(world_size - 1):
                comm = RingComm(dist.group.WORLD)
                next_k = comm.send_recv(cur_k, recv_tensor=k_bufs[step & 1])
                next_v = comm.send_recv(cur_v, recv_tensor=v_bufs[step & 1])
                comm.commit()
                comm.wait()
                cur_k, cur_v = next_k, next_v
            return cur_k, cur_v

        def packed():
            """One op per hop, including the copy that packs K and V."""
            kv_bufs[1][0].copy_(k)
            kv_bufs[1][1].copy_(v)
            for step in range(world_size - 1):
                comm = RingComm(dist.group.WORLD)
                comm.send_recv(kv_bufs[(step - 1) & 1], recv_tensor=kv_bufs[step & 1])
                comm.commit()
                comm.wait()
            final = kv_bufs[(world_size - 2) & 1]
            return final[0], final[1]

        def packed_alloc():
            """One op per hop into a fresh buffer, so no buffer is reused.

            Reusing a receive buffer across hops would let a hop's NCCL write
            race the previous hop's compute read of the same buffer, so the
            shipped path takes this variant unless the reuse is proven safe.
            """
            kv_send = torch.empty((2, *shape), dtype=k.dtype, device=device)
            kv_send[0].copy_(k)
            kv_send[1].copy_(v)
            for _ in range(world_size - 1):
                comm = RingComm(dist.group.WORLD)
                kv_recv = comm.send_recv(kv_send)
                comm.commit()
                comm.wait()
                kv_send = kv_recv
            return kv_send[0], kv_send[1]

        variants = {
            "split": split,
            "split+buf": split_buffered,
            "packed+reuse": packed,
            "packed+alloc": packed_alloc,
        }
        results = {}
        for name, fn in variants.items():
            for _ in range(args.warmup):
                fn()
            torch.accelerator.synchronize()
            samples = []
            for _ in range(args.rounds):
                dist.barrier()
                torch.accelerator.synchronize()
                start = torch.Event(device="cuda", enable_timing=True)
                end = torch.Event(device="cuda", enable_timing=True)
                start.record()
                fn()
                end.record()
                end.synchronize()
                elapsed = torch.tensor(start.elapsed_time(end), device=device, dtype=torch.float64)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                samples.append(elapsed.item())
            results[name] = (statistics.mean(samples), statistics.stdev(samples))

        if rank == 0:
            print(f"world_size={world_size}  shape_per_rank={shape}  dtype=bfloat16")
            print(f"one full ring rotation ({world_size - 1} hops), MAX rank time per round")
            base = results["split"][0]
            for name, (mean, stdev) in results.items():
                print(f"  {name:<10} {mean:8.3f} +/- {stdev:6.3f} ms   {base / mean:5.3f}x vs split")
    finally:
        torch.accelerator.synchronize()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
