# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Two-group Ulysses exchange for the four-rank MiniMax-H3 BF16 attention path."""

from collections.abc import Callable
from functools import cache
from pathlib import Path

import torch
import torch.distributed as dist
from vllm.logger import init_logger

logger = init_logger(__name__)


@cache
def _native_ops():
    # Only the rank-local pack/restore implementation depends on the compiler.
    try:
        from torch.utils.cpp_extension import load_inline

        load_inline(
            name="vllm_omni_rocm_h3_head_overlap",
            cpp_sources=[],
            cuda_sources=Path(__file__).with_suffix(".hip").read_text(),
            extra_cuda_cflags=["-O3"],
            no_implicit_headers=True,
            is_python_module=False,
        )
        return torch.ops.vllm_omni_h3_overlap
    except (ImportError, OSError, RuntimeError) as exc:
        logger.warning("Using PyTorch for H3 head-group packing and restoration: %s", exc)
        return None


def _pack_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Each destination rank gets its own block of 14 heads, divided into 4+10.
    planes = [t.unflatten(1, (4, 14)).transpose(0, 1) for t in (q, k, v)]
    return (
        torch.stack([p[:, :, :4] for p in planes], dim=2),
        torch.stack([p[:, :, 4:] for p in planes], dim=2),
    )


def _restore_torch(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return torch.cat((first, second), dim=2).transpose(0, 1).reshape(first.shape[1], 56, 128)


@cache
def _comm_stream(device: torch.device) -> torch.cuda.Stream:
    return torch.cuda.Stream(device=device)


def _exchange(packet: torch.Tensor, group: dist.ProcessGroup):
    received = torch.empty_like(packet)
    work = dist.all_to_all_single(received, packet, group=group, async_op=True)
    return received, work


def run_head_overlap(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bounds: torch.Tensor,
    group: dist.ProcessGroup,
    compute: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None],
) -> torch.Tensor:
    """Run after all four ranks have agreed to the same split wire protocol.

    The forward packets have shape [rank, local row, Q/K/V, head, channel].
    The callback receives all sequence rows for a head group and writes its
    contiguous attention output on the calling stream.
    """
    stream = torch.cuda.current_stream(q.device)
    # Q/K/V and the cached bounds can originate on a different stream.
    for tensor in (q, k, v, bounds):
        tensor.record_stream(stream)
    ops = _native_ops()
    first, second = ops.pack(q, k, v) if ops is not None else _pack_torch(q, k, v)
    comm = _comm_stream(q.device)
    comm.wait_stream(stream)
    with torch.cuda.stream(comm):
        recv_first, forward_first = _exchange(first, group)
        recv_second, forward_second = _exchange(second, group)

    # Forward communication for the 10 heads can run during the first FMHA.
    forward_first.wait()
    recv_first.record_stream(stream)
    out_first = q.new_empty((4, q.shape[0], 4, 128))
    compute(recv_first, bounds, out_first)
    comm.wait_stream(stream)
    with torch.cuda.stream(comm):
        back_first, reverse_first = _exchange(out_first, group)

    # The first return exchange can run during the second FMHA.
    forward_second.wait()
    recv_second.record_stream(stream)
    out_second = q.new_empty((4, q.shape[0], 10, 128))
    compute(recv_second, bounds, out_second)
    comm.wait_stream(stream)
    with torch.cuda.stream(comm):
        back_second, reverse_second = _exchange(out_second, group)

    # All four sends remain live until their Work.wait() has inserted a
    # dependency on their allocation stream. The returns were allocated on
    # comm and will be read by the restore on stream.
    reverse_first.wait()
    reverse_second.wait()
    back_first.record_stream(stream)
    back_second.record_stream(stream)
    return ops.restore(back_first, back_second) if ops is not None else _restore_torch(back_first, back_second)
