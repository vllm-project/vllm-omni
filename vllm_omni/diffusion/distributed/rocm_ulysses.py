# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict four-rank Ulysses for BF16 H3. The packet format is shared by all ranks."""

from functools import cache
from pathlib import Path

import torch
import torch.distributed as dist


@cache
def prepare() -> None:
    from torch.utils.cpp_extension import load_inline

    load_inline(
        name="vllm_omni_rocm_ulysses",
        cpp_sources=[],
        cuda_sources=Path(__file__).with_name("rocm_ulysses.hip").read_text(),
        no_implicit_headers=True,
        extra_cuda_cflags=["-O3"],
        is_python_module=False,
    )

    @torch.library.register_fake("vllm_omni_rocm::ulysses_pack")
    def _pack_fake(q, k, v):
        return q.new_empty((4, q.shape[1], 3, 14, 128))

    @torch.library.register_fake("vllm_omni_rocm::ulysses_restore")
    def _restore_fake(wire):
        return wire.new_empty((1, wire.shape[1], 56, 128))


def eligible(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, world_size: int) -> bool:
    # These logical dimensions agree across a strict fixed-split group. Never
    # use local pointer alignment, row strides or segmentation to select a collective.
    return (
        world_size == 4
        and not torch.is_grad_enabled()
        and q.ndim == 4
        and q.shape[0] == 1
        and q.shape[2:] == (56, 128)
        and 0 < q.shape[1] <= (2**31 - 1) // (3 * 4 * 224)
        and q.shape == k.shape == v.shape
        and q.device == k.device == v.device
        and q.is_cuda
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
    )


def exchange_qkv(
    group: dist.ProcessGroup, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, strided: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = q.shape[1]
    send = torch.ops.vllm_omni_rocm.ulysses_pack(q, k, v)
    receive = torch.empty_like(send)
    dist.all_to_all_single(receive, send, group=group)
    planes = receive.view(1, 4 * rows, 3, 14, 128)
    if strided:
        return planes[:, :, 0], planes[:, :, 1], planes[:, :, 2]
    # A backend without a strided-receive contract gets normal contiguous BSND.
    dense = planes.movedim(2, 0).contiguous()
    return dense[0], dense[1], dense[2]


def exchange_output(group: dist.ProcessGroup, output: torch.Tensor) -> torch.Tensor:
    rows = output.shape[1] // 4
    send = output.reshape(4, rows, 14, 128).resolve_neg().contiguous()
    receive = torch.empty_like(send)
    dist.all_to_all_single(receive, send, group=group)
    return torch.ops.vllm_omni_rocm.ulysses_restore(receive)
