# Copyright (c) Microsoft Corporation and Jiarui Fang
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team & Jiarui Fang
#  from https://github.com/feifeibear/long-context-attention/blob/main/yunchang/comm/all_to_all.py
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor


def all_to_all_4D(
    input: torch.tensor, scatter_idx: int = 2, gather_idx: int = 1, group=None, use_sync: bool = False
) -> torch.tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group (torch.distributed.ProcessGroup): torch process group
        use_sync (bool): whether to synchronize after all-to-all

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert input.dim() == 4, f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs).transpose(0, 2).contiguous()

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head

        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, bs, shard_hc, hs)

        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)->
        #  (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, bs, hs)

        # (hc, seqlen/N, bs, hs) -transpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int,
        gather_idx: int,
        use_sync: bool = False,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.use_sync = use_sync
        return all_to_all_4D(input, scatter_idx, gather_idx, group=group, use_sync=use_sync)


def all_to_all_5D(
    input: torch.tensor, scatter_idx: int = 3, gather_idx: int = 1, group=None, use_sync: bool = False
) -> torch.tensor:
    """
    all-to-all for QKV
    forward (bs, seqlen/N, 3, hc, hs) -> (bs, seqlen, 3, hc/N, hs)

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group (torch.distributed.ProcessGroup): torch process group
        use_sync (bool): whether to synchronize after all-to-all

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, 3, hc, hs)
    """
    assert input.dim() == 5, f"input must be 5D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 3 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, 3, hc, hs) output: (bs, seqlen, 3, hc/P, hs)
        bs, shard_seqlen, t_cnt, hc, hs = input.shape

        assert t_cnt == 3
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, 3, hc, hs) -reshape-> (bs, seq_len/P, 3, P, hc/P, hs) -transpose(0,3)->
        #  (P, seq_len/P, 3, bs, hc/P, hs)
        input_t = input.reshape(bs, shard_seqlen, 3, seq_world_size, shard_hc, hs).transpose(0, 3).contiguous()

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, 3, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, 3, bs, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, 3, bs, shard_hc, hs)

        # (seq_len, 3, bs, hc/P, hs) -trans-> (bs, seq_len, 3, hc/P, hs)
        output = output.transpose(0, 2).transpose(1, 2).contiguous()

        return output.reshape(bs, seqlen, 3, shard_hc, hs).contiguous()
    elif scatter_idx == 1 and gather_idx == 3:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, _, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, 3, hc/P, hs) -reshape-> (bs, P, seq_len/P, 3, hc/P, hs) -transpose(0, 4)->
        # (hc/P, P, seqlen/P, 3, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, 3, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, 3, shard_hc, hs)
            .transpose(0, 4)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, 3, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, 3, bs, hs)

        # (hc, seqlen/N, bs, hs) -transpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 3).contiguous()

        return output.reshape(bs, shard_seqlen, 3, hc, hs).contiguous()
    else:
        raise RuntimeError("scatter_idx must be 1 or 3 and gather_idx must be 1 or 3")


class SeqAllToAll5D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int = 3,
        gather_idx: int = 1,
        use_sync: bool = False,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.use_sync = use_sync

        return all_to_all_5D(input, scatter_idx, gather_idx, group=group, use_sync=use_sync)
