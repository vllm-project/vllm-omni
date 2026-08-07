# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import socket

import pytest
import torch
import torch.distributed as dist

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion, pytest.mark.parallel]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _partial_text_encoder_tp_worker(
    rank: int,
    world_size: int,
    text_encoder_tp_size: int,
    master_port: int,
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{master_port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

        pipeline = object.__new__(MiniMaxH3Pipeline)
        torch.nn.Module.__init__(pipeline)
        pipeline._dit_rank = rank
        pipeline.text_encoder_group = pipeline._build_text_encoder_group(text_encoder_tp_size)
        group = pipeline.text_encoder_group

        assert group.ranks == list(range(text_encoder_tp_size))
        assert group.world_size == text_encoder_tp_size
        assert group.is_member is (rank < text_encoder_tp_size)
        assert group.rank_in_group == (rank if rank < text_encoder_tp_size else -1)

        if group.is_member:
            reduced = torch.tensor(rank + 1, dtype=torch.int64)
            group.all_reduce(reduced)
            assert reduced.item() == sum(range(1, text_encoder_tp_size + 1))

            source = torch.tensor([123], dtype=torch.int64) if rank == 0 else None
            broadcast = pipeline._encoder_group_broadcast_tensor(
                source,
                dtype=torch.int64,
                device=torch.device("cpu"),
            )
            assert broadcast.tolist() == [123]
        else:
            with pytest.raises(RuntimeError, match="not a member"):
                group.all_reduce(torch.ones(1))

        # Proves that non-encoder ranks did not abort or enter encoder
        # collectives while the first N ranks used the partial group.
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_partial_text_encoder_tp_group_is_safe_for_non_members() -> None:
    world_size = 4
    text_encoder_tp_size = 2
    torch.multiprocessing.spawn(
        _partial_text_encoder_tp_worker,
        args=(world_size, text_encoder_tp_size, _find_free_port()),
        nprocs=world_size,
        join=True,
    )
