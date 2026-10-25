# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from datetime import timedelta
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = [pytest.mark.local_model, pytest.mark.cpu, pytest.mark.diffusion]


def _check_reference_parallelism(rank, rendezvous):
    from hymm.distributed.parallel_states import get_sequence_parallel_state, nccl_info

    from vllm_omni.diffusion.models.hy_nomas import hy_nomas_transformer

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("RANK", "0")
            group = Mock(ulysses_group=dist.group.WORLD, ulysses_world_size=2, world_size=2)
            patch.setattr(hy_nomas_transformer, "get_sp_group", lambda: group)
            mesh = hy_nomas_transformer.initialize_reference_parallelism(2, torch.device("cpu"))
            assert mesh.get_group("sp") is dist.group.WORLD
            assert nccl_info.sp_group is dist.group.WORLD
            assert nccl_info.sp_size == 2
            assert nccl_info.rank_within_spgroup == nccl_info.global_rank == rank
            assert nccl_info.parallel_dims.sp_group is dist.group.WORLD
            assert get_sequence_parallel_state()
            value = torch.tensor(rank + 1)
            dist.all_reduce(value, group=nccl_info.sp_group)
            assert value.item() == 3

            assert hy_nomas_transformer.initialize_reference_parallelism(1, torch.device("cpu")) is None
            assert not get_sequence_parallel_state()
            assert nccl_info.sp_size == 1
            assert nccl_info.sp_group is None
            assert nccl_info.device_mesh is None
            assert dist.is_initialized()

            group.ulysses_world_size = 4
            with pytest.raises(ValueError, match="match ulysses_degree"):
                hy_nomas_transformer.initialize_reference_parallelism(2, torch.device("cpu"))
    finally:
        dist.destroy_process_group()


def test_reference_reuses_process_group_and_resets_single_device(tmp_path):
    pytest.importorskip("hymm.distributed.parallel_states", reason="Install the pinned Hy-Nomas reference checkout")
    mp.spawn(_check_reference_parallelism, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
