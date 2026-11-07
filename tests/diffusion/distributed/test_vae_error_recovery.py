# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed.autoencoders import distributed_vae_executor as vae_executor
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3LightVideoVAE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class TileOperator:
    def split(self, tensor):
        return [vae_executor.TileTask(i, (i,), row[None]) for i, row in enumerate(tensor)], vae_executor.GridSpec(
            split_dims=(0,), grid_shape=(tensor.shape[0],)
        )

    def exec(self, task):
        return task.tensor * 2

    def merge(self, tiles, grid):
        return torch.cat([tiles[(i,)] for i in range(grid.grid_shape[0])])


def run_recovery_worker(rank, init_file):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        group = SimpleNamespace(device_group=dist.group.WORLD, cpu_group=dist.group.WORLD)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(vae_executor, "get_world_group", lambda: group)
            executor = vae_executor.DistributedVaeExecutor()
        executor.set_parallel_size(2)
        tensor = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        operator = TileOperator()
        for stage, failing_rank in [("decode", 1), ("packing", 1), ("merge", 0), ("gather", 0)]:
            with pytest.MonkeyPatch.context() as patch:
                if rank == failing_rank:

                    def fail(*args, **kwargs):
                        raise torch.OutOfMemoryError("injected OOM")

                    target, attribute = {
                        "decode": (operator, "exec"),
                        "packing": (executor, "_pack_local_tiles"),
                        "merge": (operator, "merge"),
                        "gather": (torch, "empty_like"),
                    }[stage]
                    patch.setattr(target, attribute, fail)
                with pytest.raises(torch.OutOfMemoryError, match="Distributed VAE"):
                    executor.execute(tensor, operator)
            torch.testing.assert_close(executor.execute(tensor, operator), tensor * 2)

        with pytest.raises(RuntimeError, match="Distributed VAE mixed failures failed"):
            with executor._sync_errors("mixed failures"):
                if rank == 0:
                    raise torch.OutOfMemoryError("retryable")
                raise ValueError("not retryable")
        torch.testing.assert_close(executor.execute(tensor, operator), tensor * 2)
    finally:
        dist.destroy_process_group()


def test_all_ranks_observe_failures_and_can_retry(tmp_path):
    torch.multiprocessing.spawn(run_recovery_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


def run_light_vae_decode_failure_worker(rank, init_file):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        group = SimpleNamespace(device_group=dist.group.WORLD, cpu_group=dist.group.WORLD)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(vae_executor, "get_world_group", lambda: group)
            executor = vae_executor.DistributedVaeExecutor()

        vae = object.__new__(MiniMaxH3LightVideoVAE)
        vae.distributed_executor = executor
        vae._window_batch_cap = lambda: 1
        vae._profile_stamp = lambda enabled: 0.0
        vae._temporal_plan = lambda latent_t: {
            "T": latent_t,
            "tcs": 1,
            "drop": 0,
            "tratio": 1,
            "overlap": 0,
            "chunk_num_frames": 1,
            "num_chunks": 2,
            "pad_tokens": 0,
            "frame_pre_padding": 0,
            "frame_overlap": 0,
            "intra_tail": 0,
        }
        vae._pad_temporal = lambda z, plan: z
        vae._window_latent = lambda zpad, ranges, t, w, plan: zpad

        model = SimpleNamespace(
            spatial_compression_ratio=1,
            tokens_chunk_size=1,
            token_overlap=0,
            temporal_compression_ratio=1,
            frame_pre_padding=0,
            frame_overlap=0,
            use_tiling=True,
            config=SimpleNamespace(token_drop=0, clip_length=1),
            decoder=torch.nn.Linear(1, 1),
        )

        def decode_clip(packed):
            if rank == 1:
                raise torch.OutOfMemoryError("injected OOM")
            return packed

        model._decode_clip = decode_clip
        vae.model = model

        z = torch.zeros(1, 1, 2, 1, 1)
        with pytest.raises(torch.OutOfMemoryError, match="light VAE window decode"):
            vae._decode_video_balanced(
                z,
                ys=[0],
                hs=[1],
                y_ov=[0],
                xs=[0],
                ws=[1],
                x_ov=[0],
                group=dist.group.WORLD,
                size=2,
                rank=rank,
                profile=False,
                started=0.0,
            )
    finally:
        dist.destroy_process_group()


def test_light_vae_decode_failure_is_agreed_before_gather(tmp_path):
    torch.multiprocessing.spawn(
        run_light_vae_decode_failure_worker,
        args=(str(tmp_path / "light-vae-rendezvous"),),
        nprocs=2,
        join=True,
    )


def run_light_vae_tile_prep_failure_worker(rank, init_file):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        group = SimpleNamespace(device_group=dist.group.WORLD, cpu_group=dist.group.WORLD)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(vae_executor, "get_world_group", lambda: group)
            executor = vae_executor.DistributedVaeExecutor()
        executor.set_parallel_size(2)

        vae = object.__new__(MiniMaxH3LightVideoVAE)
        vae.distributed_executor = executor
        vae._device_target = torch.device("cpu")
        vae._profile_enabled = lambda: False
        vae._profile_stamp = lambda enabled: 0.0
        vae._window_balance_enabled = lambda: False
        vae._tile_geometry = lambda latent_t, height, width: (1, 1, 0, 0)
        vae._split_tiles = lambda total, tile, overlap: (([0, 1], [1, 1], [0]) if total > 1 else ([0], [1], [0]))
        vae._stitch_tiles = lambda rows, y_ov, x_ov: rows[0][0]
        vae._light_settings = {"image_context_tokens": 1, "image_context_output_phase": 0}
        vae.is_distributed_enabled = lambda: True

        model = SimpleNamespace(
            spatial_compression_ratio=1,
            tokens_chunk_size=1,
            token_overlap=0,
            frame_pre_padding=0,
            use_tiling=True,
            decoder=torch.nn.Linear(1, 1),
            _decode_clip=lambda packed: packed,
            _decode=lambda packed: packed,
        )
        vae.model = model

        def fail_cat(*args, **kwargs):
            raise torch.OutOfMemoryError("injected packing OOM")

        z = torch.zeros(1, 1, 1, 2, 1)
        with pytest.MonkeyPatch.context() as patch:
            if rank == 1:
                patch.setattr(torch, "cat", fail_cat)
            with pytest.raises(torch.OutOfMemoryError, match="light VAE tile decode"):
                vae._decode_video(z)
    finally:
        dist.destroy_process_group()


def test_light_vae_tile_prep_failure_is_agreed_before_collective(tmp_path):
    torch.multiprocessing.spawn(
        run_light_vae_tile_prep_failure_worker,
        args=(str(tmp_path / "light-vae-prep-rendezvous"),),
        nprocs=2,
        join=True,
    )
