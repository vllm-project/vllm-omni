# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for MiniMax-H3 image-level VAE encode data parallelism."""

from __future__ import annotations

import os
import socket
import types
from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist
from PIL import Image

from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as h3
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    MiniMaxH3Pipeline,
    _assign_minimax_h3_images_to_ranks,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeImageVAE:
    """Return one row block per image, filled with the image's red channel."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []
        self.encoder_component = types.SimpleNamespace()

    def is_distributed_enabled(self) -> bool:
        return False

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        red = int(image.getpixel((0, 0))[0])
        rows = (image.height // 32) * (image.width // 32)
        self.calls.append((red, rows))
        return torch.full((rows, 96), float(red), dtype=torch.float32)


def _pipeline(*, parallel_size: int, vae: _FakeImageVAE | None = None) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        device=torch.device("cpu"),
        vae_encode_parallel_size=parallel_size,
        video_vae=vae or _FakeImageVAE(),
        _component_on_device=lambda _component: nullcontext(),
    )


def _images(sizes: list[tuple[int, int]]) -> list[Image.Image]:
    return [Image.new("RGB", size, (index + 1, 0, 0)) for index, size in enumerate(sizes)]


def _expected_rows(sizes: list[tuple[int, int]]) -> torch.Tensor:
    parts = []
    for index, (width, height) in enumerate(sizes):
        rows = (height // 32) * (width // 32)
        parts.append(torch.full((rows, 96), float(index + 1), dtype=torch.float32))
    return torch.cat(parts)


def test_image_assignment_is_deterministic_balanced_and_ordered():
    sizes = [(2048, 2048), (2048, 1152), (1152, 2048), (512, 512)]
    first = _assign_minimax_h3_images_to_ranks(sizes, workers=3)
    second = _assign_minimax_h3_images_to_ranks(sizes, workers=3)

    assert first == second
    assert sorted(index for worker in first for index in worker) == [0, 1, 2, 3]
    assert all(worker == sorted(worker) for worker in first)
    assert max(len(worker) for worker in first) - min(len(worker) for worker in first) <= 1

    # Fewer images than workers leaves trailing workers empty.
    sparse = _assign_minimax_h3_images_to_ranks(sizes[:2], workers=8)
    assert [worker for worker in sparse if worker]
    assert all(worker == [] for worker in sparse[2:])


def test_single_rank_encode_keeps_serial_rank0_path():
    sizes = [(64, 64), (96, 64), (128, 64)]
    images = _images(sizes)
    vae = _FakeImageVAE()
    pipeline = _pipeline(parallel_size=4, vae=vae)

    rows, shapes = MiniMaxH3Pipeline._encode_image_conditions(
        pipeline,
        images,
        parallel=False,
        component_ready=False,
    )

    torch.testing.assert_close(rows, _expected_rows(sizes))
    assert shapes == [(1, 4, 4), (1, 4, 6), (1, 4, 8)]
    assert vae.calls == [(1, 4), (2, 6), (3, 8)]


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _distributed_worker(
    rank: int,
    world_size: int,
    master_port: int,
    sizes: list[tuple[int, int]],
    configured_workers: int,
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        original_rank_world = h3._dit_rank_world
        h3._dit_rank_world = lambda: (dist.group.WORLD, dist.get_rank(), dist.get_world_size())
        try:
            pipeline = _pipeline(parallel_size=configured_workers)
            rows, shapes = MiniMaxH3Pipeline._encode_image_conditions(
                pipeline,
                _images(sizes),
                parallel=True,
                component_ready=False,
            )
            torch.testing.assert_close(rows, _expected_rows(sizes))
            assert shapes == [
                (1, height // 16, width // 16) for width, height in sizes
            ]
        finally:
            h3._dit_rank_world = original_rank_world
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("sizes", "configured_workers"),
    [
        ([(64, 64), (96, 64)], 8),  # two empty ranks
        ([(64, 64), (96, 64), (128, 64)], 8),  # one empty rank
    ],
)
def test_distributed_image_encode_matches_serial_order_with_empty_ranks(
    sizes: list[tuple[int, int]],
    configured_workers: int,
) -> None:
    world_size = 4
    master_port = _get_open_port()
    context = torch.multiprocessing.get_context("fork")
    processes = [
        context.Process(
            target=_distributed_worker,
            args=(rank, world_size, master_port, sizes, configured_workers),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)
        assert process.exitcode == 0, f"rank process exited with {process.exitcode}"
