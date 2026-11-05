# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for MiniMax-H3 image-level VAE encode data parallelism."""

from __future__ import annotations

import os
import socket
import types
from contextlib import contextmanager, nullcontext

import numpy as np
import pytest
import torch
import torch.distributed as dist
from PIL import Image

from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as h3
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    MiniMaxH3Pipeline,
    _assign_minimax_h3_images_to_ranks,
)
from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.models.minimax_h3.conditioning import (
    MiniMaxH3EncoderMediaInput,
)
from vllm_omni.model_executor.models.minimax_h3.encoder_processing import encode_media

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeImageVAE:
    """Return one row block per image, filled with the image's red channel."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []
        self.rank_local_calls: list[bool] = []
        self.encoder_component = types.SimpleNamespace()
        self.distributed = False

    def is_distributed_enabled(self) -> bool:
        return self.distributed

    def encode_image(self, image: Image.Image, *, rank_local: bool = False) -> torch.Tensor:
        red = int(image.getpixel((0, 0))[0])
        rows = (image.height // 32) * (image.width // 32)
        self.calls.append((red, rows))
        self.rank_local_calls.append(bool(rank_local))
        return torch.full((rows, 96), float(red), dtype=torch.float32)

    def encode_video(self, frames):
        raise AssertionError("encode_video should not be called on this rank")


class _FakeImageVideoVAE(_FakeImageVAE):
    def encode_video(self, frames):
        num_frames, height, width = frames.shape[:3]
        latent_t = max(1, int(num_frames) // 4)
        rows = latent_t * (int(height) // 32) * (int(width) // 32)
        return torch.zeros((rows, 96), dtype=torch.float32), (
            latent_t,
            int(height) // 16,
            int(width) // 16,
        )


class _FakeAudioVAE:
    def encode_waveform(self, waveform: torch.Tensor, sample_rate: int):
        return torch.zeros((400, 32), dtype=torch.float32), 200


class _FakeFailingVAE(_FakeImageVAE):
    def __init__(
        self,
        *,
        fail: bool,
        exc_type: type[Exception] = RuntimeError,
        message: str = "injected encode failure",
    ) -> None:
        super().__init__()
        self.fail = fail
        self.exc_type = exc_type
        self.message = message

    def encode_image(self, image: Image.Image, *, rank_local: bool = False) -> torch.Tensor:
        if self.fail:
            raise self.exc_type(self.message)
        return super().encode_image(image, rank_local=rank_local)


def _pipeline(*, parallel_size: int, vae: _FakeImageVAE | None = None) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        device=torch.device("cpu"),
        vae_encode_parallel_size=parallel_size,
        video_vae=vae or _FakeImageVAE(),
    )


def _expected_rows(sizes: list[tuple[int, int]]) -> torch.Tensor:
    parts = []
    for index, (width, height) in enumerate(sizes):
        rows = (height // 32) * (width // 32)
        parts.append(torch.full((rows, 96), float(index + 1), dtype=torch.float32))
    return torch.cat(parts)


def _image_tensors(sizes: list[tuple[int, int]]) -> list[torch.Tensor]:
    return [torch.from_numpy(np.asarray(Image.new("RGB", size, (index + 1, 0, 0)))) for index, size in enumerate(sizes)]


def _expected_image_rows(sizes: list[tuple[int, int]]) -> list[torch.Tensor]:
    parts = []
    for index, (width, height) in enumerate(sizes):
        rows = (height // 32) * (width // 32)
        parts.append(torch.full((rows, 96), float(index + 1), dtype=torch.float32))
    return parts


def test_image_dp_encodes_rank_local_without_tile_parallel():
    sizes = [(64, 64), (96, 64), (128, 64)]
    vae = _FakeImageVAE()
    pipeline = _pipeline(parallel_size=3, vae=vae)

    rows, shapes = MiniMaxH3Pipeline._encode_images_data_parallel(
        pipeline,
        _image_tensors(sizes),
    )

    for actual, expected in zip(rows, _expected_image_rows(sizes), strict=True):
        torch.testing.assert_close(actual, expected)
        assert actual.device.type == "cpu"
    assert shapes == [(1, 4, 4), (1, 4, 6), (1, 4, 8)]
    assert all(vae.rank_local_calls)


def test_encode_media_uses_image_encoder_callback():
    sizes = [(64, 64), (96, 64)]
    tensors = _image_tensors(sizes)
    media = MiniMaxH3EncoderMediaInput(
        task="ref2va",
        height=64,
        width=64,
        num_frames=1,
        latent_t=1,
        audio_t=0,
        images=tuple(tensors),
    )
    vae = _FakeImageVAE()
    calls: list[bool] = []

    def image_encoder():
        calls.append(True)
        return (
            _expected_image_rows(sizes),
            [(1, height // 16, width // 16) for width, height in sizes],
        )

    conditioning = encode_media(
        media,
        video_vae=vae,
        audio_vae=None,
        emit_conditioning=True,
        component_scope=lambda _component: nullcontext(),
        image_encoder=image_encoder,
    )

    assert calls == [True]
    assert vae.calls == []
    torch.testing.assert_close(conditioning.visual_condition, _expected_rows(sizes))


def test_encode_media_skips_non_image_visuals_for_dp_workers():
    sizes = [(64, 64), (96, 64)]
    media = MiniMaxH3EncoderMediaInput(
        task="ref2va",
        height=64,
        width=64,
        num_frames=1,
        latent_t=1,
        audio_t=0,
        images=tuple(_image_tensors(sizes)),
        videos=(torch.zeros((1, 64, 64, 3), dtype=torch.uint8),),
    )
    vae = _FakeImageVAE()

    def image_encoder():
        return (
            _expected_image_rows(sizes),
            [(1, height // 16, width // 16) for width, height in sizes],
        )

    result = encode_media(
        media,
        video_vae=vae,
        audio_vae=None,
        emit_conditioning=False,
        component_scope=lambda _component: nullcontext(),
        image_encoder=image_encoder,
        encode_non_image_visuals=False,
    )

    assert result is None
    assert vae.calls == []


def test_encode_media_releases_video_before_audio_codec_loads():
    events: list[tuple[str, object | None]] = []

    @contextmanager
    def recording_scope(component):
        events.append(("enter", component))
        try:
            yield
        finally:
            events.append(("exit", component))

    def on_visual_complete():
        events.append(("visual_complete", None))

    media = MiniMaxH3EncoderMediaInput(
        task="ref2va",
        height=64,
        width=64,
        num_frames=1,
        latent_t=1,
        audio_t=1,
        images=(_image_tensors([(64, 64)])[0],),
        audios=((torch.zeros(32000), 16000),),
    )
    video_vae = _FakeImageVAE()
    audio_vae = _FakeAudioVAE()

    encode_media(
        media,
        video_vae=video_vae,
        audio_vae=audio_vae,
        emit_conditioning=True,
        component_scope=recording_scope,
        on_visual_complete=on_visual_complete,
    )

    visual_complete_index = events.index(("visual_complete", None))
    audio_enter_index = events.index(("enter", audio_vae))
    assert ("exit", video_vae) in events[:visual_complete_index]
    assert visual_complete_index < audio_enter_index


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


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_processes(processes: list[torch.multiprocessing.Process], timeout: int = 120) -> None:
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=timeout)
            assert process.exitcode == 0, f"rank process exited with {process.exitcode}"
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)


def _distributed_dp_worker(
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
            rows, shapes = MiniMaxH3Pipeline._encode_images_data_parallel(
                pipeline,
                _image_tensors(sizes),
            )
            assert len(rows) == len(sizes)
            for actual, expected in zip(rows, _expected_image_rows(sizes), strict=True):
                torch.testing.assert_close(actual, expected)
            assert shapes == [(1, height // 16, width // 16) for width, height in sizes]
            assert all(pipeline.video_vae.rank_local_calls)
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
def test_distributed_image_dp_matches_serial_order_with_empty_ranks(
    sizes: list[tuple[int, int]],
    configured_workers: int,
) -> None:
    world_size = 4
    master_port = _get_open_port()
    context = torch.multiprocessing.get_context("fork")
    processes = [
        context.Process(
            target=_distributed_dp_worker,
            args=(rank, world_size, master_port, sizes, configured_workers),
        )
        for rank in range(world_size)
    ]
    _run_processes(processes)


def _distributed_error_worker(
    rank: int,
    world_size: int,
    master_port: int,
    sizes: list[tuple[int, int]],
    configured_workers: int,
    failure_kind: str,
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
            exc_type: type[Exception] = ValueError if failure_kind == "value" else RuntimeError
            message = "injected value failure" if failure_kind == "value" else "injected encode failure"
            pipeline = _pipeline(
                parallel_size=configured_workers,
                vae=_FakeFailingVAE(
                    fail=(rank == 1),
                    exc_type=exc_type,
                    message=message,
                ),
            )
            try:
                MiniMaxH3Pipeline._encode_images_data_parallel(
                    pipeline,
                    _image_tensors(sizes),
                )
            except Exception as exc:
                text = str(exc)
                assert "rank 1" in text, text
                assert message in text, text
                assert exc_type.__name__ in text, text
                if failure_kind == "value":
                    assert isinstance(exc, OmniClientError), type(exc)
                else:
                    assert not isinstance(exc, OmniClientError), type(exc)
            else:
                raise AssertionError("expected image DP encode failure to propagate")
        finally:
            h3._dit_rank_world = original_rank_world
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("failure_kind", ["runtime", "value"])
def test_distributed_image_dp_reports_nonzero_rank_failure_without_hanging(
    failure_kind: str,
) -> None:
    sizes = [(64, 64), (96, 64), (128, 64)]
    world_size = 4
    master_port = _get_open_port()
    context = torch.multiprocessing.get_context("fork")
    processes = [
        context.Process(
            target=_distributed_error_worker,
            args=(rank, world_size, master_port, sizes, 4, failure_kind),
        )
        for rank in range(world_size)
    ]
    _run_processes(processes)


def _distributed_local_media_worker(
    rank: int,
    world_size: int,
    master_port: int,
    sizes: list[tuple[int, int]],
    configured_workers: int,
    distributed_video: bool = False,
    record_residency: bool = False,
    include_video: bool = False,
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
            pipeline = object.__new__(MiniMaxH3Pipeline)
            pipeline.device = torch.device("cpu")
            pipeline.vae_encode_parallel_size = configured_workers
            pipeline.video_vae = _FakeImageVideoVAE() if include_video else _FakeImageVAE()
            pipeline.video_vae.distributed = distributed_video
            pipeline.audio_vae = None
            vae_entered: list[object] = []
            if record_residency:

                @contextmanager
                def scope(component):
                    vae_entered.append(component)
                    yield

                pipeline._component_on_device = scope
            else:
                pipeline._component_on_device = lambda _component: nullcontext()
            media = None
            if rank == 0:
                media = MiniMaxH3EncoderMediaInput(
                    task="ref2va",
                    height=64,
                    width=64,
                    num_frames=1,
                    latent_t=1,
                    audio_t=1,
                    images=tuple(_image_tensors(sizes)),
                    videos=((torch.zeros((4, 64, 64, 3), dtype=torch.uint8),) if include_video else ()),
                )
            conditioning = MiniMaxH3Pipeline._encode_local_media(pipeline, media)
            expected_rows = _expected_rows(sizes)
            expected_shapes = [(1, height // 16, width // 16) for width, height in sizes]
            if include_video:
                expected_rows = torch.cat([expected_rows, torch.zeros((4, 96), dtype=torch.float32)])
                expected_shapes.append((1, 4, 4))
            torch.testing.assert_close(conditioning.visual_condition, expected_rows)
            assert tuple(conditioning.visual_condition_shapes) == tuple(expected_shapes)
            if distributed_video and rank == 0:
                assert any(pipeline.video_vae.rank_local_calls)
            if record_residency:
                encoder_component = pipeline.video_vae.encoder_component
                if distributed_video and include_video:
                    expected_entered = [encoder_component]
                else:
                    workers = min(configured_workers, world_size, len(sizes))
                    expected_entered = [encoder_component] if rank < workers else []
                assert vae_entered == expected_entered, (rank, vae_entered)
        finally:
            h3._dit_rank_world = original_rank_world
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("sizes", "configured_workers"),
    [
        ([(64, 64)], 4),  # DP configured, serial fallback for one image
        ([(64, 64), (96, 64)], 4),  # two images, two empty ranks
        ([(64, 64), (96, 64), (128, 64)], 8),  # one empty rank
    ],
)
def test_distributed_encode_local_media_image_dp_and_fallback(
    sizes: list[tuple[int, int]],
    configured_workers: int,
) -> None:
    world_size = 4
    master_port = _get_open_port()
    context = torch.multiprocessing.get_context("fork")
    processes = [
        context.Process(
            target=_distributed_local_media_worker,
            args=(rank, world_size, master_port, sizes, configured_workers),
        )
        for rank in range(world_size)
    ]
    _run_processes(processes)


def _distributed_load_error_worker(
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
            pipeline = object.__new__(MiniMaxH3Pipeline)
            pipeline.device = torch.device("cpu")
            pipeline.vae_encode_parallel_size = configured_workers
            pipeline.video_vae = _FakeImageVAE()
            pipeline.audio_vae = None
            encoder_component = pipeline.video_vae.encoder_component

            @contextmanager
            def scope(component):
                if rank == 1 and component is encoder_component:
                    raise RuntimeError("injected load failure")
                yield

            pipeline._component_on_device = scope
            media = None
            if rank == 0:
                media = MiniMaxH3EncoderMediaInput(
                    task="ref2va",
                    height=64,
                    width=64,
                    num_frames=1,
                    latent_t=1,
                    audio_t=1,
                    images=tuple(_image_tensors(sizes)),
                )
            try:
                MiniMaxH3Pipeline._encode_local_media(pipeline, media)
            except Exception as exc:
                text = str(exc)
                assert "injected load failure" in text, text
                assert "rank 1" in text, text
                assert "RuntimeError" in text, text
                assert not isinstance(exc, OmniClientError), type(exc)
            else:
                raise AssertionError("expected encoder load failure to propagate")
        finally:
            h3._dit_rank_world = original_rank_world
    finally:
        dist.destroy_process_group()


def test_distributed_encode_local_media_reports_encoder_load_failure():
    sizes = [(64, 64), (96, 64)]
    world_size = 4
    master_port = _get_open_port()
    context = torch.multiprocessing.get_context("fork")
    processes = [
        context.Process(
            target=_distributed_load_error_worker,
            args=(rank, world_size, master_port, sizes, 4),
        )
        for rank in range(world_size)
    ]
    _run_processes(processes)


def test_distributed_encode_local_media_dp_overrides_tile_parallel():
    sizes = [(64, 64), (96, 64)]
    world_size = 4
    master_port = _get_open_port()
    context = torch.multiprocessing.get_context("fork")
    processes = [
        context.Process(
            target=_distributed_local_media_worker,
            args=(rank, world_size, master_port, sizes, 4, True),
        )
        for rank in range(world_size)
    ]
    _run_processes(processes)


def _distributed_pipeline_residency_worker(
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
            pipeline = object.__new__(MiniMaxH3Pipeline)
            pipeline.device = torch.device("cpu")
            pipeline.vae_encode_parallel_size = configured_workers
            pipeline.video_vae = _FakeImageVAE()
            pipeline.audio_vae = _FakeAudioVAE() if rank == 0 else None
            encoder_component = pipeline.video_vae.encoder_component
            events: list[tuple[str, object]] = []

            @contextmanager
            def recording_scope(component):
                if rank == 0:
                    events.append(("enter", component))
                try:
                    yield
                finally:
                    if rank == 0:
                        events.append(("exit", component))

            pipeline._component_on_device = recording_scope
            media = None
            if rank == 0:
                media = MiniMaxH3EncoderMediaInput(
                    task="ref2va",
                    height=64,
                    width=64,
                    num_frames=1,
                    latent_t=1,
                    audio_t=1,
                    images=tuple(_image_tensors(sizes)),
                    audios=((torch.zeros(32000), 16000),),
                )
            conditioning = MiniMaxH3Pipeline._encode_local_media(pipeline, media)
            torch.testing.assert_close(conditioning.visual_condition, _expected_rows(sizes))
            if rank == 0:
                audio_enter = ("enter", pipeline.audio_vae)
                assert ("enter", encoder_component) in events
                assert ("exit", encoder_component) in events
                assert audio_enter in events
                assert events.index(("exit", encoder_component)) < events.index(audio_enter)
        finally:
            h3._dit_rank_world = original_rank_world
    finally:
        dist.destroy_process_group()


def test_distributed_encode_local_media_releases_video_before_audio():
    sizes = [(64, 64), (96, 64)]
    world_size = 4
    master_port = _get_open_port()
    context = torch.multiprocessing.get_context("fork")
    processes = [
        context.Process(
            target=_distributed_pipeline_residency_worker,
            args=(rank, world_size, master_port, sizes, 4),
        )
        for rank in range(world_size)
    ]
    _run_processes(processes)


def test_distributed_encode_local_media_skips_vae_on_idle_ranks():
    sizes = [(64, 64), (96, 64), (128, 64)]
    world_size = 4
    master_port = _get_open_port()
    context = torch.multiprocessing.get_context("fork")
    processes = [
        context.Process(
            target=_distributed_local_media_worker,
            args=(rank, world_size, master_port, sizes, 8, False, True),
        )
        for rank in range(world_size)
    ]
    _run_processes(processes)


def test_distributed_encode_local_media_loads_vae_for_non_image_visuals():
    sizes = [(64, 64), (96, 64)]
    world_size = 4
    master_port = _get_open_port()
    context = torch.multiprocessing.get_context("fork")
    processes = [
        context.Process(
            target=_distributed_local_media_worker,
            kwargs={
                "rank": rank,
                "world_size": world_size,
                "master_port": master_port,
                "sizes": sizes,
                "configured_workers": 4,
                "distributed_video": True,
                "record_residency": True,
                "include_video": True,
            },
        )
        for rank in range(world_size)
    ]
    _run_processes(processes)


def test_distributed_encode_local_media_skips_vae_on_idle_ranks_with_distributed_video():
    sizes = [(64, 64), (96, 64), (128, 64)]
    world_size = 4
    master_port = _get_open_port()
    context = torch.multiprocessing.get_context("fork")
    processes = [
        context.Process(
            target=_distributed_local_media_worker,
            kwargs={
                "rank": rank,
                "world_size": world_size,
                "master_port": master_port,
                "sizes": sizes,
                "configured_workers": 8,
                "distributed_video": True,
                "record_residency": True,
                "include_video": False,
            },
        )
        for rank in range(world_size)
    ]
    _run_processes(processes)
