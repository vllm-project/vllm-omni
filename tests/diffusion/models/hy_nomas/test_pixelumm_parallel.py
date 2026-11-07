# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from vllm_omni.diffusion.models.hy_nomas.pixelumm import PixelUMMDCAELatentDecoder
from vllm_omni.diffusion.models.hy_nomas.pixelumm_model import DCAEVPixelUMMDecoder, GenExpertArch

pytestmark = [pytest.mark.cpu, pytest.mark.diffusion, pytest.mark.parallel]


class TinyDecoder(nn.Module):
    local_attention_tile = 0

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.125), requires_grad=False)
        self.calls = 0
        self.fail = False

    def forward(self, latent, noise):
        self.calls += 1
        if self.fail:
            raise RuntimeError("injected window failure")
        pixels = latent[:, :1].repeat(1, 3, 1, 1, 1)
        pixels = pixels.repeat_interleave(4, 2).repeat_interleave(32, 3).repeat_interleave(32, 4)
        return pixels * self.scale + noise * 0.01


def make_decoder(*, spatial_tile=0, num_steps=1, actual_model=False):
    if actual_model:
        torch.manual_seed(11)
        arch = GenExpertArch(
            hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2, num_kv_heads=1, head_dim=8
        )
        decoder = DCAEVPixelUMMDecoder(arch)
    else:
        decoder = TinyDecoder()
    return PixelUMMDCAELatentDecoder(
        decoder,
        seed=7,
        num_steps=num_steps,
        temporal_window_latents=4,
        temporal_overlap_latents=2,
        decoder_tile_pixels=spatial_tile,
        decoder_tile_overlap_pixels=32 if spatial_tile else 0,
    ).eval()


def run_stream(model, chunks):
    session = model.create_streaming_decode_session()
    output, positions = [], []
    for index, latent in enumerate(chunks):
        session.final_segment = index == len(chunks) - 1
        count = latent.shape[2] * 4 - (index % 3)
        output.append(session.push(latent, pixel_frames=count))
        positions.append(session.committed_pixel_frames)
    output.append(session.finish())
    positions.append(session.committed_pixel_frames)
    assert session.finished and session.buffer.numel() == 0 and session.pending is None
    assert not session.push_latent_frames and not session.push_pixel_frames
    return output, positions


def assert_output_matches(rank, actual, expected):
    if rank == 0:
        assert len(actual) == len(expected)
        for part, baseline in zip(actual, expected):
            torch.testing.assert_close(part, baseline, rtol=0, atol=0)
    else:
        assert all(part.numel() == 0 for part in actual)


def expect_collective_error(call, match):
    with pytest.raises((RuntimeError, ValueError), match=match):
        call()
    dist.barrier()


def _window_worker(rank, world_size, rendezvous):
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=120)
    )
    try:
        torch.set_num_threads(1)
        torch.manual_seed(13)
        chunks = [torch.randn(1, 32, count, 1, 1) for count in (1, 3, 2 * world_size + 3, 2)]
        baseline = make_decoder()
        expected, positions = run_stream(baseline, chunks)
        parallel = make_decoder()
        parallel.configure_distributed_decode(dist.group.WORLD)
        actual, actual_positions = run_stream(parallel, chunks)
        assert_output_matches(rank, actual, expected)
        assert actual_positions == positions
        local_calls = torch.tensor([parallel.decoder.calls], dtype=torch.int64)
        all_calls = [torch.empty_like(local_calls) for _ in range(world_size)]
        dist.all_gather(all_calls, local_calls)
        assert all(value.item() > 0 for value in all_calls)
        assert sum(value.item() for value in all_calls) == baseline.decoder.calls

        for steps in (1, 2):
            baseline = make_decoder(num_steps=steps)
            parallel = make_decoder(num_steps=steps)
            parallel.configure_distributed_decode(dist.group.WORLD)
            latent = torch.randn(1, 32, 9, 1, 1)
            expected = baseline.decode(latent, pixel_frames=33).sample
            actual = parallel.decode(latent, pixel_frames=33).sample
            assert_output_matches(rank, [actual], [expected])

        baseline = make_decoder(spatial_tile=64)
        parallel = make_decoder(spatial_tile=64)
        parallel.configure_distributed_decode(dist.group.WORLD)
        latent = torch.randn(1, 32, 5, 3, 3)
        assert_output_matches(rank, [parallel.decode(latent).sample], [baseline.decode(latent).sample])

        latent = torch.randn(1, 32, 5, 2, 3).transpose(-1, -2)
        assert not latent.is_contiguous()
        assert_output_matches(rank, [parallel.decode(latent).sample], [baseline.decode(latent).sample])

        if world_size == 2:
            baseline = make_decoder(actual_model=True)
            parallel = make_decoder(actual_model=True)
            parallel.configure_distributed_decode(dist.group.WORLD)
            latent = torch.randn(1, 32, 7, 1, 1)
            assert_output_matches(rank, [parallel.decode(latent).sample], [baseline.decode(latent).sample])
            expected, positions = run_stream(baseline, [latent[:, :, :3], latent[:, :, 3:]])
            actual, actual_positions = run_stream(parallel, [latent[:, :, :3], latent[:, :, 3:]])
            assert_output_matches(rank, actual, expected)
            assert actual_positions == positions

        latent = torch.randn(1, 32, 2 * world_size + 2, 1, 1)
        parallel = make_decoder()
        parallel.configure_distributed_decode(dist.group.WORLD)
        session = parallel.create_streaming_decode_session()
        parallel.decoder.fail = rank == world_size - 1
        expect_collective_error(lambda: session.push(latent), "injected window failure|another decode rank")
        parallel.decoder.fail = False
        expected, positions = run_stream(make_decoder(), [latent])
        actual, actual_positions = run_stream(parallel, [latent])
        assert_output_matches(rank, actual, expected)
        assert actual_positions == positions

        if world_size > 2:
            # Larger groups already exercised partitioning and non-output rank recovery.
            return

        session = parallel.create_streaming_decode_session()
        invalid = latent[:, :, :0] if rank == world_size - 1 else latent
        expect_collective_error(lambda: session.push(invalid), "non-empty|another decode rank")
        session = parallel.create_streaming_decode_session()
        pixels = latent.shape[2] * 4 - (1 if rank == world_size - 1 else 2)
        expect_collective_error(lambda: session.push(latent, pixel_frames=pixels), "identical metadata")
        session = parallel.create_streaming_decode_session()
        different = latent + (1 if rank == world_size - 1 else 0)
        expect_collective_error(lambda: session.push(different), "identical latent values")
        session = parallel.create_streaming_decode_session()
        session.final_segment = rank != world_size - 1
        expect_collective_error(lambda: session.push(latent), "identical metadata")

        session = parallel.create_streaming_decode_session()
        session.push(latent)
        if rank == 0:
            session.pending = None
        expect_collective_error(session.finish, "missing pending|another decode rank")
        expected, positions = run_stream(make_decoder(), [latent])
        actual, actual_positions = run_stream(parallel, [latent])
        assert_output_matches(rank, actual, expected)
        assert actual_positions == positions

        wrong_owner = make_decoder()
        wrong_owner.configure_distributed_decode(dist.group.WORLD, output_rank=1 if rank == world_size - 1 else 0)
        expect_collective_error(lambda: wrong_owner.decode(latent), "identical metadata")
        wrong_dtype = make_decoder()
        if rank == world_size - 1:
            wrong_dtype.double()
        wrong_dtype.configure_distributed_decode(dist.group.WORLD)
        expect_collective_error(lambda: wrong_dtype.decode(latent), "identical metadata")

        original_forward = parallel.decoder.forward
        if rank == world_size - 1:

            def wrong_shape(*args):
                return original_forward(*args)[..., :-1]

            parallel.decoder.forward = wrong_shape
        expect_collective_error(lambda: parallel.decode(latent), "distributed tensor contract|another decode rank")
        parallel.decoder.forward = original_forward

        original_merge = parallel._merge_tiles
        if rank == 0:

            def fail_merge(*args):
                raise RuntimeError("injected merge failure")

            parallel._merge_tiles = fail_merge
        expect_collective_error(lambda: parallel.decode(latent), "injected merge failure|another decode rank")
        parallel._merge_tiles = original_merge
        assert_output_matches(rank, [parallel.decode(latent).sample], [make_decoder().decode(latent).sample])
    finally:
        dist.destroy_process_group()


@pytest.mark.core_model
@pytest.mark.parametrize("world_size", [2, 4])
def test_window_parallel_matches_serial_and_recovers(tmp_path, monkeypatch, world_size):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4]))
    mp.spawn(_window_worker, args=(world_size, (tmp_path / "rendezvous").as_uri()), nprocs=world_size, join=True)


@pytest.mark.slow
@pytest.mark.parametrize("world_size", [8, 16])
def test_window_parallel_larger_groups(tmp_path, monkeypatch, world_size):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4]))
    mp.spawn(_window_worker, args=(world_size, (tmp_path / "rendezvous").as_uri()), nprocs=world_size, join=True)
