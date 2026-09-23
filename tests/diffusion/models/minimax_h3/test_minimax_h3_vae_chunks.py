# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import gc
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from vllm_omni.diffusion.models.minimax_h3.chunked_decode import decode_h3_chunks
from vllm_omni.diffusion.models.minimax_h3.temporal_chunks import decode_temporal_chunks

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(scope="module")
def cpu_process_group():
    if dist.is_initialized():
        yield dist.group.WORLD
        return

    with tempfile.NamedTemporaryFile(prefix="h3_chunk_dist_") as rendezvous:
        init_method = f"file://{rendezvous.name}"
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=init_method)
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()
        gc.collect()


class _FakeTemporalModel:
    use_3d_conv = True
    token_drop = 3
    tokens_chunk_size = 5
    token_overlap = 2
    vae_ratio_t = 4
    frame_pre_padding = 3
    frame_overlap = 5
    isolated_first_frame = False
    isolated_last_frame = False

    def _decode_temporal_output_frame_plan(self, z, z_head, z_tail, num_chunks, pad_tokens):
        del z, z_head, z_tail, num_chunks, pad_tokens
        return 35, 0, 35

    def _adaptive_decode(self, clip):
        value = float(clip[:, :, 0].mean())
        return torch.full((1, 1, 24, 2, 2), value)

    @staticmethod
    def blend(overlap, part, frame_overlap, dim):
        del overlap, frame_overlap, dim
        return part


class _FakeHost:
    def __init__(self):
        self.model = _FakeTemporalModel()

    @staticmethod
    def _denormalize_latent(latent):
        return latent

    @staticmethod
    def _normalize_decoded_frames(frames):
        return frames.float()


def test_temporal_chunks_emit_ordered_frames_and_collect_when_unconsumed():
    model = _FakeTemporalModel()
    latent = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1, 1)
    chunks = []
    marker = decode_temporal_chunks(model, latent, chunks.append)

    assert marker.shape == (0,)
    assert [chunk.shape[2] for chunk in chunks] == [17, 17, 1]
    assert torch.equal(torch.cat(chunks, dim=2), decode_temporal_chunks(model, latent, None))


def test_h3_callback_failure_is_deferred_until_temporal_decode_finishes():
    host = _FakeHost()
    latent = torch.zeros(1, 1, 8, 1, 1)
    seen = []

    def fail_once(frames):
        seen.append(frames.shape[2])
        raise RuntimeError("sink failed")

    with pytest.raises(RuntimeError, match="sink failed"):
        decode_h3_chunks(host, latent, fail_once, group=None)
    assert seen == [17]


def test_h3_without_callback_is_a_plain_full_decode_on_a_vae_group(cpu_process_group):
    host = _FakeHost()
    latent = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1, 1)

    full = decode_h3_chunks(host, latent, None, group=cpu_process_group)

    chunks = []
    decode_h3_chunks(host, latent, chunks.append, group=cpu_process_group)
    assert torch.equal(torch.cat(chunks, dim=2), full)


class _SimulatedGroup:
    """Stand in for a VAE group so one process can check every rank's verdict.

    ``all_reduce`` is replaced with the sum the real collective would produce
    for a group of ``world_size`` ranks, ``supplied`` of which handed in a
    callback. That sum is what makes a rank-local verdict observable from a
    single process.
    """

    def __init__(self, world_size: int, supplied: int):
        self.world_size = world_size
        self.supplied = supplied


def _patch_simulated_ranks(monkeypatch, simulated: _SimulatedGroup, rank: int) -> None:
    from vllm_omni.diffusion.models.minimax_h3 import chunked_decode as mod

    def fake_all_reduce(tensor, group=None):
        del group  # the simulated census below stands in for the real reduction
        tensor.copy_(torch.tensor([simulated.supplied], dtype=tensor.dtype))

    monkeypatch.setattr(mod.dist, "get_rank", lambda _group: rank)
    monkeypatch.setattr(mod.dist, "get_world_size", lambda _group: simulated.world_size)
    monkeypatch.setattr(mod.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(mod.dist, "broadcast", lambda tensor, src=0, group=None: None)
    monkeypatch.setattr(mod.dist, "get_global_rank", lambda _group, _rank: 0)


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_h3_every_rank_rejects_a_partially_supplied_callback(monkeypatch, rank):
    """Ranks that disagree must all refuse, not let one decode alone.

    A rank-local verdict let a peer fall through into the decoder collectives
    while its peers raised, hanging the stage instead of failing it.
    """
    host = _FakeHost()
    latent = torch.zeros(1, 1, 8, 1, 1)
    # Two ranks of three handed in a callback.
    group = _SimulatedGroup(world_size=3, supplied=2)
    _patch_simulated_ranks(monkeypatch, group, rank)

    with pytest.raises(ValueError, match="every rank"):
        decode_h3_chunks(host, latent, lambda _frames: None, group=object())


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_h3_only_rank_zero_publishes_when_every_rank_supplies(monkeypatch, rank):
    """Ownership is positional: all ranks stream, rank 0 publishes."""
    host = _FakeHost()
    latent = torch.zeros(1, 1, 8, 1, 1)
    group = _SimulatedGroup(world_size=3, supplied=3)
    _patch_simulated_ranks(monkeypatch, group, rank)

    seen: list[int] = []
    decode_h3_chunks(host, latent, lambda f: seen.append(int(f.shape[2])), group=object())

    # Peers run the loop to stay in the collectives but publish nothing.
    assert (seen != []) is (rank == 0)


def _stub_video_vae(*, parallel_size: int, tile_count: int):
    """A MiniMaxH3VideoVAE with only the pieces the decode entry points read."""
    from contextlib import contextmanager

    from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

    vae = object.__new__(MiniMaxH3VideoVAE)
    vae.parallel_size = parallel_size
    vae.model = SimpleNamespace(_adaptive_decode=lambda clip: clip)
    vae.entered_rank_local = False

    @contextmanager
    def rank_local():
        vae.entered_rank_local = True
        yield

    vae._decoder_tile_count = lambda latent: tile_count
    vae._rank_local_tiling = rank_local
    vae.is_distributed_enabled = lambda: False
    return vae


def test_chunked_decode_falls_back_when_tiles_are_fewer_than_ranks(monkeypatch):
    """Without the fallback, tileless ranks raise while their peers block."""
    from vllm_omni.diffusion.models.minimax_h3 import vae as vae_mod

    vae = _stub_video_vae(parallel_size=4, tile_count=1)
    monkeypatch.setattr(vae_mod, "decode_h3_chunks", lambda *args, **kwargs: torch.zeros(1))

    vae.decode_with_chunks(torch.zeros(1, 1, 8, 1, 1), on_chunk=lambda _frames: None)

    assert vae.entered_rank_local, "chunked decode must share decode_latent's too-few-tiles fallback"


def test_chunked_decode_keeps_shared_tiling_when_every_rank_has_a_tile(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import vae as vae_mod

    vae = _stub_video_vae(parallel_size=4, tile_count=8)
    monkeypatch.setattr(vae_mod, "decode_h3_chunks", lambda *args, **kwargs: torch.zeros(1))

    vae.decode_with_chunks(torch.zeros(1, 1, 8, 1, 1), on_chunk=lambda _frames: None)

    assert not vae.entered_rank_local


def test_h3_vae_declares_the_chunked_decode_capability():
    """A consumer must be able to detect the capability and the pixel range."""
    from vllm_omni.diffusion.models.interface import supports_chunked_vae_decode
    from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

    vae = object.__new__(MiniMaxH3VideoVAE)

    assert supports_chunked_vae_decode(vae)
    # H3 reverts through the checkpoint's processor, which lands in [0, 1];
    # a Wan VAE publishes [-1, 1], so the range cannot be assumed.
    assert vae.chunk_value_range == (0.0, 1.0)


def _native_decode_temporal_streaming(model, z, z_head, z_tail, num_chunks, pad_tokens):
    """The released ``_decode_temporal_streaming`` loop, as the parity oracle.

    Transcribed from the checkpoint's ``AutoencoderKLLegacy`` so the fork can be
    compared against the control flow it mirrors, including the isolated
    head/tail extraction.
    """
    total_frames, pad_frames, output_frames = model._decode_temporal_output_frame_plan(
        z, z_head, z_tail, num_chunks, pad_tokens
    )
    chunk_dec = model.tokens_chunk_size * model.vae_ratio_t
    split_count = int(model.token_drop > 0) + 1
    dec = None
    dec_overlap = None
    write_pos = 0

    def write_part(part):
        nonlocal dec, write_pos
        part_frames = int(part.shape[2])
        if part_frames <= 0:
            return
        if dec is None:
            out_shape = list(part.shape)
            out_shape[2] = output_frames
            dec = torch.empty(out_shape, dtype=part.dtype, device=part.device)
        remaining = int(dec.shape[2]) - write_pos
        copy_frames = min(part_frames, max(0, remaining))
        if copy_frames > 0:
            dec[:, :, write_pos : write_pos + copy_frames].copy_(part[:, :, :copy_frames])
            write_pos += copy_frames

    for i in range(num_chunks):
        t_start = i * model.tokens_chunk_size
        clip_z = z[:, :, t_start : t_start + model.tokens_chunk_size + model.token_overlap]
        if i == 0 and z_head is not None:
            clip_z = torch.cat([z_head, clip_z], dim=2)
        if i == num_chunks - 1 and z_tail is not None:
            clip_z = torch.cat([clip_z, z_tail], dim=2)

        clip_dec = model._adaptive_decode(clip_z)

        dec_tail = None
        if i == 0 and z_head is not None:
            write_part(clip_dec[:, :, model.vae_ratio_t - 1 : model.vae_ratio_t])
            clip_dec = clip_dec[:, :, model.vae_ratio_t :]
        if i == num_chunks - 1 and z_tail is not None:
            dec_tail = clip_dec[:, :, -1:]
            clip_dec = clip_dec[:, :, : -model.vae_ratio_t]

        for j in range(split_count):
            f_start = j * chunk_dec
            f_end = min(f_start + chunk_dec, clip_dec.shape[2])
            chunk = clip_dec[:, :, f_start:f_end][:, :, model.frame_pre_padding :]
            if j == 0:
                if dec_overlap is not None:
                    chunk = model.blend(dec_overlap, chunk, model.frame_overlap, dim=-3)
                    dec_overlap = None
                write_part(chunk)
            else:
                dec_overlap = chunk.contiguous()

        if i == num_chunks - 1:
            if dec_overlap is not None:
                write_part(dec_overlap)
                dec_overlap = None
            if dec_tail is not None:
                write_part(dec_tail)

    return dec


class _IsolatedFrameModel(_FakeTemporalModel):
    """Fake decoder with the isolated boundaries the released config can set."""

    frame_pre_padding = 0

    def __init__(self, *, first: bool, last: bool):
        self.isolated_first_frame = first
        self.isolated_last_frame = last

    def _decode_temporal_output_frame_plan(self, z, z_head, z_tail, num_chunks, pad_tokens):
        del pad_tokens
        frames = num_chunks * self.tokens_chunk_size * self.vae_ratio_t
        frames += int(z_head is not None) + int(z_tail is not None)
        del z
        return frames, 0, frames

    def _adaptive_decode(self, clip):
        # Every latent token decodes to a distinct, ordered block, so a shifted
        # or reordered emission is visible in the values themselves.
        base = float(clip[:, :, 0].mean())
        frames = int(clip.shape[2]) * self.vae_ratio_t
        return torch.arange(frames, dtype=torch.float32).view(1, 1, frames, 1, 1) + base


@pytest.mark.parametrize(
    ("first", "last"),
    [(True, False), (False, True), (True, True)],
)
def test_temporal_chunks_match_the_native_loop_at_isolated_boundaries(first, last):
    """The fork must extract the isolated head/tail the way the native loop does."""
    model = _IsolatedFrameModel(first=first, last=last)
    latent = torch.arange(12, dtype=torch.float32).view(1, 1, 12, 1, 1)

    chunks: list[torch.Tensor] = []
    decode_temporal_chunks(model, latent, chunks.append)
    streamed = torch.cat(chunks, dim=2)

    # Feed the oracle exactly what the released ``decode_temporal`` would:
    # split the isolated tokens off, then pad the remainder to a whole chunk.
    isolated = int(first) + int(last)
    pseudo_tokens = int(latent.shape[2]) - isolated + model.token_drop
    remainder = pseudo_tokens % model.tokens_chunk_size
    pad_tokens = (model.tokens_chunk_size - remainder) if remainder else 0
    num_chunks = (pseudo_tokens + pad_tokens) // model.tokens_chunk_size - int(model.token_drop > 0)

    body = latent
    z_head = None
    if first:
        z_head = body[:, :, :1]
        body = body[:, :, 1:]
    z_tail = None
    if last:
        z_tail = body[:, :, -1:]
        body = body[:, :, :-1]
    if pad_tokens:
        body = torch.cat([body, body[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

    expected = _native_decode_temporal_streaming(model, body, z_head, z_tail, num_chunks, pad_tokens)

    assert streamed.shape == expected.shape
    assert torch.equal(streamed, expected)
