# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import gc
import tempfile

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
