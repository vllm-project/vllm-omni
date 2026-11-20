# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from vllm_omni.diffusion.models.kyutai_aura.audio import to_video_rate
from vllm_omni.diffusion.models.kyutai_aura.config import (
    KYUTAI_AURA_1_3B,
    closest_bucket,
    resolve_kyutai_aura_config,
    unsupported_kyutai_aura,
)
from vllm_omni.diffusion.models.kyutai_aura.kv_cache import ChunkPositions, RollingChunkKVCache
from vllm_omni.diffusion.models.kyutai_aura.sampling import StudentSigmas, decode_spans

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _span(start: int, stop: int) -> list[int]:
    return list(range(start, stop))


# AURA-1.3B: 4 latent frames per chunk, anchor + 2 chunks, reference frame at 30 inside a
# 20-frame segment; positions freeze once chunk 4 (the last that fits a segment) is reached.
POSITIONS_1_3B = ChunkPositions.for_config(KYUTAI_AURA_1_3B)
EXPECTED_1_3B = {
    0: (_span(0, 4) + [30], _span(0, 4) + [30]),
    1: (_span(4, 8) + [30], _span(0, 8) + [30]),
    2: (_span(8, 12) + [30], _span(0, 12) + [30]),
    3: (_span(12, 16) + [30], _span(0, 4) + _span(8, 16) + [30]),
    4: (_span(16, 20) + [30], _span(0, 4) + _span(12, 20) + [30]),
    5: (_span(16, 20) + [30], _span(0, 4) + _span(12, 20) + [30]),
    8: (_span(16, 20) + [30], _span(0, 4) + _span(12, 20) + [30]),
}

# Without a freeze (the AURA-5B layout: 30-frame segments, reference at 40), the reference frame
# moves to the segment of the chunk's last frame.
POSITIONS_NO_FREEZE = ChunkPositions(frames_per_chunk=4, window_chunks=2, segment_frames=30, ref_offset=40)
EXPECTED_NO_FREEZE = {
    0: (_span(0, 4) + [40], _span(0, 4) + [40]),
    3: (_span(12, 16) + [40], _span(0, 4) + _span(8, 16) + [40]),
    7: (_span(28, 32) + [70], _span(0, 4) + _span(24, 32) + [70]),
    8: (_span(32, 36) + [70], _span(0, 4) + _span(28, 36) + [70]),
}


@pytest.mark.parametrize(
    ("positions", "expected"), [(POSITIONS_1_3B, EXPECTED_1_3B), (POSITIONS_NO_FREEZE, EXPECTED_NO_FREEZE)]
)
def test_query_and_key_positions(positions, expected):
    for chunk, (query, key) in expected.items():
        assert positions.query_positions(chunk) == query, chunk
        assert positions.key_positions(chunk) == key, chunk


def test_history_keeps_anchor_and_latest_chunks():
    cache = RollingChunkKVCache(
        num_layers=1, num_heads=1, head_dim=1, chunk_tokens=2, window_chunks=2, device=torch.device("cpu")
    )
    layouts = []
    for chunk in range(5):
        start, end = cache.current_rows()
        assert end == min(chunk + 1, 3) * 2
        cache.keys[0][:, start:end] = float(chunk)
        layouts.append(cache.keys[0][0, :end, 0, 0].tolist())
        cache.advance()
    # What each chunk attends to: the anchor (chunk 0) plus the latest chunks, itself included.
    assert layouts[2] == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
    assert layouts[3] == [0.0, 0.0, 2.0, 2.0, 3.0, 3.0]
    assert layouts[4] == [0.0, 0.0, 3.0, 3.0, 4.0, 4.0]
    assert cache.current_rows() == (4, 6)


def test_to_video_rate_pads_to_whole_chunks():
    features = torch.arange(250, dtype=torch.float32).view(1, 250, 1).expand(1, 250, 3)
    out, chunks = to_video_rate(features, fps=16, video_frames_per_chunk=16)
    assert chunks == 11 and out.shape == (1, 1, 3, 176)
    # Frame k samples the 25 Hz feature nearest to k / 16 s; frames past the speech are zero.
    assert out[0, 0, 0, 159].item() == round(159 * 25 / 16)
    assert torch.count_nonzero(out[0, 0, :, 160:]) == 0


def test_buckets_and_checkpoint_resolution(tmp_path):
    assert closest_bucket(KYUTAI_AURA_1_3B.buckets, 1080, 1920) == (512, 768)
    assert closest_bucket(KYUTAI_AURA_1_3B.buckets, 1024, 1024) == (576, 576)
    assert resolve_kyutai_aura_config("kyutai/AURA-1.3B") is KYUTAI_AURA_1_3B
    # AURA-5B is recognized, so the pipeline can say why it is not supported yet.
    assert resolve_kyutai_aura_config("https://huggingface.co/kyutai/aura-5b") is None
    assert "AURA-5B" in unsupported_kyutai_aura("kyutai/AURA-5B")
    assert unsupported_kyutai_aura("kyutai/AURA-1.3B") is None
    assert resolve_kyutai_aura_config(str(tmp_path)) is None
    (tmp_path / KYUTAI_AURA_1_3B.checkpoint).parent.mkdir(parents=True)
    (tmp_path / KYUTAI_AURA_1_3B.checkpoint).touch()
    assert resolve_kyutai_aura_config(str(tmp_path)) is KYUTAI_AURA_1_3B


def test_decode_spans_first_frame_alone_then_groups():
    assert decode_spans(4, 4, first=True) == [(0, 1), (1, 4)]
    assert decode_spans(4, 4, first=False) == [(0, 4)]
    assert decode_spans(4, 2, first=True) == [(0, 1), (1, 3), (3, 4)]
    assert decode_spans(1, 4, first=True) == [(0, 1)]


def test_student_sigmas_on_the_shifted_grid():
    sigma_of = StudentSigmas(KYUTAI_AURA_1_3B.schedule_steps, KYUTAI_AURA_1_3B.schedule_shift)
    assert sigma_of(1000) == 1.0
    sigmas = [sigma_of(t) for t in KYUTAI_AURA_1_3B.timesteps]
    assert sigmas == sorted(sigmas, reverse=True)
    for t, sigma in zip(KYUTAI_AURA_1_3B.timesteps, sigmas):
        # Each student timestep sits within half a grid step of its sigma's timestep.
        assert abs(1000 * sigma - t) < 1000 / KYUTAI_AURA_1_3B.schedule_steps, t
