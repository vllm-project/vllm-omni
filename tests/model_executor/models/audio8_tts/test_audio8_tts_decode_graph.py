# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Correctness of the Audio8 TTS codec decode CUDA graphs.

The streaming contract decodes ``[left_context | new_frames]`` windows; the
deployed shapes are the first steady window (49 frames with the shipped
chunker config) and every steady window after it (50 frames). Graph replays
must be bit-exact with eager ``ArkttsCodec.decode`` at those shapes, fall
back to eager for every other shape, and survive interleaving with eager
decodes that touch the shared mask/rope caches.
"""

import pytest
import torch
import torch.nn as nn

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.audio8_tts.audio8_tts_codec_decoder import _resolve_graph_frames
from vllm_omni.model_executor.models.audio8_tts.codec import build_arktts_codec
from vllm_omni.model_executor.models.audio8_tts.decode_graph import ArkttsDecodeGraphRunner
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model]

NUM_CODEBOOKS = 10
CAPTURED_FRAMES = [49, 50]
CUDA_DEVICE = torch.device("cuda")


def _make_codec(device: torch.device) -> nn.Module:
    torch.manual_seed(0)
    return build_arktts_codec().to(device=device, dtype=torch.float32).eval()


def _codes(frames: int, *, seed: int, device: torch.device, out_of_range: bool = False) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    semantic = torch.randint(0, 4096, (1, 1, frames), generator=gen)
    residual = torch.randint(0, 1024, (1, 9, frames), generator=gen)
    codes = torch.cat([semantic, residual], dim=1)
    if out_of_range:
        # The quantizer clamps in-place on a clone; the graph must keep that
        # off the caller's buffer exactly like eager does.
        codes[:, 0, :] = 999999  # semantic codebook (card 4096), every frame
        codes[:, 1:, -1] = -7  # residual codebooks (card 1024), last frame
    return codes.to(device)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("extra_cfg", "expected"),
    [
        # The shipped deploy config: chunk 25, left context 25, initial 4.
        (
            {"codec_chunk_frames": 25, "codec_left_context_frames": 25, "initial_codec_chunk_frames": 4},
            [49, 50],
        ),
        # Connector defaults (no initial chunk): first steady is a bare chunk.
        ({}, [25, 50]),
        # initial >= left context: the first steady window is already steady-sized.
        (
            {"codec_chunk_frames": 25, "codec_left_context_frames": 20, "initial_codec_chunk_frames": 5},
            [45],
        ),
        # Explicit override wins, deduplicated and sorted.
        ({"audio8_tts_codec_cuda_graph_frames": "50, 29, 50"}, [29, 50]),
    ],
)
def test_graph_frames_follow_the_connector_chunker(extra_cfg: dict, expected: list[int]) -> None:
    assert _resolve_graph_frames(extra_cfg) == expected


@pytest.mark.cpu
def test_graph_frames_reject_invalid_override() -> None:
    with pytest.raises(ValueError, match="cuda graph frames"):
        _resolve_graph_frames({"audio8_tts_codec_cuda_graph_frames": "50, 0"})


@pytest.mark.cpu
def test_runner_validates_arguments() -> None:
    codec = build_arktts_codec()
    with pytest.raises(ValueError, match="num_codebooks"):
        ArkttsDecodeGraphRunner(codec, num_codebooks=0, frames=[50], device=CUDA_DEVICE)
    with pytest.raises(ValueError, match="frame buckets"):
        ArkttsDecodeGraphRunner(codec, num_codebooks=NUM_CODEBOOKS, frames=[], device=CUDA_DEVICE)
    with pytest.raises(ValueError, match="frame buckets"):
        ArkttsDecodeGraphRunner(codec, num_codebooks=NUM_CODEBOOKS, frames=[50, -1], device=CUDA_DEVICE)
    with pytest.raises(ValueError, match="CUDA device"):
        ArkttsDecodeGraphRunner(codec, num_codebooks=NUM_CODEBOOKS, frames=[50], device=torch.device("cpu"))


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="CUDA required")
def test_graph_replay_matches_eager_bit_exactly() -> None:
    codec = _make_codec(CUDA_DEVICE)
    runner = ArkttsDecodeGraphRunner(codec, num_codebooks=NUM_CODEBOOKS, frames=CAPTURED_FRAMES, device=CUDA_DEVICE)
    assert runner.frames == tuple(CAPTURED_FRAMES)

    with torch.inference_mode():
        for seed in range(4):
            for frames in CAPTURED_FRAMES:
                for out_of_range in (False, True):
                    codes = _codes(frames, seed=seed, device=CUDA_DEVICE, out_of_range=out_of_range)
                    original = codes.clone()
                    graphed = runner.decode(codes)
                    eager = codec.decode(codes)
                    assert torch.equal(graphed, eager), f"frames={frames} seed={seed} oor={out_of_range} diverged"
                    assert graphed.shape == (1, 1, frames * 2048)
                    # The caller's codes are never mutated (the quantizer's
                    # in-place clamp must stay on the graph's internal clone).
                    assert torch.equal(codes, original)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="CUDA required")
def test_repeated_replays_are_deterministic_and_interleave_cleanly() -> None:
    codec = _make_codec(CUDA_DEVICE)
    runner = ArkttsDecodeGraphRunner(codec, num_codebooks=NUM_CODEBOOKS, frames=CAPTURED_FRAMES, device=CUDA_DEVICE)
    with torch.inference_mode():
        codes = {frames: _codes(frames, seed=11, device=CUDA_DEVICE) for frames in CAPTURED_FRAMES}
        first = {frames: runner.decode(codes[frames]) for frames in CAPTURED_FRAMES}
        # Eager decodes of other lengths between replays exercise the shared
        # mask/rope caches; the graphs' captured cache tensors must be stable.
        for frames in (4, 8, 24, 30, 51, 100):
            codec.decode(_codes(frames, seed=12, device=CUDA_DEVICE))
        for _ in range(3):
            for frames in CAPTURED_FRAMES:
                assert torch.equal(runner.decode(codes[frames]), first[frames])


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="CUDA required")
def test_uncaptured_shapes_fall_back() -> None:
    codec = _make_codec(CUDA_DEVICE)
    runner = ArkttsDecodeGraphRunner(codec, num_codebooks=NUM_CODEBOOKS, frames=CAPTURED_FRAMES, device=CUDA_DEVICE)
    with torch.inference_mode():
        for frames in (4, 24, 49, 50, 64):
            assert runner.can_decode(_codes(frames, seed=13, device=CUDA_DEVICE)) == (frames in CAPTURED_FRAMES)
        # Batched codes and 2-D codes are not graph shapes.
        assert not runner.can_decode(_codes(50, seed=13, device=CUDA_DEVICE).expand(2, -1, -1))
        assert not runner.can_decode(torch.zeros(NUM_CODEBOOKS, 50, dtype=torch.long, device=CUDA_DEVICE))


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="CUDA required")
def test_decoder_forward_matches_eager_with_graphs_enabled() -> None:
    """The decoder's forward output is identical with and without graphs."""
    from vllm_omni.model_executor.models.audio8_tts.audio8_tts_codec_decoder import Audio8TTSCodecDecoder

    def make_decoder(*, graph: bool):
        decoder = object.__new__(Audio8TTSCodecDecoder)
        torch.nn.Module.__init__(decoder)
        decoder._codec = _make_codec(CUDA_DEVICE)
        decoder._codec_device = CUDA_DEVICE
        decoder._codec_dtype = torch.float32
        decoder._num_codebooks = NUM_CODEBOOKS
        decoder._sample_rate = 44100
        decoder._frame_size = 2048
        decoder._logged_stats = True
        decoder._decode_graph = (
            ArkttsDecodeGraphRunner(
                decoder._codec, num_codebooks=NUM_CODEBOOKS, frames=CAPTURED_FRAMES, device=CUDA_DEVICE
            )
            if graph
            else None
        )
        decoder._ensure_codec_loaded = lambda: None
        return decoder

    graphed = make_decoder(graph=True)
    eager = make_decoder(graph=False)
    with torch.inference_mode():
        # A steady window [25 left | 25 new] and a first-steady window
        # [24 left | 25 new], as the connector ships them.
        for total, left in ((50, 25), (49, 24)):
            codes = _codes(total, seed=14, device=CUDA_DEVICE).squeeze(0)
            payload = [{"codes": {"audio": codes.clone()}, "meta": {"left_context_size": left}}]
            out_g = graphed.forward(
                input_ids=torch.tensor([0], dtype=torch.long), runtime_additional_information=payload
            )
            out_e = eager.forward(input_ids=torch.tensor([0], dtype=torch.long), runtime_additional_information=payload)
            audio_g = out_g.multimodal_outputs["model_outputs"][0]
            audio_e = out_e.multimodal_outputs["model_outputs"][0]
            assert audio_g.shape == audio_e.shape
            assert torch.equal(audio_g, audio_e)
            assert audio_g.numel() == (total - left) * 2048
        # A size the graphs do not cover decodes eagerly through the same path.
        codes = _codes(30, seed=15, device=CUDA_DEVICE).squeeze(0)
        payload = [{"codes": {"audio": codes}, "meta": {"left_context_size": 25}}]
        out = graphed.forward(input_ids=torch.tensor([0], dtype=torch.long), runtime_additional_information=payload)
        assert out.multimodal_outputs["model_outputs"][0].numel() == 5 * 2048
