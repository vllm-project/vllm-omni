# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for Fish Speech DAC decoder batched forward path.

Verifies that the DAC decoder correctly handles batch_size > 1 when
max_num_seqs is raised from 1 to 4 in the deploy config, including
correct padding, per-request audio extraction, and context trimming.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.fish_speech.fish_speech_dac_decoder import (
    FishSpeechDACDecoder,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_NUM_CODEBOOKS = 10
_SAMPLE_RATE = 44100
_HOP_LENGTH = 2048


def _make_codec_mock(hop_length: int = _HOP_LENGTH):
    """Create a mock DAC codec that returns deterministic audio.

    The mock decode() returns a waveform where each sample equals the
    batch index + 1, so we can verify per-request independence.
    """
    codec = MagicMock()
    codec.quantizer = SimpleNamespace(
        semantic_codebook_size=4096,
        num_codebooks=_NUM_CODEBOOKS - 1,
    )

    def _decode(codes_bqf, feature_lengths):
        B = codes_bqf.shape[0]
        max_frames = codes_bqf.shape[2]
        max_wav_len = max_frames * hop_length
        wav = torch.zeros((B, 1, max_wav_len), dtype=torch.float32)
        audio_lengths = torch.zeros(B, dtype=torch.long)
        for i in range(B):
            n_frames = int(feature_lengths[i].item())
            wav_len = n_frames * hop_length
            # Fill with (i+1) so each request has a unique value
            wav[i, 0, :wav_len] = float(i + 1)
            audio_lengths[i] = wav_len
        return wav, audio_lengths

    codec.decode = _decode
    return codec


def _make_model(codec=None):
    """Build a minimal FishSpeechDACDecoder without __init__."""
    model = object.__new__(FishSpeechDACDecoder)
    nn.Module.__init__(model)
    model._codec = codec or _make_codec_mock()
    model._num_codebooks = _NUM_CODEBOOKS
    model._output_sample_rate = _SAMPLE_RATE
    model._logged_codec_stats = True  # suppress one-shot log
    model._codec_decode_takes_lengths = True
    return model


def _make_flat_codes(num_frames: int, q: int = _NUM_CODEBOOKS) -> torch.Tensor:
    """Create flat codec token IDs: q * num_frames elements."""
    return torch.randint(0, 1000, (q * num_frames,), dtype=torch.long)


class TestDACDecoderBatchForward:
    """Tests for batched DAC decoder forward()."""

    def test_single_request_produces_audio(self):
        """B=1 forward produces non-empty audio at the right sample rate."""
        model = _make_model()
        codes = _make_flat_codes(num_frames=10)

        out = model.forward(
            input_ids=codes,
            positions=torch.arange(codes.numel()),
            seq_token_counts=[codes.numel()],
        )

        audios = out.multimodal_outputs["model_outputs"]
        assert len(audios) == 1
        assert audios[0].numel() == 10 * _HOP_LENGTH
        assert audios[0].dtype == torch.float32

        srs = out.multimodal_outputs["sr"]
        assert len(srs) == 1
        assert srs[0].item() == _SAMPLE_RATE

    def test_batch_of_two_produces_independent_audio(self):
        """B=2 forward produces two independent audio outputs."""
        model = _make_model()
        codes_a = _make_flat_codes(num_frames=8)
        codes_b = _make_flat_codes(num_frames=12)
        combined = torch.cat([codes_a, codes_b])

        out = model.forward(
            input_ids=combined,
            positions=torch.arange(combined.numel()),
            seq_token_counts=[codes_a.numel(), codes_b.numel()],
        )

        audios = out.multimodal_outputs["model_outputs"]
        assert len(audios) == 2
        assert audios[0].numel() == 8 * _HOP_LENGTH
        assert audios[1].numel() == 12 * _HOP_LENGTH
        # Verify independence: mock fills each row with (i+1)
        assert torch.all(audios[0] == 1.0)
        assert torch.all(audios[1] == 2.0)

    def test_batch_of_four_different_lengths(self):
        """B=4 forward with different frame counts per request."""
        model = _make_model()
        frame_counts = [5, 10, 15, 20]
        codes_list = [_make_flat_codes(n) for n in frame_counts]
        combined = torch.cat(codes_list)
        token_counts = [c.numel() for c in codes_list]

        out = model.forward(
            input_ids=combined,
            positions=torch.arange(combined.numel()),
            seq_token_counts=token_counts,
        )

        audios = out.multimodal_outputs["model_outputs"]
        assert len(audios) == 4
        for i, n in enumerate(frame_counts):
            assert audios[i].numel() == n * _HOP_LENGTH, f"req {i}"
            assert torch.all(audios[i] == float(i + 1)), f"req {i} cross-contaminated"

    def test_mixed_valid_and_empty_requests(self):
        """Empty requests get zero-length tensors, valid ones get audio."""
        model = _make_model()
        codes_valid = _make_flat_codes(num_frames=10)
        codes_empty = torch.tensor([], dtype=torch.long)
        combined = torch.cat([codes_empty, codes_valid])

        out = model.forward(
            input_ids=combined,
            positions=torch.arange(combined.numel()),
            seq_token_counts=[0, codes_valid.numel()],
        )

        audios = out.multimodal_outputs["model_outputs"]
        assert len(audios) == 2
        assert audios[0].numel() == 0
        assert audios[1].numel() == 10 * _HOP_LENGTH

    def test_context_trimming_with_batch(self):
        """Left context trimming works correctly in batched mode."""
        model = _make_model()
        codes_a = _make_flat_codes(num_frames=20)
        codes_b = _make_flat_codes(num_frames=20)
        combined = torch.cat([codes_a, codes_b])

        # Request 0 has 5 frames of left context, request 1 has none
        runtime_info = [
            {"meta": {"left_context_size": 5}},
            {"meta": {}},
        ]

        out = model.forward(
            input_ids=combined,
            positions=torch.arange(combined.numel()),
            seq_token_counts=[codes_a.numel(), codes_b.numel()],
            runtime_additional_information=runtime_info,
        )

        audios = out.multimodal_outputs["model_outputs"]
        assert len(audios) == 2
        # Request 0: 20 frames, 5 trimmed -> 15 frames of audio
        assert audios[0].numel() == 15 * _HOP_LENGTH
        # Request 1: 20 frames, 0 trimmed -> 20 frames of audio
        assert audios[1].numel() == 20 * _HOP_LENGTH

    def test_all_invalid_requests_skip_decoder(self):
        """All-invalid batch returns empty tensors without calling decode."""
        codec = _make_codec_mock()
        codec.decode = MagicMock(side_effect=AssertionError("should not be called"))
        model = _make_model(codec=codec)

        # 3 tokens not divisible by num_codebooks=10
        bad = torch.tensor([1, 2, 3], dtype=torch.long)

        out = model.forward(
            input_ids=bad,
            positions=torch.arange(3),
            seq_token_counts=[3],
        )

        audios = out.multimodal_outputs["model_outputs"]
        assert len(audios) == 1
        assert audios[0].numel() == 0
