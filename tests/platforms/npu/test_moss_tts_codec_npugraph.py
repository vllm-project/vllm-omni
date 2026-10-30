# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend hardware validation for Moss-TTS codec streaming NPUGraph replay.

Mirrors ``test_codec_cudagraph.py`` (non-streaming CUDA wrapper) but exercises
the streaming :class:`NPUGraphStreamingDecoderWrapper` which captures
``decode_streaming_tensors`` per ``(B_bucket, exact_T)`` signature.

The synthetic codec is stateless — graph vs eager parity is verified for the
same inputs.  State management (RingKVCache in-place updates) is covered by
the E2E integration test (correct audio output confirmed on 910B).
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytest.importorskip("vllm_ascend")

from tests.helpers.mark import hardware_marks
from vllm_omni.platforms.npu.models.moss_tts_streaming_decode_wrapper import (
    NPUGraphStreamingDecoderWrapper,
)

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.omni,
    *hardware_marks(res={"npu": "A3"}, num_cards=1),
]

DEVICE = torch.device("npu:0")
NQ = 4
UPSAMPLE = 4
HIDDEN = 32


class SyntheticStreamingCodec(nn.Module):
    """Minimal stateless stand-in for MossAudioTokenizerModel streaming decode.

    Exposes ``decode_streaming_tensors`` + ``reset_decoder_state_slots`` with
    the same interface the NPU wrapper expects.
    """

    def __init__(self):
        super().__init__()
        self.downsample_rate = UPSAMPLE
        self.embed = nn.Conv1d(NQ, HIDDEN, kernel_size=3, padding=1)
        self.conv = nn.Conv1d(HIDDEN, HIDDEN, kernel_size=3, padding=1)
        self.up = nn.ConvTranspose1d(HIDDEN, 1, kernel_size=UPSAMPLE, stride=UPSAMPLE)

    def decode_streaming_tensors(self, codes, lengths, slot_ids, valid_rows):
        """codes [NQ,B,T] lengths [B] slot_ids [B] valid [B] -> audio [B,1,T*up]."""
        nq, b, t = codes.shape
        x = codes.permute(1, 0, 2).to(dtype=self.embed.weight.dtype)
        x = torch.relu(self.embed(x))
        x = torch.relu(self.conv(x))
        audio = self.up(x)
        audio_lengths = lengths * self.downsample_rate
        return audio, audio_lengths

    def reset_decoder_state_slots(self, slot_ids):
        pass


@pytest.fixture(scope="module")
def codec():
    torch.manual_seed(0)
    return SyntheticStreamingCodec().to(DEVICE).eval()


@pytest.fixture(scope="module")
def wrapper(codec):
    w = NPUGraphStreamingDecoderWrapper(
        codec=codec,
        state_capacity=16,
        batch_sizes=[1, 2, 4],
        frame_sizes=[4, 8],
        num_quantizers=NQ,
        vllm_config=SimpleNamespace(),
    )
    w.warmup(DEVICE)
    return w


def _codes(b, t, device=DEVICE):
    return torch.randint(0, 64, (NQ, b, t), dtype=torch.long, device=device)


def _eager(codec, codes, slot_ids):
    b = codes.shape[1]
    t = codes.shape[2]
    lengths = torch.full((b,), t, dtype=torch.long, device=codes.device)
    valid = torch.ones(b, dtype=torch.bool, device=codes.device)
    with torch.no_grad():
        return codec.decode_streaming_tensors(codes, lengths, slot_ids, valid)


# ---------------------------------------------------------------------------
# 1. Exact-size decode — graph replay must match eager
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("b,t", [(1, 4), (2, 8), (4, 4)])
def test_exact_size_replay_matches_eager(codec, wrapper, b, t):
    slots = torch.arange(b, dtype=torch.long, device=DEVICE)
    codes = _codes(b, t)
    ref = _eager(codec, codes, slots)
    with torch.no_grad():
        out = wrapper.decode(codes, slots)
    assert out is not None
    audio, _, actual_b = out
    assert actual_b == b
    torch.testing.assert_close(audio[:b], ref[0][:b], atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# 2. Padded batch — B < bucket or T < frame_size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("b,t", [(1, 5), (2, 4), (3, 6)])
def test_padded_batch_interior_matches(codec, wrapper, b, t):
    """Padded batches must match eager run on the same padded input.

    The graph wrapper zero-pads codes to the next frame_size bucket.  We run
    eager on the same padded codes so both paths see identical input.
    """
    slots = torch.arange(b, dtype=torch.long, device=DEVICE)
    codes = _codes(b, t)
    graph_frame_size = next(fs for fs in wrapper.frame_sizes if fs >= t)
    padded_codes = torch.zeros(NQ, b, graph_frame_size, dtype=torch.long, device=DEVICE)
    padded_codes[:, :, :t] = codes
    lengths = torch.full((b,), t, dtype=torch.long, device=DEVICE)
    valid = torch.ones(b, dtype=torch.bool, device=DEVICE)
    with torch.no_grad():
        ref = codec.decode_streaming_tensors(padded_codes, lengths, slots, valid)
    with torch.no_grad():
        out = wrapper.decode(codes, slots, allow_frame_padding=True)
    assert out is not None
    audio, audio_lengths, actual_b = out
    assert actual_b == b
    expected_len = t * UPSAMPLE
    for row in range(b):
        assert int(audio_lengths[row]) == expected_len
        torch.testing.assert_close(audio[row], ref[0][row], atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# 3. Terminal tail — finished request with T < frame_size, padded
# ---------------------------------------------------------------------------


def test_terminal_tail_matches_eager(codec, wrapper):
    """Terminal tail (T=3 padded to frame_size=4) must match eager on the
    same padded input."""
    slots = torch.tensor([0], dtype=torch.long, device=DEVICE)
    t = 3
    codes = _codes(1, t)
    graph_frame_size = next(fs for fs in wrapper.frame_sizes if fs >= t)
    padded_codes = torch.zeros(NQ, 1, graph_frame_size, dtype=torch.long, device=DEVICE)
    padded_codes[:, :, :t] = codes
    lengths = torch.tensor([t], dtype=torch.long, device=DEVICE)
    valid = torch.ones(1, dtype=torch.bool, device=DEVICE)
    with torch.no_grad():
        ref = codec.decode_streaming_tensors(padded_codes, lengths, slots, valid)
    with torch.no_grad():
        out = wrapper.decode(codes, slots, allow_frame_padding=True)
    assert out is not None
    audio, audio_lengths, _ = out
    assert int(audio_lengths[0]) == t * UPSAMPLE
    torch.testing.assert_close(audio[0], ref[0][0], atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# 4. Output not aliased across replays (static buffer clone)
# ---------------------------------------------------------------------------


def test_output_not_aliased(codec, wrapper):
    """wrapper.decode returns cloned audio; a later replay must not overwrite
    a previously returned tensor."""
    slot = torch.tensor([0], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out1 = wrapper.decode(_codes(1, 4), slot)
        saved = out1[0].clone()
        out2 = wrapper.decode(_codes(1, 4), slot)
    assert out1 is not None and out2 is not None
    torch.testing.assert_close(out1[0], saved, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 5. Unsupported device returns None (graceful skip)
# ---------------------------------------------------------------------------


def test_unsupported_device_returns_none(wrapper):
    """decode on a non-NPU tensor must return None (no graph path)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for cross-device test")
    codes = _codes(1, 4, device=torch.device("cuda:0"))
    slots = torch.tensor([0], dtype=torch.long, device=codes.device)
    out = wrapper.decode(codes, slots)
    assert out is None
