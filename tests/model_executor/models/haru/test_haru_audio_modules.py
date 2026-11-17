# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Haru codec streaming, conversation prompt and text-stream rules on tiny modules (CPU)."""

from __future__ import annotations

import pytest
import torch

from tests.model_executor.models.haru.conftest import EOP, PAD, TINY_SPEC
from vllm_omni.model_executor.models.haru.prompt import build_prompt, prompt_length, silence_latent
from vllm_omni.model_executor.models.haru.sampling import DryConfig, DryState, StreamDetokenizer, dry_penalties
from vllm_omni.model_executor.models.haru.vibevoice_codec import CodecStream, open_stream

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Encoding:
    def __init__(self, ids: list[int]):
        self.ids = ids


class _Tokenizer:
    """``encode`` gives one id per character; ``decode`` holds back an odd-length run (a split character)."""

    def encode(self, text: str, add_special_tokens: bool = False) -> _Encoding:
        return _Encoding([ord(ch) % 16 for ch in text])

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "�" if len(ids) % 2 else "".join(chr(97 + i % 26) for i in ids[::2])


def test_streaming_codec_matches_whole_clip_encode_and_decode(tiny_codec) -> None:
    frames, fs = 6, TINY_SPEC.frame_samples
    audio = torch.randn(1, frames * fs)
    latents = torch.randn(1, frames, TINY_SPEC.latent_dim)

    stream = open_stream(tiny_codec, backend="auto")  # CPU float32: the PyTorch stream
    assert isinstance(stream, CodecStream)
    encoded = torch.cat([stream.encode(audio[:, k * fs : (k + 1) * fs]) for k in range(frames)], dim=1)
    decoded = torch.cat([stream.decode(latents[:, k : k + 1]) for k in range(frames)], dim=1)

    torch.testing.assert_close(encoded, tiny_codec.encode(audio))
    torch.testing.assert_close(decoded, tiny_codec.decode(latents))


def test_streaming_encode_takes_whole_frames_only(tiny_codec) -> None:
    with pytest.raises(ValueError, match="whole frames"):
        CodecStream(tiny_codec).encode(torch.zeros(1, TINY_SPEC.frame_samples + 1))


@pytest.mark.parametrize(
    ("card", "reference_frames", "speak_first"),
    [("", 0, False), ("hi", 0, False), ("", 3, False), ("hi", 3, False), ("hi", 3, True), ("", 0, True)],
)
def test_prompt_length_matches_the_built_prompt(tiny_codec, card: str, reference_frames: int, speak_first: bool):
    fs = TINY_SPEC.frame_samples
    reference = torch.randn(reference_frames * fs - 1) if reference_frames else None
    samples = 0 if reference is None else reference.numel()
    generator = torch.Generator().manual_seed(0)

    prompt = build_prompt(TINY_SPEC, tiny_codec, _Tokenizer(), card, reference, speak_first, generator)

    assert prompt.length == prompt_length(_Tokenizer(), card, samples, speak_first, fs)
    n_card = len(_Tokenizer().encode(f"<|im_start|>system\n{card}<|im_end|>\n").ids) if card else 0
    tokens = prompt.tokens.tolist()
    if reference_frames:
        assert tokens[n_card : n_card + reference_frames] == [PAD] * (reference_frames - 1) + [EOP]
    # The card carries no audio; the user stream starts only after the reference.
    assert not prompt.model_mask[:n_card].any() and prompt.model_mask[n_card:].all()
    assert not prompt.user_mask[: n_card + reference_frames].any()
    if speak_first or (not card and not reference_frames):
        torch.testing.assert_close(prompt.user_latents[-1], silence_latent(tiny_codec))


def test_dry_penalises_the_token_that_would_extend_a_repeat() -> None:
    config = DryConfig()
    assert dry_penalties([1, 2, 3, 1, 2], config) == {3: pytest.approx(config.multiplier * config.base)}
    assert dry_penalties([1, 2, 3], config) == {}

    state = DryState(config, exempt={PAD, EOP})
    for token in (1, 2, PAD, 3, 1, EOP, 2):
        state.update(token)
    assert state.history == [1, 2, 3, 1, 2]
    logits = torch.zeros(8)
    state.apply(logits)
    assert logits[3] == pytest.approx(-config.multiplier * config.base)
    assert logits.count_nonzero() == 1


def test_stream_detokenizer_renders_silence_once_and_holds_split_characters() -> None:
    detok = StreamDetokenizer(_Tokenizer(), silent_ids={PAD, EOP})
    assert [detok.push(t) for t in (PAD, EOP, PAD)] == [" ", "", ""]
    assert detok.push(3) == ""  # first half of a character
    assert detok.push(9) == "d"
    assert detok.push(PAD) == " "
