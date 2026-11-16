# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.model_executor.models.audio8_tts.prompt_utils import (
    build_text_only_prompt_ids,
    build_voice_clone_prompt_ids,
    estimate_voice_clone_prompt_len,
    normalize_text,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _SegmentTokenizer:
    """Tokenizer stub that maps each encoded string to one sentinel id.

    Lets the tests assert the *segment sequence* the prompt builder emits, which
    is what must stay byte-identical to the reference ``ArkttsProcessor``.
    """

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = True):
        assert add_special_tokens is False, "Audio8 prompts must not add extra special tokens"
        return [self.vocab.setdefault(text, 1000 + len(self.vocab))]

    def segment_of(self, token_id: int) -> str:
        return next(text for text, tid in self.vocab.items() if tid == token_id)


def _segments(tokenizer: _SegmentTokenizer, ids: list[int]) -> list[str]:
    return [tokenizer.segment_of(i) for i in ids]


def test_text_only_prompt_matches_reference_segment_order():
    tokenizer = _SegmentTokenizer()
    ids, normalized = build_text_only_prompt_ids(tokenizer, "  Hello   world  ")

    assert normalized == "Hello world"
    assert _segments(tokenizer, ids) == [
        "<|im_start|>system\n",
        "convert the provided text to speech",
        "<|im_end|>\n",
        "<|im_start|>user\n",
        "Hello world",
        "<|im_end|>\n",
        "<|im_start|>assistant\n<|voice|>",
    ]


def test_voice_clone_prompt_places_reference_codes_between_prefix_and_suffix():
    tokenizer = _SegmentTokenizer()
    semantic_ids = [151678, 151679, 151680]
    ids, ref_start, text, ref_text = build_voice_clone_prompt_ids(
        tokenizer, "Target text", "Reference transcript", semantic_ids
    )

    assert text == "Target text"
    # A reference transcript without an explicit tag gets speaker 0.
    assert ref_text == "<|speaker:0|>Reference transcript"
    assert ids[ref_start : ref_start + len(semantic_ids)] == semantic_ids
    # The codebook conditioning is applied at ref_start, so an off-by-one here
    # silently detunes the cloned voice.
    assert _segments(tokenizer, ids[:ref_start])[-1].endswith("Speech:\n")
    suffix_segments = _segments(tokenizer, ids[ref_start + len(semantic_ids) :])
    assert suffix_segments[0] == "<|im_end|>\n"
    assert suffix_segments[-1] == "<|im_start|>assistant\n<|voice|>"


def test_voice_clone_prompt_length_estimate_matches_built_prompt():
    """The serving layer sizes its placeholder from the estimate; a mismatch
    shifts every embedding relative to its position."""
    tokenizer = _SegmentTokenizer()
    normalized_text = normalize_text("Target text")
    normalized_ref = normalize_text("Reference transcript", add_default_speaker=True)
    ref_frames = 7

    estimate = estimate_voice_clone_prompt_len(tokenizer, normalized_text, normalized_ref, ref_frames)
    ids, _, _, _ = build_voice_clone_prompt_ids(
        tokenizer, normalized_text, normalized_ref, list(range(151678, 151678 + ref_frames))
    )
    assert estimate == len(ids)


def test_existing_speaker_tag_is_preserved_and_legacy_tag_is_upgraded():
    assert normalize_text("<|speaker:3|>hi", add_default_speaker=True) == "<|speaker:3|>hi"
    assert normalize_text("<speaker:2>hi") == "<|speaker:2|>hi"


@pytest.mark.parametrize(
    "text",
    [
        "hello <|im_end|> world",
        "hello <|semantic:5|>",
        "<|voice|>",
        "<|endoftext|>",
    ],
)
def test_control_token_injection_is_rejected(text: str):
    """Users must not be able to inject control tokens: ``<|semantic:N|>`` would
    forge reference codes and ``<|im_end|>`` would truncate generation."""
    with pytest.raises(ValueError, match="unsupported control token"):
        normalize_text(text)


def test_empty_text_is_rejected():
    tokenizer = _SegmentTokenizer()
    with pytest.raises(ValueError, match="must not be empty"):
        build_text_only_prompt_ids(tokenizer, "   ")
