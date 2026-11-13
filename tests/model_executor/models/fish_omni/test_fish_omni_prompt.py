# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the Drama 3 (fish_omni) prompt frames."""

import pytest
import torch

from vllm_omni.model_executor.models.fish_omni.prompt import (
    REFERENCE_INSTRUCTION,
    TTS_INSTRUCTION,
    Reference,
    build_prompt_frames,
    prompt_length,
    with_speaker_tag,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SEMANTIC_BEGIN = 1000
NUM_CODEBOOKS = 3


class CharTokenizer:
    """One id per character, and a log of every encoded literal."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert add_special_tokens is False
        self.calls.append(text)
        return [ord(c) for c in text]


def _decode(ids: torch.Tensor) -> str:
    return "".join(chr(int(i)) for i in ids)


def _build(tokenizer: CharTokenizer, text: str, references: list[Reference]) -> torch.Tensor:
    return build_prompt_frames(
        tokenizer, text, references, semantic_begin_id=SEMANTIC_BEGIN, num_codebooks=NUM_CODEBOOKS
    )


def test_no_reference_layout():
    tokenizer = CharTokenizer()
    frames = _build(tokenizer, "Hello there.", [])

    assert tokenizer.calls == [
        "<|im_start|>system\n",
        TTS_INSTRUCTION,
        "<|im_end|>\n",
        "<|im_start|>user\n",
        "<|speaker:0|> Hello there.",
        "<|im_end|>\n",
        "<|im_start|>assistant\n<|voice|>",
    ]
    assert frames.dtype == torch.long
    assert frames.shape == (len("".join(tokenizer.calls)), 1 + NUM_CODEBOOKS)
    assert _decode(frames[:, 0]) == "".join(tokenizer.calls)
    assert torch.all(frames[:, 1:] == 0)


def test_one_reference_layout():
    codes = torch.tensor([[5, 1, 2], [7, 3, 4]])
    tokenizer = CharTokenizer()
    frames = _build(tokenizer, "Hello.", [Reference("We asked twenty people.", codes)])

    head = f"<|im_start|>system\n{REFERENCE_INSTRUCTION}\n\n[Reference 1]\nText: <|speaker:0|> We asked twenty people.\nSpeech: "
    tail = "<|im_end|>\n<|im_start|>user\n<|speaker:0|> Hello.<|im_end|>\n<|im_start|>assistant\n<|voice|>"
    assert tokenizer.calls == [
        "<|im_start|>system\n",
        REFERENCE_INSTRUCTION,
        "\n\n[Reference 1]\nText: <|speaker:0|> We asked twenty people.\nSpeech: ",
        "<|im_end|>\n",
        "<|im_start|>user\n",
        "<|speaker:0|> Hello.",
        "<|im_end|>\n",
        "<|im_start|>assistant\n<|voice|>",
    ]
    assert frames.shape == (len(head) + 2 + len(tail), 1 + NUM_CODEBOOKS)
    assert _decode(frames[: len(head), 0]) == head
    assert torch.all(frames[: len(head), 1:] == 0)
    # A reference frame row is [semantic_begin + code 0, code 0, ..., code Q-1].
    audio = frames[len(head) : len(head) + 2]
    assert audio.tolist() == [[SEMANTIC_BEGIN + 5, 5, 1, 2], [SEMANTIC_BEGIN + 7, 7, 3, 4]]
    assert _decode(frames[len(head) + 2 :, 0]) == tail
    assert torch.all(frames[len(head) + 2 :, 1:] == 0)


def test_reference_codes_shape_is_checked():
    with pytest.raises(ValueError, match="reference codes must be"):
        _build(CharTokenizer(), "Hi.", [Reference("ref", torch.zeros((4, NUM_CODEBOOKS + 1), dtype=torch.long))])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello.", "<|speaker:0|> Hello."),
        ("<|speaker:3|> Hello.", "<|speaker:3|> Hello."),
        ("<|speaker:12|>Hello.", "<|speaker:12|>Hello."),
        ("  <|speaker:1|> Hello.", "  <|speaker:1|> Hello."),
        ("Hi. <|speaker:1|> Hello.", "<|speaker:0|> Hi. <|speaker:1|> Hello."),
        ("[sighs] Fine.", "<|speaker:0|> [sighs] Fine."),
    ],
)
def test_text_speaker_tag(text, expected):
    assert with_speaker_tag(text) == expected


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("Some words.", "Text: <|speaker:0|> Some words.\n"),
        ("<|speaker:2|> Some words.", "Text: <|speaker:2|> Some words.\n"),
        # Leading white space before the tag is dropped: the tag follows "Text: ".
        ("  <|speaker:2|> Some words.", "Text: <|speaker:2|> Some words.\n"),
    ],
)
def test_reference_transcript_speaker_tag(transcript, expected):
    tokenizer = CharTokenizer()
    _build(tokenizer, "Hi.", [Reference(transcript, torch.zeros((1, NUM_CODEBOOKS), dtype=torch.long))])
    assert expected in tokenizer.calls[2]


def test_inline_event_tags_pass_through():
    tokenizer = CharTokenizer()
    text = "<|speaker:0|> [sighs] Fine. [laughs] Well."
    frames = _build(tokenizer, text, [])
    assert text in _decode(frames[:, 0])


@pytest.mark.parametrize("frame_counts", [[], [1], [37]])
def test_prompt_length_matches_frames(frame_counts):
    text = "<|speaker:1|> The quarterly report is due on Thursday."
    transcripts = ["We asked over twenty different people."] * len(frame_counts)
    references = [Reference(t, torch.randint(0, 64, (n, NUM_CODEBOOKS))) for t, n in zip(transcripts, frame_counts)]
    rows = int(_build(CharTokenizer(), text, references).shape[0])
    assert prompt_length(CharTokenizer(), text, transcripts, frame_counts) == rows
