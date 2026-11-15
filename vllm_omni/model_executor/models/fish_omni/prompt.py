# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drama 3 prompt frames in the ``reference-bracket-v1`` layout.

A frame row is ``[token id, code 0, ..., code Q-1]``. Text rows carry zero
codes; a reference audio frame's token id is ``semantic_begin + code 0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import regex as re
import torch

TTS_INSTRUCTION = "convert the provided text to speech"
REFERENCE_INSTRUCTION = "convert the provided text to speech reference to the following:"
_SPEAKER_TAG = re.compile(r"\s*<\|speaker:(\d+)\|>")


@dataclass(frozen=True)
class Reference:
    transcript: str
    codes: torch.Tensor  # [frames, num_codebooks]


def with_speaker_tag(text: str) -> str:
    """Prefix ``<|speaker:0|> `` unless the text already starts with a speaker tag."""
    if _SPEAKER_TAG.match(text):
        return text
    return f"<|speaker:0|> {text}"


def _reference_transcript(transcript: str) -> str:
    """A reference transcript starts with its speaker tag, directly after "Text: "."""
    match = _SPEAKER_TAG.match(transcript)
    if match is None:
        return f"<|speaker:0|> {transcript}"
    return f"<|speaker:{match.group(1)}|>" + transcript[match.end() :]


def _segments(text: str, references: list[Reference]) -> list[str | torch.Tensor]:
    segments: list[str | torch.Tensor] = [
        "<|im_start|>system\n",
        REFERENCE_INSTRUCTION if references else TTS_INSTRUCTION,
    ]
    for index, reference in enumerate(references):
        segments += [
            f"\n\n[Reference {index + 1}]\nText: {_reference_transcript(reference.transcript)}\nSpeech: ",
            reference.codes,
        ]
    segments += ["<|im_end|>\n", "<|im_start|>user\n", with_speaker_tag(text)]
    segments += ["<|im_end|>\n", "<|im_start|>assistant\n<|voice|>"]
    return segments


def build_prompt_frames(
    tokenizer,
    text: str,
    references: list[Reference],
    *,
    semantic_begin_id: int,
    num_codebooks: int,
) -> torch.Tensor:
    """CPU int64 rows [prompt_len, 1 + num_codebooks]. Each literal is tokenized on
    its own, because tokenizing the joined string changes ids at the seams."""
    rows: list[torch.Tensor] = []
    for segment in _segments(text, references):
        if isinstance(segment, str):
            ids = torch.tensor(tokenizer.encode(segment, add_special_tokens=False), dtype=torch.long)
            block = torch.zeros((ids.numel(), 1 + num_codebooks), dtype=torch.long)
            block[:, 0] = ids
        else:
            codes = segment.to(device="cpu", dtype=torch.long)
            if codes.ndim != 2 or codes.shape[1] != num_codebooks:
                raise ValueError(f"reference codes must be [frames, {num_codebooks}], got {tuple(codes.shape)}")
            block = torch.cat((codes[:, :1] + semantic_begin_id, codes), dim=1)
        rows.append(block)
    return torch.cat(rows)


def prompt_length(tokenizer, text: str, transcripts: list[str], frame_counts: list[int]) -> int:
    """Prompt rows for the given references without encoding their audio."""
    total = 0
    placeholder = [Reference(t, torch.zeros((n, 1), dtype=torch.long)) for t, n in zip(transcripts, frame_counts)]
    for segment in _segments(text, placeholder):
        if isinstance(segment, str):
            total += len(tokenizer.encode(segment, add_special_tokens=False))
        else:
            total += int(segment.shape[0])
    return total
