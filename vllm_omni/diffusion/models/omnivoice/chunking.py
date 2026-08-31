# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 Xiaomi Corp. (authors: Han Zhu)

from __future__ import annotations

import torch

_SENTENCE_BOUNDARIES = frozenset(".,;:!?。，；：！？")
_CLAUSE_BOUNDARIES = frozenset(",;:，；：、")
_CLOSING_MARKS = frozenset("\"'“”‘’）]》>」】")
_ABBREVIATIONS = frozenset(
    {
        "Mr.",
        "Mrs.",
        "Ms.",
        "Dr.",
        "Prof.",
        "Sr.",
        "Jr.",
        "Rev.",
        "Fr.",
        "Hon.",
        "Pres.",
        "Gov.",
        "Capt.",
        "Gen.",
        "Sen.",
        "Rep.",
        "Col.",
        "Maj.",
        "Lt.",
        "Cmdr.",
        "Sgt.",
        "Cpl.",
        "Co.",
        "Corp.",
        "Inc.",
        "Ltd.",
        "Est.",
        "Dept.",
        "St.",
        "Ave.",
        "Blvd.",
        "Rd.",
        "Mt.",
        "Ft.",
        "No.",
        "Jan.",
        "Feb.",
        "Mar.",
        "Apr.",
        "Aug.",
        "Sep.",
        "Sept.",
        "Oct.",
        "Nov.",
        "Dec.",
        "i.e.",
        "e.g.",
        "vs.",
        "Vs.",
        "Etc.",
        "approx.",
        "fig.",
        "def.",
        "apt.",
        "D.I.Y.",
        "D.I.Y",
        "R.S.V.P.",
        "R.S.V.P",
        "P.S.",
        "P.S",
        "al.",
    }
)


def _period_belongs_to_abbreviation(text: str, period_index: int) -> bool:
    word_start = period_index
    while word_start > 0 and not text[word_start - 1].isspace():
        word_start -= 1

    word_end = period_index + 1
    while word_end < len(text) and not text[word_end].isspace():
        word_end += 1

    word = text[word_start:word_end].rstrip("".join(_CLOSING_MARKS | (_SENTENCE_BOUNDARIES - {"."})))
    return word in _ABBREVIATIONS


def _split_at_sentence_boundaries(text: str) -> list[str]:
    sentences: list[list[str]] = []
    current_sentence: list[str] = []

    for index, character in enumerate(text):
        if not current_sentence and sentences and (character in _SENTENCE_BOUNDARIES or character in _CLOSING_MARKS):
            sentences[-1].append(character)
            continue

        current_sentence.append(character)
        if character not in _SENTENCE_BOUNDARIES:
            continue
        if character == "." and _period_belongs_to_abbreviation(text, index):
            continue

        sentences.append(current_sentence)
        current_sentence = []

    if current_sentence:
        sentences.append(current_sentence)
    return ["".join(sentence) for sentence in sentences]


def _merge_sentences(sentences: list[str], max_characters: int) -> list[str]:
    chunks: list[str] = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_characters:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def _merge_short_chunks(
    chunks: list[str],
    min_characters: int,
    max_characters: int,
) -> list[str]:
    if len(chunks) < 2:
        return chunks
    if len(chunks[0]) < min_characters and len(chunks[0]) + len(chunks[1]) <= max_characters:
        chunks[1] = chunks[0] + chunks[1]
        chunks = chunks[1:]

    merged: list[str] = []
    for chunk in chunks:
        if len(chunk) < min_characters and merged and len(merged[-1]) + len(chunk) <= max_characters:
            merged[-1] += chunk
        else:
            merged.append(chunk)
    return merged


def _split_oversized_chunk(text: str, max_characters: int) -> list[str]:
    chunks: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_characters:
        window = remaining[:max_characters]
        split_index = max(
            (index + 1 for index, character in enumerate(window) if character in _CLAUSE_BOUNDARIES),
            default=0,
        )
        if split_index == 0:
            split_index = max(
                (index for index, character in enumerate(window) if character.isspace()),
                default=0,
            )
        if split_index == 0:
            split_index = max_characters

        chunks.append(remaining[:split_index].strip())
        remaining = remaining[split_index:].strip()

    if remaining:
        chunks.append(remaining)
    return chunks


def split_text_into_chunks(
    text: str,
    max_characters: int,
    min_characters: int = 3,
) -> list[str]:
    """Split text at sentence boundaries, then bound oversized sentences."""
    if max_characters <= 0:
        raise ValueError("max_characters must be positive")

    sentences = _split_at_sentence_boundaries(text)
    chunks = _merge_short_chunks(
        _merge_sentences(sentences, max_characters),
        min_characters,
        max_characters,
    )
    return [
        bounded_chunk
        for chunk in chunks
        for bounded_chunk in _split_oversized_chunk(chunk, max_characters)
        if bounded_chunk
    ]


def join_audio_chunks(
    chunks: list[torch.Tensor],
    sample_rate: int,
    silence_duration_seconds: float = 0.3,
) -> torch.Tensor:
    """Join decoded audio chunks with a fade and short silence at each boundary."""
    if not chunks:
        raise ValueError("chunks must not be empty")
    if len(chunks) == 1:
        return chunks[0]

    boundary_samples = int(silence_duration_seconds * sample_rate) // 3
    # Work on copies so changing each chunk's start and end does not change the input audio.
    joined_parts = [chunks[0].clone()]
    for chunk in chunks[1:]:
        previous_chunk = joined_parts[-1]
        fade_out_samples = min(boundary_samples, previous_chunk.shape[-1])
        if fade_out_samples:
            fade_out = torch.linspace(
                1,
                0,
                fade_out_samples,
                dtype=previous_chunk.dtype,
                device=previous_chunk.device,
            )
            previous_chunk[..., -fade_out_samples:] *= fade_out

        fade_in_audio = chunk.clone()
        fade_in_samples = min(boundary_samples, fade_in_audio.shape[-1])
        if fade_in_samples:
            fade_in = torch.linspace(
                0,
                1,
                fade_in_samples,
                dtype=fade_in_audio.dtype,
                device=fade_in_audio.device,
            )
            fade_in_audio[..., :fade_in_samples] *= fade_in

        silence = torch.zeros(
            (*previous_chunk.shape[:-1], boundary_samples),
            dtype=previous_chunk.dtype,
            device=previous_chunk.device,
        )
        joined_parts.extend([silence, fade_in_audio])
    return torch.cat(joined_parts, dim=-1)
