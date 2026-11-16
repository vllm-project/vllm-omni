# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Prompt normalization follows the MiniMaxAI/MiniMax-Music3 input contract.
# The transformations are reproduced byte-for-byte because any divergence
# shifts token positions and therefore changes the generated audio.
"""MiniMax Music 3 prompt construction.

The prompt is a single flat string, not a chat template::

    <|im_start|><|caption_start|>{caption}<|caption_end|>
    <|lyrics_start|>{lyrics}<|lyrics_end|><|im_end|><|audio_start|>

The classifier-free-guidance twin reuses the same token ids with the whole
caption-and-lyrics span (delimiters included) overwritten by
``<|audio_cfg|>``, so only the chat markers and ``<|audio_start|>`` survive.
Both rows therefore have identical length and identical audio-start position.
"""

from __future__ import annotations

from typing import Any

import regex as re

# Pinned by ``validate_tokenizer_ids``: a checkpoint whose tokenizer disagrees
# is not this model, and silently producing noise is worse than failing.
SPECIAL_TOKEN_IDS: dict[str, int] = {
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|audio_cfg|>": 151654,
    "<|audio_start|>": 151669,
    "<|audio_end|>": 151670,
    "<|caption_start|>": 151671,
    "<|caption_end|>": 151672,
    "<|lyrics_start|>": 151673,
    "<|lyrics_end|>": 151674,
}

# c0 codes occupy the backbone vocabulary immediately above the text tokens.
AUDIO_CODE_OFFSET = 151675

_SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_LEADING_TAGS_RE = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")


def _strip_markdown_residuals(text: str) -> str:
    """Strip the Markdown syntax the model input contract does not accept."""
    if not text:
        return text
    lines_out: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[*+-]\s+", "", line)
        line = re.sub(r"^\s*\*\s+", "", line)
        while "**" in line:
            updated = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            if updated == line:
                break
            line = updated
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        lines_out.append(line.rstrip())
    text = "\n".join(lines_out)
    return re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)


def _remove_markdown_format(text: str) -> str:
    return _strip_markdown_residuals(text).replace("• ", "").replace("    ", "")


def clean_caption(caption: str) -> str:
    """Apply the source order: special tokens, then Markdown, then newlines.

    ``<|tag value|>`` forms are rewritten to ``tag is value`` so a caption
    cannot inject a real special token into the prompt.
    """

    def _special(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        parts = inner.split(None, 1)
        return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else inner

    text = _SPECIAL_TAG_RE.sub(_special, caption)
    text = _remove_markdown_format(text)
    # The source collapses runs of two or more newlines, not three or more;
    # this shifts blank-line token positions and therefore the audio.
    return re.sub(r"\n{2,}", "\n", text)


def _strip_text_after_leading_tags(text: str) -> str:
    """Keep only the consecutive structural tags that start a line.

    Lyrics written on the same line as a ``[Verse]`` tag are dropped, which is
    the documented contract: a tag must sit on its own line.
    """
    if not text:
        return text
    output: list[str] = []
    for line in text.split("\n"):
        match = _LEADING_TAGS_RE.match(line)
        output.append(match.group(1).strip() if match else line)
    return "\n".join(output)


def normalize_lyrics(lyrics: str) -> str:
    """Normalize lyrics to the form the checkpoint was trained on."""
    text = _strip_text_after_leading_tags(lyrics)
    text = text.replace("] ", "]\n")
    text = text.replace(" [", "\n[")
    text = text.replace(" ^ ", "\n")
    text = re.sub(r"\[([^\]]+)\]", lambda match: f"[{match.group(1).lower()}]", text)
    return f"[start]\n{text}"


def build_prompt(caption: str, lyrics: str) -> str:
    """Build the conditioned prompt string."""
    return (
        "<|im_start|><|caption_start|>"
        f"{clean_caption(caption)}"
        "<|caption_end|><|lyrics_start|>"
        f"{normalize_lyrics(lyrics)}"
        "<|lyrics_end|><|im_end|><|audio_start|>"
    )


def build_cfg_null_token_ids(prompt_token_ids: list[int]) -> list[int]:
    """Return the unconditioned twin's token ids for a conditioned prompt.

    Everything between the leading ``<|im_start|>`` and the trailing
    ``<|im_end|><|audio_start|>`` becomes ``<|audio_cfg|>``. Length and the
    audio-start position are preserved, which is what lets the two rows decode
    in lockstep.

    Raises:
        ValueError: If the prompt is too short to carry a caption and lyrics,
            or does not have the expected framing tokens.
    """
    if len(prompt_token_ids) < 4:
        raise ValueError(f"MiniMax Music 3 prompt is too short to build a CFG twin: {len(prompt_token_ids)} tokens")
    expected = (
        SPECIAL_TOKEN_IDS["<|im_start|>"],
        SPECIAL_TOKEN_IDS["<|im_end|>"],
        SPECIAL_TOKEN_IDS["<|audio_start|>"],
    )
    actual = (prompt_token_ids[0], prompt_token_ids[-2], prompt_token_ids[-1])
    if actual != expected:
        raise ValueError(
            "MiniMax Music 3 prompt framing is wrong: expected "
            f"(im_start, im_end, audio_start)={expected}, got {actual}"
        )
    null_ids = list(prompt_token_ids)
    null_ids[1:-2] = [SPECIAL_TOKEN_IDS["<|audio_cfg|>"]] * (len(null_ids) - 3)
    return null_ids


def validate_tokenizer_ids(tokenizer: Any) -> None:
    """Fail fast when a tokenizer is not the supported music tokenizer."""
    for token, expected in SPECIAL_TOKEN_IDS.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != expected:
            raise ValueError(f"MiniMax Music 3 tokenizer mismatch for {token}: expected {expected}, got {token_id}")


__all__ = [
    "AUDIO_CODE_OFFSET",
    "SPECIAL_TOKEN_IDS",
    "build_cfg_null_token_ids",
    "build_prompt",
    "clean_caption",
    "normalize_lyrics",
    "validate_tokenizer_ids",
]
