# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Text and tokenization utilities for F5-TTS inference.

Ported from F5-TTS:
  - Text preprocessing (process_text)
  - Legacy `pinyin` tokenizer backed by `vocab.txt`
  - Duration estimation
  - Sequence quantization and padding
"""

from __future__ import annotations

import logging
import math
from typing import Protocol, runtime_checkable

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Language info (subset needed for process_text)
# ---------------------------------------------------------------------------

# Which languages use spaces between words.
_LANGUAGES_WITH_WORD_SPACES: set[str] = {
    "en",
    "fr",
    "de",
    "es",
    "pt",
    "it",
    "nl",
    "pl",
    "ru",
    "uk",
    "cs",
    "sv",
    "da",
    "no",
    "fi",
    "ro",
    "hu",
    "el",
    "bg",
    "hr",
    "sk",
    "sl",
    "lt",
    "lv",
    "et",
    "ga",
    "cy",
    "eu",
    "ca",
    "gl",
    "af",
    "sw",
    "id",
    "ms",
    "tl",
    "tr",
    "vi",
    "hi",
    "bn",
    "ta",
    "te",
    "kn",
    "ml",
    "mr",
    "gu",
    "pa",
    "ur",
    "fa",
    "ar",
    "he",
    "th",
}


def has_space_between_words(lang: str) -> bool:
    """Check whether *lang* uses spaces between words.

    Falls back to ``True`` for unknown languages (safe default for most).
    """
    if lang in ("zh", "ja", "ko"):
        return False
    return True


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------


def process_text(
    cond_text: str,
    target_text: str,
    lang: str,
) -> tuple[str, str]:
    """Preprocess conditioning and target text for TTS.

    Ported from ``f5tts.eval.utils.process_text``.

    - Strips whitespace from both texts.
    - For languages that use inter-word spaces, prepends a space to
      *target_text* so the model sees a word boundary between the
      conditioning and target regions.

    Args:
        cond_text: Text corresponding to the conditioning audio.
        target_text: Text to be synthesized.
        lang: ISO language code (e.g. ``"en"``).

    Returns:
        ``(cond_text, target_text)`` — preprocessed.
    """
    cond_text = cond_text.strip()
    target_text = target_text.strip()

    if cond_text:
        # Match F5-TTS preprocessing: ensure the conditioning transcript
        # ends with sentence punctuation followed by a word boundary.
        if not cond_text.endswith(". ") and not cond_text.endswith("。"):
            if cond_text.endswith("."):
                cond_text += " "
            else:
                cond_text += ". "
        elif cond_text.endswith(".") and not cond_text.endswith(". "):
            cond_text += " "

        # F5-TTS inference keeps a trailing space after ASCII endings so the
        # target text starts at a clear boundary when concatenated.
        if len(cond_text[-1].encode("utf-8")) == 1:
            cond_text += " "
    elif has_space_between_words(lang):
        target_text = " " + target_text

    return cond_text, target_text


def _is_cjk_char(ch: str) -> bool:
    # chinese, japanese, korean
    return "\u3100" <= ch <= "\u9fff"


def _convert_char_to_pinyin_tokens(text: str) -> list[str]:
    """Convert mixed-language text into the F5-TTS pinyin token stream.

    This mirrors `convert_char_to_pinyin` behavior closely enough
    for inference while avoiding the hard dependency on `tts_tokenizer`.
    """
    custom_trans = str.maketrans({";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"})
    text = text.translate(custom_trans)

    lazy_pinyin = None
    pinyin_style = None
    if any(_is_cjk_char(ch) for ch in text):
        try:
            from pypinyin import Style  # type: ignore[import-untyped]
            from pypinyin import lazy_pinyin as _lazy_pinyin
        except ImportError as exc:
            raise ImportError(
                "pypinyin is required for tokenizer_type='pinyin' when the text contains Chinese characters."
            ) from exc
        lazy_pinyin = _lazy_pinyin
        pinyin_style = Style.TONE3

    try:
        import rjieba  # type: ignore[import-untyped]
    except ImportError:
        rjieba = None

    segments = rjieba.cut(text) if rjieba is not None else [text]

    tokens: list[str] = []
    for seg in segments:
        seg_byte_len = len(seg.encode("utf-8"))
        if seg_byte_len == len(seg):
            if tokens and seg_byte_len > 1 and tokens[-1] not in " :'\"":
                tokens.append(" ")
            tokens.extend(seg)
            continue

        if lazy_pinyin is not None and pinyin_style is not None and seg_byte_len == 3 * len(seg):
            seg_pinyin = lazy_pinyin(seg, style=pinyin_style, tone_sandhi=True)
            for idx, ch in enumerate(seg):
                if _is_cjk_char(ch):
                    tokens.append(" ")
                tokens.append(seg_pinyin[idx])
            continue

        for ch in seg:
            if ord(ch) < 256:
                tokens.extend(ch)
            elif _is_cjk_char(ch):
                assert lazy_pinyin is not None and pinyin_style is not None, (
                    f"CJK char U+{ord(ch):04X} found but pypinyin not loaded"
                )
                tokens.append(" ")
                tokens.extend(lazy_pinyin(ch, style=pinyin_style, tone_sandhi=True))
            else:
                tokens.append(ch)
    return tokens


# ---------------------------------------------------------------------------
# Tokenizer protocol & implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for text tokenizers (matches ``f5tts.model.utils.Tokenizer``)."""

    def encode(self, text: str, lang: str, *, deterministic: bool = True) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...

    @property
    def vocab_size(self) -> int: ...


class LegacyTokenizer:
    """Legacy character and vocab-based tokenizer.

    F5-TTS uses the legacy `pinyin` vocabulary format backed by a
    line-based `vocab.txt`.
    """

    def __init__(
        self,
        tokenizer_name: str,
        vocab_path: str,
        wrap_bos_eos: bool = True,
    ) -> None:
        if tokenizer_name != "pinyin":
            raise ValueError(f"Unsupported F5 tokenizer_type={tokenizer_name!r}; expected 'pinyin'.")
        self._vocab_char_map = _load_vocab(tokenizer=tokenizer_name, tokenizer_path=vocab_path)
        self._inv_vocab_char_map = {v: k for k, v in self._vocab_char_map.items()}
        self._wrap_bos_eos = wrap_bos_eos

    def encode(self, text: str, lang: str, *, deterministic: bool = True) -> list[int]:
        tokens = _convert_char_to_pinyin_tokens(text)
        ids = [self._vocab_char_map.get(tok, 0) for tok in tokens]

        lang_id = self._vocab_char_map.get(f"<lang_{lang}>")
        if lang_id is not None:
            ids = [lang_id, *ids]

        if self._wrap_bos_eos:
            bos_id = self._vocab_char_map.get("<bos>")
            eos_id = self._vocab_char_map.get("<eos>")
            if bos_id is not None and eos_id is not None:
                ids = [bos_id, *ids, eos_id]

        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self._inv_vocab_char_map.get(i, "") for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_char_map)


def _load_vocab(*, tokenizer: str, tokenizer_path: str) -> dict[str, int]:
    """Load a character-level vocabulary from a text file.

    Each line is either ``token`` (ID = line number) or ``token\\tID``.
    """
    vocab: dict[str, int] = {}
    with open(tokenizer_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.rstrip("\r\n")
            parts = line.split("\t")
            if len(parts) == 2:
                token, token_id = parts[0], int(parts[1])
            else:
                token, token_id = line, idx
            vocab[token] = token_id
    return vocab


def load_tokenizer(tokenizer_type: str, path: str) -> Tokenizer:
    """Load the F5 legacy `pinyin` tokenizer."""
    return LegacyTokenizer(tokenizer_type, path)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Duration estimation
# ---------------------------------------------------------------------------


def estimate_duration(
    cond_mel_len: int,
    cond_text: str,
    target_text: str,
    speed: float = 1.0,
) -> int:
    """Estimate the total mel spectrogram length for generation.

    Ported from ``f5tts.infer.model.estimate_duraton``.

    Uses the ratio of conditioning audio frames per byte of conditioning
    text, projected onto the target text length.

    Args:
        cond_mel_len: Number of mel frames in the conditioning audio.
        cond_text: Conditioning text (already preprocessed).
        target_text: Target text to generate (already preprocessed).
        speed: Speed factor (>1 = faster/shorter, <1 = slower/longer).

    Returns:
        Estimated total mel frame count (conditioning + generated).
    """
    ref_text_len = len(cond_text.encode("utf-8"))
    gen_text_len = len(target_text.encode("utf-8"))

    local_speed = speed
    if gen_text_len < 10:
        local_speed = 0.3

    if ref_text_len == 0:
        logger.warning("cond_text is empty, using default duration ratio")
        ref_text_len = 1
    total_mel_len = cond_mel_len + int(cond_mel_len / ref_text_len * gen_text_len / local_speed)
    return total_mel_len


# ---------------------------------------------------------------------------
# Sequence quantization and padding
# ---------------------------------------------------------------------------


def quantize(x: int, mul: int = 64) -> int:
    """Round *x* up to the nearest multiple of *mul*.

    Ported from ``f5tts.eval.eval_infer.quantize``.
    Used to pad sequence lengths for efficient GPU execution.
    """
    return math.ceil(x / mul) * mul


def pad_and_batch_items(
    items: list[tuple[torch.Tensor, list[int], int]],
    *,
    pad_multiple: int = 64,
    text_pad_value: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, int, list[int], list[int], list[int]]:
    """Pad multiple (cond_mel, text_token_ids, total_mel_len) items to a shared
    quantized sequence length and stack into batch tensors.

    Args:
        items: List of (cond_mel [T_cond, D], text_token_ids, total_mel_len).
        pad_multiple: Pad to nearest multiple of this value.
        text_pad_value: Padding value for text tokens (default -1).

    Returns:
        tuple: A tuple containing (cond_audio, cond_text, seq_len,
            effective_lens, cond_mel_lens, total_mel_lens) where cond_audio
            has shape [B, seq_len, D] and cond_text has shape [B, seq_len].
    """
    if not items:
        raise ValueError("items cannot be empty")

    batch_size = len(items)
    device = items[0][0].device
    dtype = items[0][0].dtype
    mel_dim = items[0][0].shape[1]

    cond_mel_lens = [item[0].shape[0] for item in items]
    total_mel_lens = [item[2] for item in items]

    max_total_mel_len = max(total_mel_lens)
    seq_len = quantize(max_total_mel_len, pad_multiple)

    # Effective length: for B=1, match upstream single-item behavior where
    # effective length extends to quantized seq_len; for B>1, use total_mel_lens.
    if batch_size == 1:
        effective_lens = [seq_len]
    else:
        effective_lens = [min(t, seq_len) for t in total_mel_lens]

    cond_audio = torch.zeros(batch_size, seq_len, mel_dim, dtype=dtype, device=device)
    cond_text = torch.full((batch_size, seq_len), fill_value=text_pad_value, dtype=torch.long, device=device)

    for i, (cond_mel, text_ids, _) in enumerate(items):
        c_len = min(cond_mel.shape[0], seq_len)
        cond_audio[i, :c_len] = cond_mel[:c_len]

        t_len = min(len(text_ids), seq_len)
        if t_len > 0:
            cond_text[i, :t_len] = torch.tensor(text_ids[:t_len], dtype=torch.long, device=device)

    return cond_audio, cond_text, seq_len, effective_lens, cond_mel_lens, total_mel_lens


def pad_and_batch(
    cond_mel: torch.Tensor,
    text_token_ids: list[int],
    total_mel_len: int,
    *,
    pad_multiple: int = 64,
    text_pad_value: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Pad mel and text to a quantized sequence length and add batch dim.

    Ported from the padding logic in ``f5tts.infer.model.generate``.

    Args:
        cond_mel: Conditioning mel ``[T_cond, D]``.
        text_token_ids: Token ID list (len <= total_mel_len).
        total_mel_len: Target total mel length (before quantization).
        pad_multiple: Pad to nearest multiple of this value.
        text_pad_value: Padding value for text tokens (default -1,
            which ``TextEmbedding`` shifts to 0 = filler).

    Returns:
        ``(cond_audio, cond_text, seq_len, cond_mel_len)``
        where ``cond_audio`` is ``[1, seq_len, D]`` and
        ``cond_text`` is ``[1, seq_len]``.
    """
    cond_audio, cond_text, seq_len, _, cond_mel_lens, _ = pad_and_batch_items(
        [(cond_mel, text_token_ids, total_mel_len)],
        pad_multiple=pad_multiple,
        text_pad_value=text_pad_value,
    )
    return (
        cond_audio,
        cond_text,
        seq_len,
        cond_mel_lens[0],
    )
