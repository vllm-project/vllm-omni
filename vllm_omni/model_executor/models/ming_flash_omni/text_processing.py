# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright (c) Ant Group. All rights reserved.
# Adapted from:
# https://github.com/inclusionAI/Ming/tree/e58533db227031990c5a6864dcf5f08fb53ed0d2/front

"""Text segmentation and normalization utilities for Ming TTS."""

from __future__ import annotations

import re
import string

from vllm.logger import init_logger

logger = init_logger(__name__)

# Ported from
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/front/toolkit.py
_TOKENIZE_PATTERN = re.compile(r"(?:[a-zA-Z]\.)+|[a-zA-Z]+(?:['\-][a-zA-Z]+)*|\d+(?:\.\d+)?|[\u4e00-\u9fff]|\s+|\S")


def tokenize_mixed_text(text: str) -> list[str]:
    return re.findall(_TOKENIZE_PATTERN, text)


def tokenize_mixed_text_iterator(text: str):
    for match in _TOKENIZE_PATTERN.finditer(text):
        yield match.group(0)


# Ported from
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/front/text_segment_cut.py
def is_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def get_semantic_length(text: str) -> int:
    """1 CJK char = 1 unit; 1 contiguous English word = 1 unit."""
    chinese_char_count = len(re.findall(r"[\u4e00-\u9fa5]", text))
    english_word_count = len(re.findall(r"[a-zA-Z]+", text))
    return chinese_char_count + english_word_count


def has_valid_content(text: str) -> bool:
    punctuation_and_whitespace = string.punctuation + string.whitespace
    for char in text:
        if char not in punctuation_and_whitespace:
            return True
    return False


def append_text_fragment(
    fragments: list[str],
    new_text: str,
    max_len: int,
    min_tail_length: int,
) -> list[str]:
    new_text = new_text.lstrip("，,:;" + string.whitespace)
    if not has_valid_content(new_text):
        return fragments
    if not fragments:
        fragments.append(new_text)
        return fragments

    last_fragment = fragments[-1]
    last_semantic_len = get_semantic_length(last_fragment)
    new_semantic_len = get_semantic_length(new_text)

    if last_semantic_len + new_semantic_len <= max_len:
        if last_fragment.endswith(("。", "！", "？")) and new_semantic_len < min_tail_length:
            fragments.append(new_text)
        else:
            separator = ""
            if not last_fragment.endswith(" ") and re.match(r"^[a-zA-Z0-9]", new_text):
                separator = " "
            fragments[-1] += separator + new_text
    else:
        fragments.append(new_text)
    return fragments


def split_long_fragment(text_fragment: str, max_len: int) -> list[str]:
    if get_semantic_length(text_fragment) <= max_len:
        return [text_fragment]

    fragments: list[str] = []
    current_fragment = ""
    semantic_units = re.findall(r"([\u4e00-\u9fa5]|[a-zA-Z]+|[^a-zA-Z\u4e00-\u9fa5]+)", text_fragment)
    for unit in semantic_units:
        unit_len = get_semantic_length(unit)
        current_len = get_semantic_length(current_fragment)
        if current_len + unit_len <= max_len:
            current_fragment += unit
        else:
            if current_fragment:
                fragments.append(current_fragment)
            if unit_len > max_len:
                fragments.append(unit)
                current_fragment = ""
            else:
                current_fragment = unit
    if current_fragment:
        fragments.append(current_fragment)
    return fragments


_DOT_PLACEHOLDER = "##DOT##"
# default soft cap on fragment length in semantic units
_DEFAULT_MAX_SEMANTIC_LENGTH: int = 50
# default tail length controls when a short trailing fragment is
# merged with the previous one to avoid leaving an awkward stub.
_DEFAULT_MIN_TAIL_LENGTH: int = 5


def cut_text_by_semantic_length(
    text: str,
    max_semantic_length: int = _DEFAULT_MAX_SEMANTIC_LENGTH,
    min_tail_length: int = _DEFAULT_MIN_TAIL_LENGTH,
) -> list[str]:
    """Segment text into fragments respecting semantic length limits.

    Ported from upstream Ming's `front/text_segment_cut.py`.
    Position tracking is omitted (not needed for non-streaming VAE decode).
    """
    if not has_valid_content(text):
        return []

    processed = re.sub(r"(\d)\.(\d)", r"\1" + _DOT_PLACEHOLDER + r"\2", text)
    for _ in range(3):
        processed = re.sub(r"([A-Z])\.([A-Z])", r"\1" + _DOT_PLACEHOLDER + r"\2", processed)
    processed = processed.replace("\n", " ").replace("。，", "。")

    if get_semantic_length(processed) <= max_semantic_length:
        return [processed.replace(_DOT_PLACEHOLDER, ".")]

    normalized = processed.replace(".", "。").replace("!", "！").replace("?", "？").replace(",", "，")

    # Phase 1: split into sentences on 。！？
    sentences: list[str] = []
    current: list[str] = []
    for char in normalized:
        current.append(char)
        if char in "。！？":
            s = "".join(current).strip()
            if s:
                sentences.append(s)
            current = []
    if current:
        s = "".join(current).strip()
        if s:
            if not s.endswith(("。", "！", "？")):
                s += "。"
            sentences.append(s)

    # Phase 2: merge whole sentences; only clause-split oversized ones.
    # This ensures split points land on sentence boundaries (。！？)
    # rather than mid-sentence commas.
    result_fragments: list[str] = []
    for sentence in sentences:
        sent_len = get_semantic_length(sentence)

        if sent_len > max_semantic_length:
            # Oversized sentence: fall back to clause-level splitting
            clauses: list[str] = []
            clause_buf: list[str] = []
            for char in sentence:
                clause_buf.append(char)
                if char in "，;；":
                    cl = "".join(clause_buf).strip()
                    if cl and has_valid_content(cl):
                        clauses.append(cl)
                    elif cl and clauses:
                        clauses[-1] += cl
                    clause_buf = []
            if clause_buf:
                cl = "".join(clause_buf).strip()
                if cl and has_valid_content(cl):
                    clauses.append(cl)
                elif cl and clauses:
                    clauses[-1] += cl

            i = 0
            while i < len(clauses):
                clause = clauses[i]
                clause_len = get_semantic_length(clause)

                if clause_len < min_tail_length and i + 1 < len(clauses):
                    combined = clause + clauses[i + 1]
                    if get_semantic_length(combined) <= max_semantic_length:
                        result_fragments = append_text_fragment(
                            result_fragments, combined, max_semantic_length, min_tail_length
                        )
                        i += 2
                        continue

                if clause_len > max_semantic_length:
                    for frag in split_long_fragment(clause, max_semantic_length):
                        result_fragments = append_text_fragment(
                            result_fragments, frag, max_semantic_length, min_tail_length
                        )
                else:
                    result_fragments = append_text_fragment(
                        result_fragments, clause, max_semantic_length, min_tail_length
                    )
                i += 1
        else:
            # Normal sentence: merge at sentence level
            if not result_fragments:
                result_fragments.append(sentence)
            else:
                last_len = get_semantic_length(result_fragments[-1])
                if last_len + sent_len <= max_semantic_length:
                    result_fragments[-1] += sentence
                else:
                    result_fragments.append(sentence)

    return [f.replace(_DOT_PLACEHOLDER, ".") for f in result_fragments]


# Streaming sentence boundary detection

_RE_CJK = re.compile(r"[\u4e00-\u9fff]")
_RE_DIGIT_LAST = re.compile(r"[0-9]")


# Left for reference for now
# Alternative to `cut_text_by_semantic_length`
def detect_sentence_boundaries(
    text: str,
    max_length: int = 50,
) -> list[str]:
    """Accumulate tokens and flush at sentence boundaries.

    Ported from the streaming sentence detection loop from the Ming repo
    TTS branch, but operates on the full text since we have it available upfront.
    """
    sentences: list[str] = []
    streaming_text: list[str] = []
    count = 0

    for ele in tokenize_mixed_text_iterator(text):
        if len(ele) == 0:
            continue

        should_process = False
        min_tokens = 12 if count == 0 else 8

        if ele[-1] in "！？。，!?":
            if len(streaming_text) >= min_tokens:
                should_process = True
            streaming_text.append(ele)

        elif ele[-1] == ".":
            if (
                len(streaming_text) >= min_tokens
                and streaming_text
                and not _RE_DIGIT_LAST.search(streaming_text[-1][-1])
            ):
                should_process = True
            streaming_text.append(ele)

        elif ele[-1] == "\n":
            if streaming_text:
                joined = "".join(streaming_text)
                if _RE_CJK.search(joined):
                    if _RE_CJK.search(streaming_text[-1][-1]):
                        ele = "，"
                        streaming_text.append(ele)
                else:
                    if len(ele) > 1 and re.search(r"[a-zA-Z]", ele[-2]):
                        ele = ele[:-1] + "."
                    else:
                        ele = ele[:-1]
                    streaming_text.append(ele)

            if len(streaming_text) >= min_tokens:
                should_process = True
        else:
            streaming_text.append(ele)
            continue

        if should_process:
            joined = "".join(streaming_text)
            fragments = cut_text_by_semantic_length(joined, max_length)
            sentences.extend(fragments)
            streaming_text = []
            count += 1

    # Flush remaining
    if streaming_text and re.search(r"[a-zA-Z\u4e00-\u9fff1-9]", "".join(streaming_text)):
        joined = "".join(streaming_text)
        fragments = cut_text_by_semantic_length(joined, max_length)
        sentences.extend(fragments)

    return sentences


# number normalization for English. Ported from
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/front/number_en.py


_inflect_engine = None


def _get_inflect():
    global _inflect_engine
    if _inflect_engine is not None:
        return _inflect_engine
    try:
        import inflect

        _inflect_engine = inflect.engine()
    except ImportError:
        logger.warning(
            "Package 'inflect' not installed - English number normalization "
            "will be skipped.  Install with: pip install inflect"
        )
        _inflect_engine = None
    return _inflect_engine


_comma_number_re = re.compile(r"([0-9][0-9,]+[0-9])")
_percent_number_re = re.compile(r"(-?[0-9.,]*[0-9]+)%")
_pounds_re = re.compile(r"£(-?[0-9,]*[0-9]+(?:\.[0-9]+)?)")
_dollars_re = re.compile(r"\$(-?[0-9.,]*[0-9]+(?:\.[0-9]+)?)")
_fraction_re = re.compile(r"([0-9]+)\/([0-9]+)")
_ordinal_re = re.compile(r"\b[0-9]+(st|[nr]d|th)\b")
_number_re = re.compile(r"\b-?[0-9]+(?:\.[0-9]+)?\b")
_unit_re = re.compile(
    r"\b(-?\d+(?:\.\d+)?)\s*"
    r"(ms|s|Hz|kHz|MHz|GHz|kb|mb|gb|tb|KB|MB|GB|TB|bps|kbps|Mbps|Gbps|cm|km|kg|V|A|W|°C|°F)\b",
    re.IGNORECASE,
)
_version_re = re.compile(r"\b([a-zA-Z]+)([-]?)([0-9]+(?:\.[0-9]+)?)\b")
_whitespace_re = re.compile(r"\s+")

_unit_mapping = {
    "ms": "milliseconds",
    "s": "seconds",
    "hz": "hertz",
    "khz": "kilohertz",
    "mhz": "megahertz",
    "ghz": "gigahertz",
    "kb": "kilobytes",
    "mb": "megabytes",
    "gb": "gigabytes",
    "tb": "terabytes",
    "kbps": "kilobits per second",
    "mbps": "megabits per second",
    "gbps": "gigabits per second",
    "bps": "bits per second",
    "cm": "centimeters",
    "km": "kilometers",
    "kg": "kilograms",
    "v": "volts",
    "a": "amperes",
    "w": "watts",
    "°c": "degrees celsius",
    "°f": "degrees fahrenheit",
}


def _num_to_words(n: int) -> str:
    p = _get_inflect()
    if p is None:
        return str(n)
    return p.number_to_words(n, andword="")


def _expand_decimal(num_str: str) -> str:
    """Expand a decimal number string like '3.14' -> 'three point one four'."""
    is_negative = num_str.startswith("-")
    clean = num_str.lstrip("-") or "0"

    if "." in clean:
        parts = clean.split(".", 1)
        integer_part = parts[0] or "0"
        decimal_part = parts[1]
        if not integer_part.isdigit() or not decimal_part.isdigit():
            return num_str
        int_word = _num_to_words(int(integer_part)) if integer_part != "0" else "zero"
        dec_words = " ".join(_num_to_words(int(d)) for d in decimal_part if d.isdigit())
        word = f"{int_word} point {dec_words}"
    else:
        if not clean.isdigit():
            return num_str
        word = _num_to_words(int(clean))

    if is_negative:
        word = f"minus {word}"
    return word


def _remove_commas(m: re.Match) -> str:
    return m.group(1).replace(",", "")


_NUM_PARSE_EXC: tuple[type[BaseException], ...] = (ValueError, TypeError)


def _expand_unit(m: re.Match) -> str:
    num_str, unit = m.group(1), m.group(2).lower()
    unit_word = _unit_mapping.get(unit, unit)
    try:
        return f" {_expand_decimal(num_str)} {unit_word} "
    except _NUM_PARSE_EXC:
        return f" {num_str} {unit} "


def _expand_percent(m: re.Match) -> str:
    try:
        return f" {_expand_decimal(m.group(1))} percent "
    except _NUM_PARSE_EXC:
        return f" {m.group(1)} percent "


def _expand_dollars(m: re.Match) -> str:
    raw = m.group(1)
    clean = raw.lstrip("-") or "0"
    try:
        word = _expand_decimal(raw)
        value = float(clean)
        unit = "dollar" if abs(value) == 1.0 else "dollars"
        return f" {word} {unit} "
    except _NUM_PARSE_EXC:
        return f" {clean} dollars "


def _expand_pounds(m: re.Match) -> str:
    raw = m.group(1)
    clean = raw.lstrip("-") or "0"
    try:
        word = _expand_decimal(raw)
        value = float(clean)
        unit = "pound" if abs(value) == 1.0 else "pounds"
        return f" {word} {unit} "
    except _NUM_PARSE_EXC:
        return f" {clean} pounds "


def _expand_fraction(m: re.Match) -> str:
    p = _get_inflect()
    if p is None:
        return m.group(0)
    try:
        num, den = int(m.group(1)), int(m.group(2))
        if num == 1 and den == 2:
            return " one half "
        if num == 1 and den == 4:
            return " one quarter "
        if den == 2:
            plural = " half" if num == 1 else " halves"
            return f" {p.number_to_words(num)}{plural} "
        if den == 4:
            plural = " quarter" if num == 1 else " quarters"
            return f" {p.number_to_words(num)}{plural} "
        ordinal = p.ordinal(p.number_to_words(den))
        return f" {p.number_to_words(num)} {ordinal} "
    except _NUM_PARSE_EXC:
        return f" {m.group(1)} over {m.group(2)} "


def _expand_ordinal(m: re.Match) -> str:
    try:
        num = int(re.sub(r"(st|and|rd|th)", "", m.group(0)))
        return f" {_num_to_words(num)} "
    except _NUM_PARSE_EXC:
        return m.group(0)


def _expand_number(m: re.Match) -> str:
    try:
        return f" {_expand_decimal(m.group(0))} "
    except _NUM_PARSE_EXC:
        return f" {m.group(0)} "


def _expand_version(m: re.Match) -> str:
    prefix, _, num_str = m.group(1), m.group(2), m.group(3)
    try:
        word = _expand_decimal(num_str)
    except _NUM_PARSE_EXC:
        return m.group(0)
    return f"{prefix} {word}"


def normalize_numbers(text: str) -> str:
    """Expand English numbers, currencies, units, etc. to words.

    Returns text unchanged if `inflect` package is not installed.
    """
    if _get_inflect() is None:
        return text
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_unit_re, _expand_unit, text)
    text = re.sub(_pounds_re, _expand_pounds, text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_fraction_re, _expand_fraction, text)
    text = re.sub(_percent_number_re, _expand_percent, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_version_re, _expand_version, text)
    text = re.sub(_number_re, _expand_number, text)
    text = re.sub(_whitespace_re, " ", text)
    return text.strip()


# Top-level API
def segment_and_normalize(
    text: str,
    max_length: int = _DEFAULT_MAX_SEMANTIC_LENGTH,
) -> list[str]:
    """Segment text into fragments and expand English numbers for Ming TTS.

    This function cuts text by semantic length directly rather than following
    the streaming algorithm to detect sentence boundaries in the upstream
    Ming repo (which is more aggressively splitting at commas). It produces
    fewer and larger segments at natural sentence boundaries.
    """
    if not text or not text.strip():
        return []

    segments = cut_text_by_semantic_length(text.strip(), max_length)

    normalized: list[str] = []
    for seg in segments:
        if not is_chinese(seg):
            seg = normalize_numbers(seg)
        if seg and seg[0] == "，":
            seg = seg[1:]
        seg = seg.strip()
        if seg:
            normalized.append(seg)

    return normalized if normalized else [text.strip()]
