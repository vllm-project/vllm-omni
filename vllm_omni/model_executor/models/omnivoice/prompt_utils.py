# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import difflib

import regex as re

from vllm_omni.model_executor.models.omnivoice.instruct_constants import (
    _INSTRUCT_ALL_VALID,
    _INSTRUCT_EN_TO_ZH,
    _INSTRUCT_MUTUALLY_EXCLUSIVE,
    _INSTRUCT_VALID_EN,
    _INSTRUCT_VALID_ZH,
    _INSTRUCT_ZH_TO_EN,
    _ZH_RE,
)


def validate_instruction(instruct):
    """
    Based on original instruct validation at https://github.com/k2-fsa/OmniVoice/blob/38e992bc60f85548faeb77e8fa70158ba71deb30/omnivoice/models/omnivoice.py#L1492
    """

    if instruct is None:
        return None

    instruct_str = instruct.strip()
    if not instruct_str:
        return None

    # Validate each item
    normalised = normalise(instruct)
    unknown = []

    for raw in raw_items(instruct):
        r = raw.strip().lower()
        if r not in normalised:
            sug = difflib.get_close_matches(r, _INSTRUCT_ALL_VALID, n=1, cutoff=0.6)
            unknown.append((raw, r, sug[0] if sug else None))

    if unknown:
        lines = []
        for raw, n, sug in unknown:
            if sug:
                lines.append(f"  '{raw}' -> '{n}' (unsupported; did you mean '{sug}'?)")
            else:
                lines.append(f"  '{raw}' -> '{n}' (unsupported)")
        warning = (
            f"Unsupported instruct items found in '{instruct_str}':\n"
            + "\n".join(lines)
            + "\nValid English items: "
            + ", ".join(sorted(_INSTRUCT_VALID_EN))
            + "\nValid Chinese items: "
            + "，".join(sorted(_INSTRUCT_VALID_ZH))
            + "\nTip: Use only English or only Chinese instructs. "
            "English instructs should use comma + space (e.g. "
            "'male, indian accent'),\nChinese instructs should use full-width "
            "comma (e.g. '男，河南话')."
        )
        return warning

    # --- Language consistency: dialect forces Chinese, accent forces English ---
    if has_dialect(normalised) and has_accent(normalised):
        warning = (
            "Cannot mix Chinese dialect and English accent in a single instruct"
            + "Dialects are for Chinese speech, accents for English speech."
        )
        return warning

    # --- Unify to single language ---
    normalised = unify_language(normalised)

    # --- Category conflict check ---
    conflicts = []
    for cat in _INSTRUCT_MUTUALLY_EXCLUSIVE:
        hits = [n for n in normalised if n in cat]
        if len(hits) > 1:
            conflicts.append(hits)
    if conflicts:
        parts = []
        for group in conflicts:
            parts.append(" vs ".join(f"'{x}'" for x in group))
        warning = (
            "Conflicting instruct items within the same category: "
            + "; ".join(parts)
            + ". Each category (gender, age, pitch, style, accent, dialect) allows at most one item."
        )
        return warning

    return None


def raw_items(instruct_str: str) -> list:
    # Split on both half-width and full-width commas
    raw_items = re.split(r"\s*[,，]\s*", instruct_str)
    raw_items = [x for x in raw_items if x]
    return raw_items


def normalise(instruct_str) -> list:
    normalised = []

    for raw in raw_items(instruct_str):
        n = raw.strip().lower()
        if n in _INSTRUCT_ALL_VALID:
            normalised.append(n)
    return normalised


def prepare_instruct(instruct_str) -> str:
    normalised = normalise(instruct_str)
    has_zh = any(any("\u4e00" <= c <= "\u9fff" for c in n) for n in normalised)
    separator = "，" if has_zh else ", "
    return separator.join(normalised)


def unify_language(normalised_instruct: list) -> list:
    if use_zh(normalised_instruct):
        return [_INSTRUCT_EN_TO_ZH.get(n, n) for n in normalised_instruct]
    else:
        return [_INSTRUCT_ZH_TO_EN.get(n, n) for n in normalised_instruct]


def use_zh(normalised_instruct: list) -> bool:
    if has_dialect(normalised_instruct):
        return True
    elif has_accent(normalised_instruct):
        return False
    else:
        return any(_ZH_RE.search(instruct) for instruct in normalised_instruct)


def has_dialect(normalised_instruct: list) -> bool:
    return any(n.endswith("话") for n in normalised_instruct)


def has_accent(normalised_instruct: list) -> bool:
    return any(" accent" in n for n in normalised_instruct)
