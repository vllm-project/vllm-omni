# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from SGLang (https://github.com/sgl-project/sglang/pull/32848).

from __future__ import annotations

import json

import regex as re
from PIL import Image

from vllm_omni.diffusion.models.lingbot_video.rewriter import (
    RewriterBackend,
    parse_caption,
)
from vllm_omni.diffusion.models.lingbot_video.rewriter_prompts import NEGATIVE_PRUNE

_ROOT = "universal_negative"


def _hint_pattern(*hints: str) -> re.Pattern:
    alternatives = "|".join(re.escape(hint) for hint in hints)
    return re.compile(rf"(?<![a-z0-9])(?:{alternatives})(?![a-z0-9])")


_BLOCK_HINTS = {
    "physical_plausibility": _hint_pattern(
        "fantasy",
        "surreal",
        "dreamlike",
        "dream-like",
        "magic",
        "magical",
        "supernatural",
        "physics-bending",
        "physics bending",
        "impossible physics",
        "anti-gravity",
        "antigravity",
        "zero gravity",
        "zero-gravity",
        "weightless",
        "weightlessness",
        "floating in space",
        "outer space",
        "astronaut",
    ),
    "artistic_style": _hint_pattern(
        "painting",
        "illustration",
        "cartoon",
        "drawing",
        "sketch",
        "cgi",
        "3d render",
        "3d-render",
        "digital art",
        "anime",
        "stylized animation",
        "claymation",
        "stop motion",
        "stop-motion",
    ),
}

_FORCED_DELETIONS = (
    (
        _hint_pattern(
            "dark",
            "dim",
            "dimly",
            "low light",
            "low-light",
            "night",
            "nighttime",
            "moody",
            "gloomy",
            "ominous",
            "deep shadow",
            "deep shadows",
            "dark shadows",
        ),
        ("underexposed", "subject hidden in darkness", "crushed blacks"),
    ),
    (
        _hint_pattern(
            "motion blur",
            "motion-blur",
            "blurred background",
            "blurred landscape",
            "blurred scenery",
            "blurred surroundings",
            "speed blur",
            "long exposure",
        ),
        ("motion blur",),
    ),
)


def categorized_negative(negative_prompt: str) -> dict | None:
    """Return the shipped category shape, or ``None`` for free text."""

    parsed = parse_caption(negative_prompt)
    if not isinstance(parsed, dict):
        return None
    categories = parsed.get(_ROOT)
    if not isinstance(categories, dict):
        return None
    for terms in categories.values():
        if not isinstance(terms, list) or not all(isinstance(term, str) for term in terms):
            return None
    return {_ROOT: categories}


def build_prune_prompt(caption: str, mode: str, default: dict) -> str:
    return (
        f"{NEGATIVE_PRUNE}\n\n## MODE: {mode}\n\n"
        f"## INTENDED CONTENT (structured caption):\n```json\n{caption}\n```\n\n"
        "## DEFAULT NEGATIVE (delete the contradicting terms, keep the rest):\n"
        f"```json\n{json.dumps(default, ensure_ascii=False)}\n```\n\n"
        "Output ONLY the edited negative JSON now."
    )


def prune_negative(default: dict, pruned: dict | None, caption: str) -> dict:
    """Keep the default order and allow the model to delete terms only."""

    kept = pruned.get(_ROOT) if isinstance(pruned, dict) else None
    lowered = caption.lower()
    out = {}
    for category, terms in default[_ROOT].items():
        survivors = kept.get(category) if isinstance(kept, dict) else None
        if not isinstance(survivors, list):
            out[category] = list(terms)
            continue
        survivor_set = set(survivors)
        out[category] = [term for term in terms if term in survivor_set]
        hints = _BLOCK_HINTS.get(category)
        if not out[category] and hints is not None and not hints.search(lowered):
            out[category] = list(terms)
    for hints, deletions in _FORCED_DELETIONS:
        if not hints.search(lowered):
            continue
        for category, terms in out.items():
            out[category] = [term for term in terms if term not in deletions]
    return {_ROOT: out}


def customize_negative_prompt(
    *,
    backend: RewriterBackend,
    caption: str,
    mode: str,
    negative_prompt: str,
    image: Image.Image | None,
) -> str:
    default = categorized_negative(negative_prompt)
    if default is None:
        return negative_prompt
    raw = backend.generate(
        build_prune_prompt(caption, mode, default),
        image,
        use_lora=False,
    )
    negative = prune_negative(default, parse_caption(raw), caption)
    return json.dumps(negative, ensure_ascii=False, separators=(",", ":"))
