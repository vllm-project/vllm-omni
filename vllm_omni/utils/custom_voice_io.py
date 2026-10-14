"""Shared helpers for custom voice profile tools."""

from __future__ import annotations

import re


def safe_voice_stem(name: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name.strip())
    if not stem or stem in (".", ".."):
        raise ValueError(f"Invalid voice name: {name!r}")
    return stem[:200]
