# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared helpers for normalizing diffusion request prompts."""

from __future__ import annotations

from typing import Any


def _effective_negative(negative: Any) -> str | None:
    """Return the negative prompt as a string, or None if it is None.

    Whitespace-only strings (e.g. ``" "``) are intentionally preserved;
    some models (e.g. Qwen-Image-Edit) pass them to enable CFG.
    """
    if negative is None:
        return None
    return negative if isinstance(negative, str) else str(negative)


def normalize_prompt_entry(raw: Any) -> str | dict[str, Any]:
    """Coerce one prompt entry to ``str`` or a dict with a normalized ``prompt`` field.

    Removes ``negative_prompt`` when it is ``None`` or blank so downstream code can
    distinguish "no negative prompt" from an explicit empty CFG string.
    Other dict keys (e.g. ``multi_modal_data``) are preserved.
    """
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        out = dict(raw)
        out["prompt"] = str(out.get("prompt") or "").strip()
        negative = _effective_negative(out.get("negative_prompt"))
        if negative is None:
            out.pop("negative_prompt", None)
        else:
            out["negative_prompt"] = negative
        return out
    raise TypeError(f"Diffusion prompt must be str or dict, got {type(raw)!r}")


def normalize_omni_diffusion_prompts(prompts: list[Any]) -> list[Any]:
    """Normalize every prompt entry in a diffusion request."""
    return [normalize_prompt_entry(p) for p in prompts]


def has_negative_prompt(prompts: list[Any]) -> bool:
    """Return True if any prompt entry carries a non-empty negative prompt."""
    for entry in prompts:
        if isinstance(entry, dict) and _effective_negative(entry.get("negative_prompt")) is not None:
            return True
    return False


def extract_batch_prompts(
    prompts: list[Any],
) -> tuple[list[str], list[str] | None]:
    """Extract batched text prompts and optional negative prompts for diffusion pipelines.

    Returns:
        A pair ``(prompt, negative_prompt)`` where ``prompt`` is a list of strings.
        ``negative_prompt`` is ``None`` when no entry has a negative prompt; otherwise
        a list aligned with ``prompt`` (``""`` for entries without one).
    """
    if not prompts:
        return [], None

    prompt: list[str] = []
    for entry in prompts:
        if isinstance(entry, str):
            prompt.append(entry)
        elif isinstance(entry, dict):
            prompt.append(str(entry.get("prompt") or ""))
        else:
            raise TypeError(f"Diffusion prompt must be str or dict, got {type(entry)!r}")

    if not has_negative_prompt(prompts):
        return prompt, None

    negative_prompt: list[str] = []
    for entry in prompts:
        if isinstance(entry, dict):
            negative_prompt.append(_effective_negative(entry.get("negative_prompt")) or "")
        else:
            negative_prompt.append("")
    return prompt, negative_prompt
