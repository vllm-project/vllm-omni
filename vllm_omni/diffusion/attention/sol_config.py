# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SOL layer selectors, resolved once when attention is built."""

import math

import regex


def parse_sol_block_ranges(spec: str) -> list[tuple[int, int]]:
    """Accept inclusive ranges and negative indices, e.g. ``0-2,-1``."""
    if not isinstance(spec, str):
        raise TypeError("sol_attn block selectors must be strings, e.g. '0-2,-1'.")
    ranges = []
    for part in "".join(spec.split()).split(","):
        if not part:
            continue
        match = regex.fullmatch(r"(-?\d+)(?:-(-?\d+))?", part)
        if match is None:
            raise ValueError(f"Invalid sol_attn block selector {part!r}; use indices or ranges like '0-2,-1'.")
        first = int(match.group(1))
        ranges.append((first, first if match.group(2) is None else int(match.group(2))))
    return ranges


def sol_block_selected(spec: str, layer_index: int, layer_count: int | None) -> bool:
    """Resolve against the model's block stack without expanding large ranges."""
    selected = False
    for first, last in parse_sol_block_ranges(spec):
        if first < 0 or last < 0:
            if layer_count is None:
                raise ValueError("Negative sol_attn block indices require model-owned layer_count metadata.")
            first = layer_count + first if first < 0 else first
            last = layer_count + last if last < 0 else last
        selected |= min(first, last) <= layer_index <= max(first, last)
    return selected


def parse_sol_tau_profile(spec: str) -> list[tuple[str, float]]:
    """Parse ordered ``blocks=tau`` rules; later matching entries win.

    Semicolons or newlines separate entries, and ``#`` starts a comment.
    Unlisted blocks retain the base tau.
    """
    if not isinstance(spec, str):
        raise TypeError("sol_attn.tau_profile must be a string, e.g. '0-30=2.0;39-42=0.9'.")
    rules = []
    for entry in regex.split(r"[;\n]", spec):
        entry = entry.split("#", 1)[0].strip()
        if not entry:
            continue
        selector, separator, raw_tau = entry.partition("=")
        if not separator or not parse_sol_block_ranges(selector):
            raise ValueError(f"Invalid sol_attn.tau_profile entry {entry!r}; use 'blocks=tau'.")
        try:
            tau = float(raw_tau)
        except ValueError as error:
            raise ValueError(f"Invalid sol_attn.tau_profile tau in {entry!r}.") from error
        if not math.isfinite(tau) or tau < 0:
            raise ValueError("sol_attn.tau_profile values must be finite and non-negative.")
        rules.append((selector, tau))
    return rules
