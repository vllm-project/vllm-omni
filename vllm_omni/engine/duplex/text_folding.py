# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fit labeled transcript excerpts without publishing partial memory.

Callers own source validation and commit. Rendering and exact capacity checks
remain model-specific; offsets always address the original transcript.
"""

from collections.abc import Callable, Sequence
from typing import TypeVar

Prompt = TypeVar("Prompt")
TextSpan = tuple[int, int, int, str]


def fit_text_excerpt(
    transcript: str,
    spans: Sequence[TextSpan],
    *,
    offset: int,
    max_chars: int | None,
    render: Callable[[str], Prompt],
    fits: Callable[[Prompt], bool],
) -> tuple[Prompt, int] | None:
    """Return a fitting excerpt and its exclusive source offset, or no fit.

    Every fragment repeats its complete speaker label. A continued message is
    marked explicitly. A soft character cap may cross a label to consume at
    least one evidence character; the model capacity check remains mandatory.
    """
    low = offset + 1
    high = min(len(transcript), offset + max_chars) if max_chars else len(transcript)
    first_text = next((max(offset, text_start) for _, text_start, end, _ in spans if end > offset), offset)
    high = max(high, first_text + 1)

    def candidate(stop: int) -> Prompt | None:
        fragments = []
        for start, text_start, end, label in spans:
            if stop <= start or offset >= end:
                continue
            first, last = max(offset, text_start), min(stop, end)
            if first < last:
                continued = "(continued) " if offset > text_start else ""
                fragments.append(label + continued + transcript[first:last])
        if not fragments:
            return None
        excerpt = f"[Conversation excerpt {offset}:{stop} of {len(transcript)} characters]\n"
        return render(excerpt + "\n".join(fragments))

    selected = None
    while low <= high:
        stop = (low + high) // 2
        prompt = candidate(stop)
        if prompt is None:
            low = stop + 1
        elif fits(prompt):
            selected = prompt, stop
            low = stop + 1
        else:
            high = stop - 1
    if selected is not None and selected[1] < len(transcript):
        window = transcript[offset : selected[1]]
        boundary = window.rfind("\n")
        if boundary <= 0:
            boundary = max(window.rfind(" "), window.rfind("\t"))
        if boundary > 0:
            stop = offset + boundary
            prompt = candidate(stop)
            if prompt is not None and fits(prompt):
                selected = prompt, stop
    return selected
