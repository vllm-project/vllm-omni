# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fail-closed tensor accounting for the released Ming-3.0-TTS checkpoint."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

RELEASE_TENSOR_COUNTS = {
    "model": 9283,
    "audio": 643,
    "flowloss": 236,
    "linear_proj_audio": 119,
    "spk_head": 2,
    "stop_head": 2,
}


def validate_ming3_checkpoint_keys(
    names: Iterable[str],
    *,
    require_release_layout: bool = True,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    unknown: list[str] = []
    for name in names:
        prefix = name.split(".", 1)[0]
        if prefix not in RELEASE_TENSOR_COUNTS:
            unknown.append(name)
        else:
            counts[prefix] += 1
    if unknown:
        raise RuntimeError(f"Ming3 checkpoint contains {len(unknown)} unclassified tensors: {unknown[:8]}")
    actual = {prefix: counts[prefix] for prefix in RELEASE_TENSOR_COUNTS}
    if require_release_layout and actual != RELEASE_TENSOR_COUNTS:
        raise RuntimeError(
            "Ming3 checkpoint tensor counts do not match the validated release: "
            f"expected {RELEASE_TENSOR_COUNTS}, got {actual}"
        )
    return actual


__all__ = ["RELEASE_TENSOR_COUNTS", "validate_ming3_checkpoint_keys"]
