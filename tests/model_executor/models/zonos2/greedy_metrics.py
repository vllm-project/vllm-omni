# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict P2-09 metrics for aligned raw nine-codebook greedy frame streams."""

from __future__ import annotations

from typing import Any

import torch


def greedy_code_parity(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    for name, codes in (("reference", reference), ("actual", actual)):
        if codes.ndim != 2 or codes.shape[1] != 9:
            raise ValueError(f"{name} codes must have shape [T,9]")
        if codes.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} codes must be int32 or int64")
        if codes.numel() and (bool((codes < 0).any()) or bool((codes > 1025).any())):
            raise ValueError(f"{name} contains out-of-vocabulary codes")
    ref_len, actual_len = len(reference), len(actual)
    overlap = min(ref_len, actual_len)
    total = max(ref_len, actual_len)
    equal = reference[:overlap] == actual[:overlap]
    prefix_compared = min(overlap, 12)
    prefix_matches = int(equal[:prefix_compared].sum())
    prefix_pass = overlap >= 12 and prefix_matches == 108
    frame_matches = int(equal.all(dim=-1).sum())
    frame_agreement = frame_matches / total if total else 0.0
    bad = (~equal).nonzero()
    examples = [
        {
            "frame": int(row),
            "codebook": int(col),
            "reference": int(reference[row, col]),
            "actual": int(actual[row, col]),
        }
        for row, col in bad[:32].tolist()
    ]
    first_difference = int(bad[0, 0]) if len(bad) else (overlap if ref_len != actual_len else None)
    return {
        "reference_frames": ref_len,
        "actual_frames": actual_len,
        "prefix_required_frames": 12,
        "prefix_required_codes": 108,
        "prefix_compared_frames": prefix_compared,
        "prefix_matches": prefix_matches,
        "prefix_pass": prefix_pass,
        "whole_frame_matches": frame_matches,
        "whole_frame_total": total,
        "whole_frame_agreement": frame_agreement,
        "code_matches": int(equal.sum()),
        "code_total": total * 9,
        "code_agreement": int(equal.sum()) / (total * 9) if total else 0.0,
        "per_codebook_agreement": [(int(equal[:, j].sum()) / total if total else 0.0) for j in range(9)],
        "unpaired_tail_frames": abs(ref_len - actual_len),
        "first_difference_frame": first_difference,
        "mismatch_examples": examples,
        "full_sequence_threshold": 0.95,
        "l2_pass": prefix_pass and frame_agreement >= 0.95,
    }
