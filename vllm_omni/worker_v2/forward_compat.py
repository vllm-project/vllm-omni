# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Forward-call compatibility helpers for Omni MR V2 runners."""

from __future__ import annotations

from typing import Any


def add_forward_compat_kwargs(
    model_inputs: dict[str, Any],
    input_batch: Any,
    sampler: Any,
) -> None:
    """Add Omni model forward kwargs that V1 model runners provided.

    Some Omni model stages inspect these kwargs inside ``forward`` rather
    than only in the runner.  Qwen2.5-Omni's talker, for example, rewrites
    text-model pad ids in ``sampling_metadata`` to a talker-vocab pad id
    before sampling codec tokens.
    """
    model_inputs.setdefault(
        "sampling_metadata",
        getattr(input_batch, "sampling_metadata", None),
    )
    logits_index = getattr(
        input_batch,
        "logits_indices",
        getattr(input_batch, "logits_index", None),
    )
    model_inputs.setdefault("logits_index", logits_index)
    model_inputs.setdefault("sampler", sampler)
