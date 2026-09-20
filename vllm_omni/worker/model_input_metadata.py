# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Runner-owned asynchronous CPU token feedback; no model or runner dependency."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class AsyncSampledTokenFeedback:
    """Prior-step sampled tokens needed by CPU-side model preprocess hooks."""

    sampled_token_ids_cpu: torch.Tensor
    ready_event: Any
    req_id_to_index: dict[str, int]
    materialized_rows: list[list[int]] | None = None

    def rows(self) -> list[list[int]]:
        if self.materialized_rows is None:
            if self.sampled_token_ids_cpu.ndim != 2:
                raise RuntimeError(
                    "Async sampled-token feedback must have rank 2, got "
                    f"shape {tuple(self.sampled_token_ids_cpu.shape)}."
                )
            self.ready_event.synchronize()
            self.materialized_rows = [
                [int(token_id) for token_id in row] for row in self.sampled_token_ids_cpu.tolist()
            ]
        return self.materialized_rows


def resolve_async_sampled_token_ids_cpu(
    *,
    feedback: AsyncSampledTokenFeedback | None,
    req_id: str,
    req_index: int,
    start_token_index: int,
    token_ids: tuple[int, ...],
) -> tuple[int, ...]:
    sentinel_indices = [index for index, token_id in enumerate(token_ids) if token_id == -1]
    if not sentinel_indices:
        return token_ids
    if len(sentinel_indices) != 1:
        raise RuntimeError(
            "Omni runner cannot resolve multiple async sampled-token sentinels "
            f"for request {req_id!r} at batch index {req_index}, token start "
            f"{start_token_index}: {token_ids}."
        )

    if feedback is None:
        raise RuntimeError(
            "Omni runner is missing async sampled-token feedback for "
            f"request {req_id!r} at batch index {req_index}, token start "
            f"{start_token_index}."
        )
    previous_index = feedback.req_id_to_index.get(req_id)
    if previous_index is None:
        raise RuntimeError(
            f"Omni runner cannot map async sampled-token feedback for request {req_id!r} at batch index {req_index}."
        )

    rows = feedback.rows()
    if previous_index < 0 or previous_index >= len(rows):
        raise RuntimeError(
            "Omni runner async sampled-token row index is out of range for "
            f"request {req_id!r}: index {previous_index}, rows {len(rows)}."
        )
    sampled_row = rows[previous_index]
    if len(sampled_row) != 1 or sampled_row[0] < 0:
        raise RuntimeError(
            f"Omni runner expected exactly one valid async sampled token for request {req_id!r}, got {sampled_row}."
        )

    resolved = list(token_ids)
    resolved[sentinel_indices[0]] = sampled_row[0]
    return tuple(resolved)
