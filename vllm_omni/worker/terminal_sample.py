# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU-only terminal drain selection; synchronization stays in the runner."""

from collections.abc import Collection, Mapping
from typing import Any


def find_length_capped_indices(
    *, req_ids: list[str], requests: Mapping[str, Any], max_model_len: int, invalid_req_indices: list[int] | None
) -> list[int]:
    invalid_indices = set(invalid_req_indices or ())
    capped_indices: list[int] = []
    for req_index, request_id in enumerate(req_ids):
        if req_index in invalid_indices:
            continue
        state = requests.get(request_id)
        if state is None or state.sampling_params is None:
            continue
        max_tokens = state.sampling_params.max_tokens
        output_length_capped = max_tokens is not None and len(state.output_token_ids) >= int(max_tokens)
        model_length_capped = state.num_tokens >= max_model_len
        if output_length_capped or model_length_capped:
            capped_indices.append(req_index)
    return capped_indices


def select_terminal_drain_requests(
    *,
    capped_indices: list[int],
    req_ids: list[str],
    sampled_ids_by_index: list[list[int]],
    requests: Mapping[str, Any],
    terminal_token_ids: Collection[int],
) -> list[str]:
    terminal_request_ids: list[str] = []
    for req_index in capped_indices:
        if req_index >= len(sampled_ids_by_index):
            continue
        sampled_ids = sampled_ids_by_index[req_index]
        if not sampled_ids:
            continue
        if len(sampled_ids) != 1:
            raise RuntimeError("Terminal sampled-token drain requires exactly one accepted token per request.")
        token_id = int(sampled_ids[0])
        if token_id not in terminal_token_ids:
            continue
        state = requests[req_ids[req_index]]
        sampling_params = state.sampling_params
        assert sampling_params is not None
        # Match scheduler stop semantics: min_tokens gates all stopping;
        # afterwards EOS/stop-token termination wins before length caps.
        if len(state.output_token_ids) < sampling_params.min_tokens:
            continue
        if token_id == sampling_params.eos_token_id or token_id in (sampling_params.stop_token_ids or ()):
            continue
        terminal_request_ids.append(req_ids[req_index])
    return terminal_request_ids
