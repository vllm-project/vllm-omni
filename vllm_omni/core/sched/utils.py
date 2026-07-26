# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared utilities for omni schedulers."""

from typing import Any

import numpy as np
from vllm.v1.outputs import RoutedExpertsLists


def split_free_request_result(
    result: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Normalize old KV-only and new KV/EC scheduler cleanup results."""
    if result is None or isinstance(result, dict):
        return result, None
    if isinstance(result, tuple) and len(result) == 2:
        kv_transfer_params, ec_transfer_params = result
        if kv_transfer_params is not None and not isinstance(kv_transfer_params, dict):
            raise TypeError("_free_request KV transfer params must be a dict or None")
        if ec_transfer_params is not None and not isinstance(ec_transfer_params, dict):
            raise TypeError("_free_request EC transfer params must be a dict or None")
        return kv_transfer_params, ec_transfer_params
    raise TypeError(f"unsupported _free_request result: {type(result).__name__}")


def omni_routed_experts_for_request(routed_experts: RoutedExpertsLists, request) -> np.ndarray | None:
    """Extract per-request routed experts from RoutedExpertsLists using slot_mapping.

    Matches upstream RoutedExpertsManager.get() pattern — filters routing_data
    rows whose slot_mapping entries belong to this request's block_table.
    """
    if routed_experts is None:
        return None
    slots = getattr(request, "block_table", None)
    if slots is None:
        return None
    slot_set = set(slots)
    mask = np.isin(routed_experts.slot_mapping, list(slot_set))
    data = routed_experts.routing_data[mask]
    return data if data.size > 0 else None
