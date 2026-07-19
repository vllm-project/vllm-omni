# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared utilities for omni schedulers."""

import numpy as np
from vllm.v1.outputs import RoutedExpertsLists


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
