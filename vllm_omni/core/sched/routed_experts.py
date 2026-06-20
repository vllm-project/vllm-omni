# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility helpers for routed-experts scheduler outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
else:
    NDArray = np.ndarray

try:
    from vllm.v1.outputs import RoutedExpertsLists as VllmRoutedExpertsLists
except ImportError:

    @dataclass
    class VllmRoutedExpertsLists:
        """Fallback shape used by vLLM versions without the export."""

        routing_data: NDArray[np.int32] | NDArray[np.int64] | np.ndarray
        slot_mapping: NDArray[np.int32] | NDArray[np.int64] | np.ndarray


RoutedExpertsLists = VllmRoutedExpertsLists

__all__ = ["RoutedExpertsLists"]
