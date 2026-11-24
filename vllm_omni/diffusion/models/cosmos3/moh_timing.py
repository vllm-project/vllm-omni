# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Versioned timing boundary shared by the worker, HTTP bridge and evaluator."""

import math
from typing import Any

MOH_TIMING_CONTRACT = "moh_sample_actions_v1"


def model_infer_ms(timing: Any, contract: Any) -> float | None:
    """Accept only a worker's explicitly versioned sampler measurement.

    Old policy timings measured HTTP round trips. Never silently reuse them
    as model timings, or substitute job/pipeline/profiler durations.
    """
    if contract != MOH_TIMING_CONTRACT or not isinstance(timing, dict):
        return None
    value = timing.get("infer_ms")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None
