# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Helpers for models that call `torch.compile` directly, outside vLLM's compile wrapper."""

import contextlib
import logging
import os
import time
from collections.abc import Iterator

from torch._dynamo.utils import counters


@contextlib.contextmanager
def log_torch_compile_stats(logger: logging.Logger, label: str) -> Iterator[None]:
    """Log the block's duration and its Inductor FX graph cache hits, misses and bypasses.

    `torch.compile` is lazy, so wrap the first calls (e.g. a warmup), not the `torch.compile` call.
    """
    before = counters["inductor"].copy()
    started = time.perf_counter()
    yield
    compiled = counters["inductor"] - before
    logger.info(
        "%s: compiled region took %.1f s; Inductor cache %s: %d hits, %d misses, %d bypasses",
        label,
        time.perf_counter() - started,
        os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        compiled["fxgraph_cache_hit"],
        compiled["fxgraph_cache_miss"],
        compiled["fxgraph_cache_bypass"],
    )
