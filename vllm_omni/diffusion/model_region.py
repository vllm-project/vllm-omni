# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Acceleration-neutral execution seams for semantic model regions.

Models may expose a named region whose output is useful to framework services
such as tracing, profiling, or caching.  The model does not know which service,
if any, handles the region.  Without a handler in the active ForwardContext,
``execute_model_region`` calls the supplied computation directly.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Protocol, TypeVar

import torch.nn as nn

T = TypeVar("T")


class ModelRegion(str, Enum):
    """Stable semantic model regions understood by optional framework services."""

    REFERENCE_HINTS = "reference_hints"


class ModelRegionHandler(Protocol):
    """Request-scoped handler for a named semantic model region."""

    def execute(
        self,
        region: ModelRegion,
        owner: nn.Module,
        compute: Callable[[], T],
    ) -> T:
        """Return the region output, invoking ``compute`` when needed."""
        ...


def execute_model_region(
    region: ModelRegion,
    owner: nn.Module,
    compute: Callable[[], T],
) -> T:
    """Execute a semantic model region through the active request handler.

    The direct-compute path is the default and preserves model behavior when no
    framework service has registered a handler.
    """
    from vllm_omni.diffusion.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if not is_forward_context_available():
        return compute()
    handler = get_forward_context().model_region_handler
    if handler is None:
        return compute()
    return handler.execute(region, owner, compute)
