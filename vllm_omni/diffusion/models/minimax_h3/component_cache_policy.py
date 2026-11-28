# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cache-release policy for MiniMax-H3 manual component offload.

Retaining idle allocator blocks does not retain live model tensors or change
their dtype, values, synchronization, or CPU-master rebinding. Explicit bounded
cache owners and failure cleanup remain authoritative.
"""

from collections.abc import Mapping


def resolve_component_cache_reuse(additional_config: Mapping[str, object] | None) -> bool:
    """Read the optional boolean allocator policy at model construction."""
    value = (additional_config or {}).get("component_cache_reuse", False)
    if not isinstance(value, bool):
        raise ValueError("additional_config.component_cache_reuse must be a boolean")
    return value
