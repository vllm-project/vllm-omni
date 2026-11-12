# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cache-release policy for MiniMax-H3 manual component offload.

Retaining idle allocator blocks does not retain live model tensors or change
their dtype, values, synchronization, or CPU-master rebinding. Explicit bounded
cache owners and failure cleanup remain authoritative.
"""

import os


def release_component_cache() -> bool:
    """Keep the default purge unless the legacy cache-reuse switch is enabled."""
    return os.environ.get("H3_COMPONENT_CACHE_REUSE", "0") != "1"
