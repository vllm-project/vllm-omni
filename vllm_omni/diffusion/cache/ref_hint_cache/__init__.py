# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reference-hint cache (RFC #4710, P1).

The framework-side backend handles reuse/forecast policy and request state.
Reference-conditioned models expose the acceleration-neutral
``ModelRegion.REFERENCE_HINTS`` seam.
"""

from vllm_omni.diffusion.cache.ref_hint_cache.state import RefHintCacheState

__all__ = ["RefHintCacheState", "RefHintCacheBackend"]


def __getattr__(name):
    # Lazy import so ``RefHintCacheState`` stays importable without vllm/torch deps.
    if name == "RefHintCacheBackend":
        from vllm_omni.diffusion.cache.ref_hint_cache.backend import RefHintCacheBackend

        return RefHintCacheBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
