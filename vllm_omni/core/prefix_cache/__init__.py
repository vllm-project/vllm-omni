# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Omni prefix cache.

Structure mirrors vllm/v1/core.

Import invariant: this package must not import ``vllm`` at module
scope (``TYPE_CHECKING`` and function-body imports are fine).
``tests/core/test_prefix_cache.py`` loads these modules without a
vllm install; a top-level ``from vllm...`` turns that suite red.

That is also why these files use ``logging.getLogger`` instead of
the repo-wide ``init_logger`` — ``init_logger`` comes from
``vllm.logger``. The deviation is required, not an oversight.

``_merge_uncached_mm`` is the exception at *call* time: it lazily
imports ``vllm_omni.utils.mm_outputs``, which pulls in vllm. Import
of this package still succeeds without vllm; leftover materialize
of uncached mm does not.
"""

from vllm_omni.core.prefix_cache.block_pool import PrefixBlockPool
from vllm_omni.core.prefix_cache.controller import OmniPrefixCacheController
from vllm_omni.core.prefix_cache.group_view import (
    FullAttentionGroupView,
    check_prefix_cache_kv_groups,
    get_prefix_cache_group_view,
)
from vllm_omni.core.prefix_cache.interface import (
    HIDDEN_KEY,
    ModelCachePolicy,
    OmniPrefixCacheUnmatchError,
    PrefixCacheConfig,
    StageCacheOutputs,
    WriteSchedule,
)
from vllm_omni.core.prefix_cache.manager import OmniPrefixCacheManager

__all__ = [
    "HIDDEN_KEY",
    "FullAttentionGroupView",
    "ModelCachePolicy",
    "OmniPrefixCacheController",
    "OmniPrefixCacheManager",
    "OmniPrefixCacheUnmatchError",
    "StageCacheOutputs",
    "PrefixBlockPool",
    "PrefixCacheConfig",
    "WriteSchedule",
    "check_prefix_cache_kv_groups",
    "get_prefix_cache_group_view",
]
