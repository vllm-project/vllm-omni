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

``_split_other_mm`` is the exception at *call* time: it lazily
imports ``vllm_omni.utils.mm_outputs``, which pulls in vllm. Import
of this package still succeeds without vllm; merge_batch of leftover
mm does not.
"""

from vllm_omni.core.prefix_cache.adapter import (
    PrefixCacheEventKind,
    PrefixCacheRequestEvent,
    PrefixCacheSchedulerAdapter,
    PrefixCacheStep,
    PrefixCacheWrite,
    PrefixCacheWriteLayout,
)
from vllm_omni.core.prefix_cache.group_view import (
    FullAttentionGroupView,
    check_prefix_cache_kv_groups,
    check_prefix_cache_kv_transfer,
    get_prefix_cache_group_view,
    stage_prefix_cache_config,
)
from vllm_omni.core.prefix_cache.interface import (
    HIDDEN_KEY,
    ModelCachePolicy,
    OmniPrefixCacheUnmatchError,
    PrefixCacheConfig,
    StageCacheOutputs,
)
from vllm_omni.core.prefix_cache.manager import OmniPrefixCacheManager
from vllm_omni.core.prefix_cache.slot_index import SlotIndex

__all__ = [
    "HIDDEN_KEY",
    "FullAttentionGroupView",
    "ModelCachePolicy",
    "OmniPrefixCacheManager",
    "PrefixCacheEventKind",
    "PrefixCacheRequestEvent",
    "PrefixCacheSchedulerAdapter",
    "PrefixCacheStep",
    "PrefixCacheWrite",
    "PrefixCacheWriteLayout",
    "OmniPrefixCacheUnmatchError",
    "StageCacheOutputs",
    "PrefixCacheConfig",
    "SlotIndex",
    "check_prefix_cache_kv_groups",
    "check_prefix_cache_kv_transfer",
    "get_prefix_cache_group_view",
    "stage_prefix_cache_config",
]
