# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Enable Ascend's existing shared KV allocator for MiniCPM-o MRv2.

The worker's budget and the runner's allocator consult the same predicate.
Extend that predicate before either runs, without changing installed Ascend
sources or the allocation policy of other models.
"""

from collections.abc import Iterable
from functools import wraps

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

logger = init_logger(__name__)


def _is_minicpmo_v2_thinker(config: VllmConfig) -> bool:
    model = config.model_config
    return (
        getattr(model, "use_v2_model_runner", False)
        and getattr(model, "model_stage", None) == "llm"
        and "MiniCPMO45OmniForConditionalGeneration" in (model.architectures or ())
    )


def can_share_minicpmo_kv(config: VllmConfig, specs: Iterable[KVCacheSpec]) -> bool:
    """Accept only the validated, unquantized full/duplex DSpark layout."""
    if not _is_minicpmo_v2_thinker(config):
        return False
    if getattr(config.speculative_config, "method", None) != "dspark":
        return False
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_kv import MiniCPMO45DuplexWindowSpec

    specs = tuple(specs)
    if {type(spec) for spec in specs} != {FullAttentionSpec, MiniCPMO45DuplexWindowSpec}:
        return False
    if config.cache_config.cache_dtype not in ("auto", "float16", "bfloat16"):
        return False
    if config.kv_transfer_config is not None:
        return False
    parallel = config.parallel_config
    if parallel.decode_context_parallel_size != 1 or parallel.prefill_context_parallel_size != 1:
        return False
    layout = config.cache_config.get_resolved_kv_cache_layout()
    if not (layout.is_layer_compact and layout.is_block_compact):
        return False
    from vllm_ascend.ascend_config import KVPPConfig
    from vllm_ascend.worker.v2.attn_utils import KVQuantMode, enable_sfa

    if enable_sfa(config) or KVPPConfig.from_vllm_config(config).size > 1:
        return False
    return len({(s.block_size, s.num_kv_heads, s.head_size, s.dtype, s.page_size_bytes) for s in specs}) == 1 and all(
        s.dtype in (torch.float16, torch.bfloat16)
        and s.kv_quant_mode == KVQuantMode.NONE
        and s.head_size == s.head_size_v
        and s.page_size_bytes == s.real_page_size_bytes
        and s.tokens_per_state == 1
        and s.num_head_slots is None
        and s.state_content_bytes is None
        for s in specs
    )


def install_minicpmo_shared_kv(config: VllmConfig) -> None:
    """Install once in a MiniCPM-o Thinker worker before KV budgeting."""
    if not _is_minicpmo_v2_thinker(config):
        return
    from vllm_ascend.worker.v2 import attn_utils

    original = attn_utils.can_share_full_and_sliding_kv_cache
    if getattr(original, "_omni_minicpmo_shared_kv", False):
        return

    @wraps(original)
    def can_share(config: VllmConfig, specs: Iterable[KVCacheSpec]) -> bool:
        specs = tuple(specs)
        return original(config, specs) or can_share_minicpmo_kv(config, specs)

    setattr(can_share, "_omni_minicpmo_shared_kv", True)
    attn_utils.can_share_full_and_sliding_kv_cache = can_share
    logger.info("Enabled MiniCPM-o MRv2 shared KV eligibility for Ascend budget and allocation")
