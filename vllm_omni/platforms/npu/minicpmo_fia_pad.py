# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FIA pad-to-bucket for the MiniCPM-o 4.5 stage1 talker decode graph.

The stock FULL-graph bucket-1 capture bakes a dummy kv descriptor and
``sparse_mode=3``, so every replay frame must re-issue all 20 FIA layer ops
through ``graph_task_update`` to re-bake the real kv length (~2.3ms of host
time per frame). This patch re-forges the capture instead:

* kv descriptor baked at 512 (the bucket) instead of the dummy value;
* ``sparse_mode=0`` with a persistent wide int8 mask ``[1,1,1,4096]`` whose
  content is rebuilt *inside* the ACL graph on every replay from a
  persistent klen device scalar (``mask[kv] = 1 iff kv >= klen``; the
  pointer is baked, the content is mutable -- same law as the block-table
  data);
* the baked block_table is a narrow contiguous view (``bucket / 128``
  columns, same base pointer as the persistent buffer) so the kernel does
  not traverse the full 4096-token extent;
* replay frames then only re-arm the baked ``ExternalEvent`` set (the
  phase-0-probe validated deadlock guard) instead of re-issuing ops.

Correctness: masked lanes contribute ``exp(-inf) = +0`` exactly, and
microbenchmarks (fia_mb*, 2026-08-30) show outputs matching the stock
rightDownCausal path bit-for-bit while ``klen <= bucket``. The KV pool is
zero-filled once at capture time because NaN pad memory *does* leak through
the mask (the mask is applied as a score bias, not a select; ``NaN +
(-inf)`` stays NaN). klen beyond the bucket promotes the baked kv to the
next bucket in ``{512, 1024, 2048, 4096}`` with one stock update pass; a
new short request demotes back to 512 the same way.

Kill switch: ``MINICPMO_FIA_PAD=0`` (stock capture/update everywhere).
``MINICPMO_FIA_PAD=2`` forges the pad capture but keeps the stock per-frame
update loop (degraded smoke mode).
"""

from __future__ import annotations

import os

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False
_orig_full_graph_fia = None

_TALKER_ARCHS = frozenset(
    {"MiniCPMO", "MiniCPMO45OmniForConditionalGeneration"}
)


def fia_pad_mode() -> int:
    """0 = off, 1 = full (default), 2 = capture forge only (degraded smoke)."""
    return int(os.environ.get("MINICPMO_FIA_PAD", "1"))


class FiaPadState:
    # gate (set once by the stage1 runner __init__)
    enabled = False
    mode = 1
    degraded = False        # True -> stock per-frame update loop (correct, slow)
    # capture forge lifecycle
    capturing_pad = False    # True only during the bucket-1 FULL capture pass
    rebuild_emitted = False  # mask-rebuild ops emitted into current capture
    pad_captured = False     # bucket-1 pad capture completed this boot
    # kv descriptor currently baked into the captured graph
    baked_kv = 0
    # bucket ladder
    KV_PAD = 512
    KV_MAX = 4096
    BUCKETS = (512, 1024, 2048, 4096)
    # runtime state
    skip_logged = False
    pool_zeroed = False
    # persistent buffers (allocated by the stage1 runner; pointers baked
    # into the captured graph)
    klen_dev = None          # int32 [1] device
    klen_host = None         # int32 [1] pinned host mirror
    mask_buf = None          # int8 [1,1,1,KV_MAX] device
    arange_buf = None        # int32 [1,1,1,KV_MAX] device, constant content
    cmp_buf = None           # bool  [1,1,1,KV_MAX] device, rebuild scratch


STATE = FiaPadState()


def _patched_full_graph_fia(self, query, key, value, attn_metadata, output, layer=None):
    st = STATE
    if st.enabled and st.capturing_pad:
        if not st.pool_zeroed:
            # NaN pad memory leaks through the mask bias; finite pad memory
            # is exactly invisible. One-time zero-fill of the whole pool
            # (blocks are only ever rewritten with finite KV data after).
            if self.key_cache is not None:
                self.key_cache.zero_()
            if self.value_cache is not None:
                self.value_cache.zero_()
            st.pool_zeroed = True
        if not st.rebuild_emitted:
            # Plain elementwise ops with out=: no allocation, captured INSIDE
            # the ACL graph, re-executed on every replay reading the CURRENT
            # klen_dev content (pointer baked, content mutable).
            torch.ge(st.arange_buf, st.klen_dev, out=st.cmp_buf)
            st.mask_buf.copy_(st.cmp_buf)
            st.rebuild_emitted = True
    return _orig_full_graph_fia(self, query, key, value, attn_metadata, output, layer)


def apply_minicpmo_fia_pad_patch() -> None:
    global _PATCHED, _orig_full_graph_fia
    if _PATCHED:
        return
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

    _orig_full_graph_fia = AscendAttentionBackendImpl.full_graph_fia
    AscendAttentionBackendImpl.full_graph_fia = _patched_full_graph_fia
    _PATCHED = True
    logger.info("[fia_pad] full_graph_fia wrapper installed (per-engine gate decides)")


def talker_gate(model_config) -> bool:
    # The minicpmo 4.5 pipeline resolves BOTH stage0 (thinker) and stage1
    # (talker) to the same top-level arch; stage1 is the tts sub-config
    # stage (see model_executor/models/minicpmo_4_5/pipeline.py). The stage
    # id is the reliable discriminator (omni_ar_scheduler uses the same
    # signal for its stage-1 talker paths).
    if getattr(model_config, "stage_id", None) != 1:
        return False
    archs = set(getattr(model_config, "architectures", None) or [])
    arch = getattr(model_config, "model_arch", None)
    if arch:
        archs.add(arch)
    return bool(archs & _TALKER_ARCHS)
