# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Early remote output from the last head group of dense, noncausal attention."""

import math


def eligible(kernel, metadata, distributed_rows):
    owner = getattr(kernel, "__self__", None)
    if getattr(owner, "causal", None) is not False or not hasattr(owner, "effective_attention"):
        return False
    _, backend, _ = owner.effective_attention()
    if backend is None or backend.get_name() != "FLASH_ATTN":
        return False
    if metadata is None:
        return True
    layout = metadata.video_layout
    if layout is None:
        if metadata.extra:
            return False
    else:
        # Dense FA3 ignores grid geometry. Only admit a single fully occupied
        # video grid; masks, packed rows and sparse backends keep their routes.
        grid = metadata.extra.get("vsa_dit_seq_shape")
        if (
            layout.prefix_len != 0
            or layout.video_spans
            or layout.used_len != distributed_rows
            or layout.latent_grid is None
            or math.prod(layout.latent_grid) != distributed_rows
            or len(metadata.extra) != 1
            or type(grid) is not tuple
            or any(type(dim) is not int for dim in grid)
            or grid != layout.latent_grid
            or metadata.joint_query is not None
        ):
            return False
    return (
        (metadata.joint_query is None or metadata.joint_query.shape[1] == 0)
        and metadata.attn_mask is None
        and metadata.joint_attn_mask is None
        and metadata.full_attn_spans is None
        and metadata.query_ranges is None
        and metadata.packed_padding is None
        and metadata.cross_query_mask is None
    )


def candidate(heads, rows, world, sms):
    """Propose the smallest leading head group and a useful local Q tail.

    The 128-row SM90 wave estimate just bounds the real-prefix candidates.
    Prefer the full local return. A half tail is useful near a wave boundary.
    """

    def waves(length, width):
        return math.ceil(math.ceil(length / 128) * width / sms)

    local = rows // world
    head_only = ((heads,),)
    for first in range(1, heads // 2 + 1):
        last = heads - first
        if waves(rows, first) + waves(rows, last) > waves(rows, heads):
            continue
        choices = ((heads,), (first, last))
        if len(head_only) == 1:
            head_only = choices
        for tail in (local, local // 2):
            if tail and waves(rows - tail, last) + waves(tail, last) <= waves(rows, last):
                return choices, tail
    # A useful split can leave only one wave in its second group. In that
    # case ordinary head overlap can still win the real-layer comparison.
    return head_only, 0
