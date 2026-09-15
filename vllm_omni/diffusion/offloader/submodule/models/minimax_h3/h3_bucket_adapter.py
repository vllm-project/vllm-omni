# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3's dense single-request contract for the shared head-bucket pipeline."""

import torch.distributed as dist

from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope

from ...common.head_bucket_adapter import HeadBucketAdapter
from ...common.head_buckets import HeadBucketPlan


class H3BucketAdapter(HeadBucketAdapter):
    def __init__(self, module, group, buckets=2, **kwargs):
        if getattr(module, "to_gate_compress", None) is not None:
            raise ValueError("H3 head-split supports dense attention, not FastH3 VSA")
        if module.num_heads != module.num_kv_heads or module.qkv_proj.total_num_heads != module.num_heads:
            raise ValueError("H3 adapter requires dense MHA and TP=1")
        plan = HeadBucketPlan(
            module.num_heads, module.num_kv_heads, module.head_dim, dist.get_world_size(group), buckets
        )
        super().__init__(module, group, plan, module.qkv_proj, module.out_proj, **kwargs)

    def forward(
        self,
        x,
        *,
        rope_table,
        cu_seqlens,
        max_seqlen,
        packed_total=None,
        num_requests=1,
        sp_seq_lens=None,
        video_layout=None,
        vsa_prefix_segments=(),
    ):
        m = self.module
        if getattr(m, "to_gate_compress", None) is not None:
            raise ValueError("H3 head-split supports dense attention, not FastH3 VSA")
        if num_requests != 1 or packed_total != self.plan.world_size * x.shape[0]:
            raise ValueError("Only equal SP shards of one packed request are supported")
        if not 0 < max_seqlen <= packed_total:
            raise ValueError("Invalid packed valid length")
        if sp_seq_lens is not None and any(length != x.shape[0] for length in sp_seq_lens):
            raise ValueError("Uneven sequence shards need a separate collective plan")
        # Preserve real/padding document boundaries without reading device scalars.
        seq_ends = [packed_total] if max_seqlen == packed_total else [max_seqlen, packed_total]
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() != len(seq_ends) + 1:
            raise ValueError("H3 TND requires the original single-request real/padding boundaries")

        def prepare_qk(q, k):
            if rope_table is None:
                return m.q_norm(q), m.k_norm(k)
            return fused_qk_norm_rope(q, k, m.q_norm.weight, m.k_norm.weight, rope_table, m.q_norm.variance_epsilon)

        return self.run(x, prepare_qk, seq_ends)
