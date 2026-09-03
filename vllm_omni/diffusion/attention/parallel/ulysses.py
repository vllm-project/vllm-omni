# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel.base import ParallelAttentionContext
from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D
from vllm_omni.diffusion.distributed.group_coordinator import SequenceParallelGroupCoordinator
from vllm_omni.diffusion.forward_context import get_ulysses_mode

logger = init_logger(__name__)

# When advanced_uaa pads Q by the GQA ratio, MQA/very-uneven-GQA shapes can
# inflate the query-head count substantially (worst case: MQA @ U=N pads Q from
# H to N*H, an N x blow-up in Q attention work and temporary VRAM). Warn once
# per (Q, KV, world_size) tuple when the ratio crosses this threshold so users
# can pick a friendlier ulysses_degree if the overhead is unacceptable.
_UAA_PAD_RATIO_WARN_THRESHOLD = 1.5
_uaa_pad_ratio_warned: set[tuple[int, int, int, str]] = set()


def _maybe_warn_uaa_pad_ratio(
    query_head_cnt: int,
    kv_head_cnt: int,
    padded_query_head_cnt: int,
    ulysses_world_size: int,
    tensor_label: str,
) -> None:
    """Emit a one-shot warning when advanced_uaa padding materially inflates Q.

    The padding itself is mandatory for correctness (Ulysses splits along the
    head dim, so KV must be a multiple of ulysses_world_size and Q must stay a
    multiple of KV to remain a valid GQA shape). There is no cheap fallback,
    but the caller may want to know when the inflation is large enough to
    justify picking a different ulysses_degree.
    """
    if query_head_cnt <= 0 or padded_query_head_cnt <= query_head_cnt:
        return
    ratio = padded_query_head_cnt / query_head_cnt
    if ratio < _UAA_PAD_RATIO_WARN_THRESHOLD:
        return
    key = (query_head_cnt, kv_head_cnt, ulysses_world_size, tensor_label)
    if key in _uaa_pad_ratio_warned:
        return
    _uaa_pad_ratio_warned.add(key)
    logger.warning(
        "Ulysses advanced_uaa GQA padding inflates %s heads %.2fx "
        "(Q=%d, KV=%d, ulysses_degree=%d -> padded Q=%d). "
        "Attention FLOPs and temporary VRAM for this tensor grow by the same "
        "factor. If this is unacceptable, choose a ulysses_degree that divides "
        "KV_heads (or the GCD of Q/KV) to reduce or eliminate padding.",
        tensor_label,
        ratio,
        query_head_cnt,
        kv_head_cnt,
        ulysses_world_size,
        padded_query_head_cnt,
    )


def _a2a_permute_enabled(enabled: bool, scatter_idx: int, gather_idx: int, world_size: int) -> bool:
    """Whether the fused permute-free all-to-all applies (strict layout only)."""
    return enabled and world_size > 1 and scatter_idx == 2 and gather_idx == 1


def _ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def _positive_divisors(n: int) -> list[int]:
    if n <= 0:
        return []
    divs = set()
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
        i += 1
    return sorted(divs)


@torch.compiler.disable
def _all_gather_int(pg: dist.ProcessGroup, value: int, *, device: torch.device) -> list[int]:
    """All-gather a scalar int across pg.

    Note: we use a device tensor so this works for NCCL subgroups (e.g. Ulysses/Ring).
    """
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return [int(value)]

    t = torch.tensor([int(value)], dtype=torch.int64, device=device)
    gathered = [torch.empty_like(t) for _ in range(world_size)]
    dist.all_gather(gathered, t, group=pg)
    return [int(x.item()) for x in gathered]


def _ulysses_all_to_all_any_qkv(
    pg: dist.ProcessGroup,
    x: torch.Tensor,  # (B, S_local, H, D)
    *,
    seq_lens: list[int],
    use_sync: bool,
    padded_head_cnt: int | None = None,
) -> tuple[torch.Tensor, int]:
    """UAA forward all-to-all: (B, S_local, H, D) -> (B, S_global, H_local, D).

    Returns:
        (resharded, orig_head_cnt)
    """
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return x, int(x.shape[2])

    bsz, s_local, head_cnt, head_dim = x.shape
    orig_head_cnt = int(head_cnt)
    if padded_head_cnt is None:
        padded_head_cnt = _ceil_div(orig_head_cnt, world_size) * world_size
    if padded_head_cnt < orig_head_cnt or padded_head_cnt % world_size != 0:
        raise ValueError(
            f"Invalid padded head count {padded_head_cnt} for original heads={orig_head_cnt}, world_size={world_size}."
        )
    head_pad = padded_head_cnt - orig_head_cnt
    if head_pad:
        x = F.pad(x, (0, 0, 0, head_pad))

    head_cnt_local = padded_head_cnt // world_size

    # (B, S_local, H, D) -> (world_size, S_local, B, H_local, D)
    x_t = x.reshape(bsz, s_local, world_size, head_cnt_local, head_dim).permute(2, 1, 0, 3, 4).contiguous()
    # (world_size, S_local, B, H_local, D) -> (world_size * S_local, B, H_local, D)
    x_t = x_t.flatten(0, 1)

    input_split_sizes = [s_local] * world_size
    output_split_sizes = seq_lens
    s_global = int(sum(output_split_sizes))

    out = torch.empty((s_global, bsz, head_cnt_local, head_dim), device=x.device, dtype=x.dtype)
    dist.all_to_all_single(
        out,
        x_t,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=pg,
    )
    if use_sync:
        from vllm_omni.platforms import current_omni_platform

        current_omni_platform.synchronize()

    # (S_global, B, H_local, D) -> (B, S_global, H_local, D)
    out = out.permute(1, 0, 2, 3).contiguous()
    return out, orig_head_cnt


def _ulysses_all_to_all_any_o(
    pg: dist.ProcessGroup,
    x: torch.Tensor,  # (B, S_global, H_local, D)
    *,
    seq_lens: list[int],
    local_seq_len: int,
    orig_head_cnt: int,
    use_sync: bool,
) -> torch.Tensor:
    """UAA reverse all-to-all: (B, S_global, H_local, D) -> (B, S_local, H, D)."""
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return x

    bsz, s_global, head_cnt_local, head_dim = x.shape
    s_local = int(local_seq_len)

    # (B, S_global, H_local, D) -> (S_global, B, H_local, D)
    x_t = x.permute(1, 0, 2, 3).contiguous()

    input_split_sizes = seq_lens
    output_split_sizes = [s_local] * world_size

    out = torch.empty((world_size * s_local, bsz, head_cnt_local, head_dim), device=x.device, dtype=x.dtype)
    dist.all_to_all_single(
        out,
        x_t,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=pg,
    )
    if use_sync:
        from vllm_omni.platforms import current_omni_platform

        current_omni_platform.synchronize()

    # (world_size * S_local, B, H_local, D) -> (B, S_local, H, D)
    out = out.reshape(world_size, s_local, bsz, head_cnt_local, head_dim).permute(2, 1, 0, 3, 4).contiguous()
    out = out.reshape(bsz, s_local, world_size * head_cnt_local, head_dim)

    if out.shape[2] != orig_head_cnt:
        out = out[:, :, :orig_head_cnt, :].contiguous()
    return out


@dataclass(frozen=True, slots=True)
class _UlyssesCtx(ParallelAttentionContext):
    """Per-forward context for Ulysses sequence-parallel attention."""

    ulysses_pg: dist.ProcessGroup
    scatter_idx: int
    gather_idx: int
    use_sync: bool
    joint_len: int = 0
    joint_strategy: str = "front"
    # UAA (Ulysses Anything Attention) metadata
    use_uaa: bool = False
    uaa_seq_lens: tuple[int, ...] = ()
    uaa_local_seq_len: int = 0
    orig_head_cnt: int = 0
    joint_orig_head_cnt: int = 0


class UlyssesParallelAttention:
    """Ulysses sequence-parallel strategy (all-to-all over seq/head dims).

    This preserves the semantics previously implemented in
    `Attention._forward_ulysses`:
    - If `AttentionMetadata.joint_*` is provided, joint_query/key/value are
      concatenated *after* all-to-all.
    - joint_key/value are assumed to be replicated across SP ranks and are sliced
      by ulysses head rank before concatenation.
    """

    def __init__(
        self,
        sp_group: SequenceParallelGroupCoordinator,
        scatter_idx: int,
        gather_idx: int,
        use_sync: bool,
        ulysses_a2a_permute: bool = False,
    ) -> None:
        self._sp_group = sp_group
        self._ulysses_pg = sp_group.ulysses_group
        self._scatter_idx = scatter_idx
        self._gather_idx = gather_idx
        self._use_sync = use_sync
        self._ulysses_a2a_permute = ulysses_a2a_permute
        if _a2a_permute_enabled(
            ulysses_a2a_permute,
            scatter_idx,
            gather_idx,
            sp_group.ulysses_world_size,
        ):
            from vllm_omni.diffusion.distributed.a2a_permute import ensure_a2a_permute_available

            ensure_a2a_permute_available()

    @property
    def enabled(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "ulysses"

    def pre_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ):
        mode = get_ulysses_mode(default="strict")
        ulysses_world_size = self._sp_group.ulysses_world_size
        gate_compress = None
        if attn_metadata is not None:
            candidate = attn_metadata.extra.get("gate_compress")
            if isinstance(candidate, torch.Tensor):
                if candidate.shape != query.shape:
                    raise ValueError(
                        f"gate_compress must match pre-Ulysses query shape, got {candidate.shape} vs {query.shape}"
                    )
                gate_compress = candidate

        # advanced_uaa pads non-divisible head counts before the Ulysses
        # all-to-all. This also holds in hybrid Ulysses+Ring: Q is derived
        # from the padded K/V count by the GQA ratio (see below), so padded
        # heads pair with padded heads; non-GQA layouts are rejected below.

        joint_tensor_query = joint_tensor_key = joint_tensor_value = None
        joint_strategy = "front"
        joint_len = 0
        joint_orig_head_cnt = 0

        if attn_metadata is not None:
            joint_tensor_query = attn_metadata.joint_query
            joint_tensor_key = attn_metadata.joint_key
            joint_tensor_value = attn_metadata.joint_value
            joint_strategy = attn_metadata.joint_strategy

        is_joint = False
        if joint_tensor_query is not None and joint_tensor_key is not None and joint_tensor_value is not None:
            supported_joint_strategy = ["front", "rear"]
            if joint_strategy not in supported_joint_strategy:
                raise ValueError(
                    f"joint_strategy: {joint_strategy} not supported."
                    f" supported joint strategy: {supported_joint_strategy}"
                )

            # Slice joint_query for this Ulysses rank
            # joint_query is (B, S, H, D). We split H (dim 2).
            ulysses_rank = self._sp_group.ulysses_rank
            joint_q_head_cnt = int(joint_tensor_query.shape[-2])
            joint_k_head_cnt = int(joint_tensor_key.shape[-2])
            joint_v_head_cnt = int(joint_tensor_value.shape[-2])
            joint_orig_head_cnt = joint_q_head_cnt

            if joint_k_head_cnt != joint_v_head_cnt or joint_q_head_cnt % joint_k_head_cnt != 0:
                raise ValueError(
                    "Ulysses joint attention requires equal K/V head counts and joint query "
                    "heads to be a multiple of joint KV heads, got "
                    f"joint_Q={joint_q_head_cnt}, joint_K={joint_k_head_cnt}, joint_V={joint_v_head_cnt}."
                )

            if mode == "advanced_uaa":
                # Pad joint KV to a world_size multiple, then derive joint Q by the
                # GQA ratio so the ratio survives the head-dim split (mirrors the
                # main-tensor path below; e.g. 28Q/7KV at SP2 becomes 32/8).
                padded_joint_kv_head_cnt = _ceil_div(joint_k_head_cnt, ulysses_world_size) * ulysses_world_size
                gqa_ratio = joint_q_head_cnt // joint_k_head_cnt
                padded_joint_q_head_cnt = padded_joint_kv_head_cnt * gqa_ratio
                _maybe_warn_uaa_pad_ratio(
                    joint_q_head_cnt,
                    joint_k_head_cnt,
                    padded_joint_q_head_cnt,
                    ulysses_world_size,
                    "joint_query",
                )

                joint_q_pad = padded_joint_q_head_cnt - joint_q_head_cnt
                joint_kv_pad = padded_joint_kv_head_cnt - joint_k_head_cnt
                if joint_q_pad:
                    joint_tensor_query = F.pad(joint_tensor_query, (0, 0, 0, joint_q_pad))
                if joint_kv_pad:
                    joint_tensor_key = F.pad(joint_tensor_key, (0, 0, 0, joint_kv_pad))
                    joint_tensor_value = F.pad(joint_tensor_value, (0, 0, 0, joint_kv_pad))
                joint_q_head_cnt = padded_joint_q_head_cnt
                joint_k_head_cnt = padded_joint_kv_head_cnt
            else:
                for name, cnt in (
                    ("joint_query", joint_q_head_cnt),
                    ("joint_key", joint_k_head_cnt),
                    ("joint_value", joint_v_head_cnt),
                ):
                    if cnt % ulysses_world_size != 0:
                        supported = _positive_divisors(cnt)
                        raise ValueError(
                            "Ulysses-SP strict mode requires joint head_cnt divisible by ulysses_degree. "
                            f"{name}_head_cnt={cnt}, ulysses_degree={ulysses_world_size}. "
                            f"Try ulysses_degree in {supported}, or set ulysses_mode='advanced_uaa'."
                        )

            attn_heads_per_ulysses_rank_q = joint_q_head_cnt // ulysses_world_size

            joint_tensor_query = joint_tensor_query[
                ...,
                attn_heads_per_ulysses_rank_q * ulysses_rank : attn_heads_per_ulysses_rank_q * (ulysses_rank + 1),
                :,
            ]

            joint_len = joint_tensor_query.shape[1]

            is_joint = True
        elif joint_tensor_query is None and joint_tensor_key is None and joint_tensor_value is None:
            pass
        else:
            raise ValueError("joint_query, joint_key, and joint_value should be None or not None simultaneously.")

        if is_joint:
            # Slice joint key/value heads for this ulysses rank.
            # Using the KV head count (post-pad in advanced_uaa) so the per-rank
            # KV shape matches the main K/V slice and preserves the GQA ratio.
            attn_heads_per_ulysses_rank_kv = joint_k_head_cnt // ulysses_world_size

            joint_tensor_key = joint_tensor_key[
                ...,
                attn_heads_per_ulysses_rank_kv * ulysses_rank : attn_heads_per_ulysses_rank_kv * (ulysses_rank + 1),
                :,
            ]
            joint_tensor_value = joint_tensor_value[
                ...,
                attn_heads_per_ulysses_rank_kv * ulysses_rank : attn_heads_per_ulysses_rank_kv * (ulysses_rank + 1),
                :,
            ]

            # Update metadata with sliced tensors so Ring attention can use them if needed
            if attn_metadata is not None:
                attn_metadata.joint_key = joint_tensor_key
                attn_metadata.joint_value = joint_tensor_value

        ulysses_world_size = self._sp_group.ulysses_world_size
        if mode == "advanced_uaa":
            if self._scatter_idx != 2 or self._gather_idx != 1:
                raise ValueError(
                    "ulysses_mode='advanced_uaa' currently only supports scatter_idx=2, gather_idx=1 "
                    f"(got scatter_idx={self._scatter_idx}, gather_idx={self._gather_idx})."
                )

            local_seq_len = int(query.shape[1])
            seq_lens = _all_gather_int(self._ulysses_pg, local_seq_len, device=query.device)
            s_global = int(sum(seq_lens))

            # In hybrid Ulysses+Ring, Ring attention uses P2P send/recv with fixed-shape
            # buffers. This requires all ring ranks to have the same seq_len after the
            # Ulysses all-to-all (i.e. per-ring-rank S_global must match).
            if self._sp_group.ring_world_size > 1:
                ring_s_globals = _all_gather_int(self._sp_group.ring_group, s_global, device=query.device)
                if len(set(ring_s_globals)) != 1:
                    raise ValueError(
                        "ulysses_mode='advanced_uaa' with hybrid Ulysses+Ring requires the "
                        "post-Ulysses seq_len to be equal across ring ranks, but got "
                        f"{ring_s_globals} (ring_degree={self._sp_group.ring_world_size}). "
                        "This typically means the input sequence was not evenly shardable across the ring. "
                        "Try setting ring_degree=1, or choose a sequence length divisible by ring_degree."
                    )

            # Pad KV to a world_size multiple, then scale Q by the GQA ratio so
            # the ratio survives the head-dim split (e.g. 28Q/7KV at SP2 becomes
            # 32/8, i.e. 16/4 per rank -- padding each independently would give
            # 14/4, which is not a valid GQA shape).
            #
            # Overhead: padded_Q = ceil(KV/U)*U * (Q/KV). This is exact for
            # well-aligned shapes (0% overhead when U divides KV) but can be
            # substantial for uneven GQA/MQA: 28Q/7KV @ U=2 pads to 32Q (+14%),
            # 32Q/1KV @ U=8 pads to 256Q (8x). _maybe_warn_uaa_pad_ratio()
            # surfaces a one-shot warning once the blow-up crosses
            # _UAA_PAD_RATIO_WARN_THRESHOLD so callers can adjust
            # ulysses_degree if the extra work / VRAM is unacceptable.
            query_head_cnt = int(query.shape[2])
            kv_head_cnt = int(key.shape[2])
            if key.shape[2] != value.shape[2] or query_head_cnt % kv_head_cnt != 0:
                raise ValueError(
                    "Ulysses GQA requires equal K/V head counts and query heads "
                    f"to be a multiple of KV heads, got Q={query_head_cnt}, "
                    f"K={kv_head_cnt}, V={int(value.shape[2])}."
                )
            padded_kv_head_cnt = _ceil_div(kv_head_cnt, ulysses_world_size) * ulysses_world_size
            padded_query_head_cnt = padded_kv_head_cnt * (query_head_cnt // kv_head_cnt)
            _maybe_warn_uaa_pad_ratio(
                query_head_cnt,
                kv_head_cnt,
                padded_query_head_cnt,
                ulysses_world_size,
                "query",
            )

            query, orig_head_cnt = _ulysses_all_to_all_any_qkv(
                self._ulysses_pg,
                query,
                seq_lens=seq_lens,
                use_sync=self._use_sync,
                padded_head_cnt=padded_query_head_cnt,
            )
            key, _ = _ulysses_all_to_all_any_qkv(
                self._ulysses_pg,
                key,
                seq_lens=seq_lens,
                use_sync=self._use_sync,
                padded_head_cnt=padded_kv_head_cnt,
            )
            value, _ = _ulysses_all_to_all_any_qkv(
                self._ulysses_pg,
                value,
                seq_lens=seq_lens,
                use_sync=self._use_sync,
                padded_head_cnt=padded_kv_head_cnt,
            )
            if gate_compress is not None:
                # gate_compress mirrors the query layout, so it must follow the
                # same GQA-derived head padding as the query all-to-all.
                gate_compress, _ = _ulysses_all_to_all_any_qkv(
                    self._ulysses_pg,
                    gate_compress,
                    seq_lens=seq_lens,
                    use_sync=self._use_sync,
                    padded_head_cnt=padded_query_head_cnt,
                )
        else:
            # Strict mode: fail fast with actionable errors for head divisibility.
            for name, t in (("query", query), ("key", key), ("value", value)):
                head_cnt = int(t.shape[2])
                if head_cnt % ulysses_world_size != 0:
                    supported = _positive_divisors(head_cnt)
                    raise ValueError(
                        "Ulysses-SP strict mode requires head_cnt divisible by ulysses_degree. "
                        f"{name}_head_cnt={head_cnt}, ulysses_degree={ulysses_world_size}. "
                        f"Try ulysses_degree in {supported}, or set ulysses_mode='advanced_uaa'."
                    )

            # (bs, seq_len/P, head_cnt, head_size) -> (bs, seq_len, head_cnt/P, head_size)
            if _a2a_permute_enabled(
                self._ulysses_a2a_permute,
                self._scatter_idx,
                self._gather_idx,
                ulysses_world_size,
            ):
                from vllm_omni.diffusion.distributed.a2a_permute import ulysses_qkv_fwd

                group_name = self._ulysses_pg.group_name
                query = ulysses_qkv_fwd(query, group_name, ulysses_world_size)
                key = ulysses_qkv_fwd(key, group_name, ulysses_world_size)
                value = ulysses_qkv_fwd(value, group_name, ulysses_world_size)
                if gate_compress is not None:
                    gate_compress = ulysses_qkv_fwd(gate_compress, group_name, ulysses_world_size)
            else:
                query = SeqAllToAll4D.apply(
                    self._ulysses_pg, query, self._scatter_idx, self._gather_idx, self._use_sync
                )
                key = SeqAllToAll4D.apply(self._ulysses_pg, key, self._scatter_idx, self._gather_idx, self._use_sync)
                value = SeqAllToAll4D.apply(
                    self._ulysses_pg, value, self._scatter_idx, self._gather_idx, self._use_sync
                )
                if gate_compress is not None:
                    gate_compress = SeqAllToAll4D.apply(
                        self._ulysses_pg, gate_compress, self._scatter_idx, self._gather_idx, self._use_sync
                    )
            seq_lens = []
            local_seq_len = 0
            orig_head_cnt = 0

        if is_joint:
            # Concatenate joint query AFTER AllToAll
            # Image query is now (B, S, H/P, D). Joint query is (B, S_txt, H/P, D).
            # This is dimensionally consistent.
            if joint_strategy == "rear":
                query = torch.cat([query, joint_tensor_query], dim=1)
            else:
                query = torch.cat([joint_tensor_query, query], dim=1)

        # Check if Ring Attention is also active (Hybrid mode)
        # If Ring is active, we should NOT concatenate joint_key/value to k/v here.
        # Instead, they should remain in attn_metadata and be passed to the Ring kernel.
        use_ring = self._sp_group.ring_world_size > 1

        if is_joint and not use_ring:
            # Concatenate joint key/value after all-to-all ONLY for pure Ulysses (Local Attention).
            if joint_strategy == "front":
                key = torch.cat([joint_tensor_key, key], dim=1)
                value = torch.cat([joint_tensor_value, value], dim=1)
            else:  # "rear"
                key = torch.cat([key, joint_tensor_key], dim=1)
                value = torch.cat([value, joint_tensor_value], dim=1)

        if gate_compress is not None and attn_metadata is not None:
            # The VSA compression gate follows the same S<->H all-to-all as
            # Q/K/V so every rank sees the full packed sequence for its local
            # head shard, matching FastVideo's DistributedAttention_VSA.
            attn_metadata.extra["gate_compress"] = gate_compress

        ctx = _UlyssesCtx(
            name=self.name,
            ulysses_pg=self._ulysses_pg,
            scatter_idx=self._scatter_idx,
            gather_idx=self._gather_idx,
            use_sync=self._use_sync,
            joint_len=joint_len,
            joint_strategy=joint_strategy,
            use_uaa=(mode == "advanced_uaa"),
            uaa_seq_lens=tuple(int(x) for x in seq_lens) if mode == "advanced_uaa" else (),
            uaa_local_seq_len=int(local_seq_len) if mode == "advanced_uaa" else 0,
            orig_head_cnt=int(orig_head_cnt) if mode == "advanced_uaa" else 0,
            joint_orig_head_cnt=int(joint_orig_head_cnt) if mode == "advanced_uaa" else 0,
        )
        use_2d_mask = False
        if attn_metadata is not None:
            if attn_metadata.attn_mask is not None and attn_metadata.attn_mask.ndim == 2:
                use_2d_mask = True
            if attn_metadata.joint_attn_mask is not None and attn_metadata.joint_attn_mask.ndim == 2:
                use_2d_mask = True

        if attn_metadata is not None and use_2d_mask:
            if is_joint:
                if attn_metadata.joint_attn_mask is None and attn_metadata.attn_mask is None:
                    attn_metadata.attn_mask = None
                else:
                    if attn_metadata.attn_mask is None:
                        assert attn_metadata.joint_attn_mask is not None
                        attn_metadata.attn_mask = torch.ones(
                            [query.shape[0], query.shape[1] - attn_metadata.joint_attn_mask.shape[1]],
                            dtype=torch.bool,
                            device=query.device,
                        )
                    elif attn_metadata.joint_attn_mask is None:
                        attn_metadata.joint_attn_mask = torch.ones(
                            [query.shape[0], query.shape[1] - attn_metadata.attn_mask.shape[1]],
                            dtype=torch.bool,
                            device=query.device,
                        )
                    attn_metadata.attn_mask = (
                        torch.cat([attn_metadata.joint_attn_mask, attn_metadata.attn_mask], dim=1)
                        if joint_strategy == "front"
                        else torch.cat([attn_metadata.attn_mask, attn_metadata.joint_attn_mask], dim=1)
                    )

            if attn_metadata.attn_mask is not None:
                # the final attn_mask is ready, the length should be aligedn with query length
                assert attn_metadata.attn_mask.shape[1] == query.shape[1], (
                    f"attn_mask length: {attn_metadata.attn_mask.shape[1]} != query length: {query.shape[1]}"
                )
                attn_metadata.attn_mask = attn_metadata.attn_mask.bool().contiguous()
        return query, key, value, attn_metadata, ctx

    def post_attention(self, attn_output: torch.Tensor, ctx: ParallelAttentionContext | None) -> torch.Tensor:
        assert isinstance(ctx, _UlyssesCtx), f"Unexpected ctx type: {type(ctx)!r}"

        if ctx.joint_len > 0:
            joint_len = ctx.joint_len

            if ctx.joint_strategy == "front":
                output_joint = attn_output[:, :joint_len]
                output_img = attn_output[:, joint_len:]
            else:
                output_img = attn_output[:, :-joint_len]
                output_joint = attn_output[:, -joint_len:]

            # 1. Process Image part: Standard Ulysses Reverse (AllToAll)
            # (bs, seq_len, head_cnt/P, head_size) -> (bs, seq_len/P, head_cnt, head_size)
            # SeqAllToAll4D handles: Scatter gather_idx, Gather scatter_idx.
            # Forward: Scatter 2 (H), Gather 1 (S).
            # Reverse: Scatter 1 (S), Gather 2 (H).
            if ctx.use_uaa:
                output_img = _ulysses_all_to_all_any_o(
                    ctx.ulysses_pg,
                    output_img,
                    seq_lens=list(ctx.uaa_seq_lens),
                    local_seq_len=ctx.uaa_local_seq_len,
                    orig_head_cnt=ctx.orig_head_cnt,
                    use_sync=ctx.use_sync,
                )
            elif _a2a_permute_enabled(
                self._ulysses_a2a_permute,
                ctx.scatter_idx,
                ctx.gather_idx,
                dist.get_world_size(ctx.ulysses_pg),
            ):
                from vllm_omni.diffusion.distributed.a2a_permute import ulysses_o_rev

                output_img = ulysses_o_rev(
                    output_img,
                    ctx.ulysses_pg.group_name,
                    dist.get_world_size(ctx.ulysses_pg),
                )
            else:
                output_img = SeqAllToAll4D.apply(
                    ctx.ulysses_pg, output_img, ctx.gather_idx, ctx.scatter_idx, ctx.use_sync
                )

            # 2. Process Joint part: AllGather on Heads
            # Input: (B, JointLen, H/P, D). Output: (B, JointLen, H, D).
            # AllGather along dim 2.
            # Ensure tensor is contiguous for all_gather (slicing may create non-contiguous views)
            output_joint = output_joint.contiguous()
            gathered_joint = [torch.zeros_like(output_joint) for _ in range(dist.get_world_size(ctx.ulysses_pg))]
            dist.all_gather(gathered_joint, output_joint, group=ctx.ulysses_pg)
            output_joint = torch.cat(gathered_joint, dim=2)
            if ctx.use_uaa and ctx.joint_orig_head_cnt > 0 and output_joint.shape[2] != ctx.joint_orig_head_cnt:
                output_joint = output_joint[:, :, : ctx.joint_orig_head_cnt, :].contiguous()

            # 3. Recombine
            if ctx.joint_strategy == "front":
                return torch.cat([output_joint, output_img], dim=1)
            else:
                return torch.cat([output_img, output_joint], dim=1)

        # Standard Ulysses Reverse
        if ctx.use_uaa:
            return _ulysses_all_to_all_any_o(
                ctx.ulysses_pg,
                attn_output,
                seq_lens=list(ctx.uaa_seq_lens),
                local_seq_len=ctx.uaa_local_seq_len,
                orig_head_cnt=ctx.orig_head_cnt,
                use_sync=ctx.use_sync,
            )
        if _a2a_permute_enabled(
            self._ulysses_a2a_permute,
            ctx.scatter_idx,
            ctx.gather_idx,
            dist.get_world_size(ctx.ulysses_pg),
        ):
            from vllm_omni.diffusion.distributed.a2a_permute import ulysses_o_rev

            return ulysses_o_rev(
                attn_output,
                ctx.ulysses_pg.group_name,
                dist.get_world_size(ctx.ulysses_pg),
            )
        return SeqAllToAll4D.apply(ctx.ulysses_pg, attn_output, ctx.gather_idx, ctx.scatter_idx, ctx.use_sync)
