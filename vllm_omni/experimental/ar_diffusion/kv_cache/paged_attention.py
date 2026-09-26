# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Paged self-attention helpers for AR-Diffusion KV reuse."""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any, ClassVar, NamedTuple

import torch

from vllm_omni.experimental.ar_diffusion.kv_cache.config import contiguous_kv_gather_enabled
from vllm_omni.experimental.ar_diffusion.kv_cache.paged import compute_slot_mapping

_LAYER_IDX_TENSORS: dict[int, torch.Tensor] = {}


def _layer_idx_tensor(layer_idx: int) -> torch.Tensor:
    t = _LAYER_IDX_TENSORS.get(layer_idx)
    if t is None:
        t = torch.tensor(layer_idx, dtype=torch.int64)
        _LAYER_IDX_TENSORS[layer_idx] = t
    return t


def _to_device_async(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Host->device copy that does not stall the CPU.

    ``tensor.to(device)`` from pageable memory is synchronous: the CPU blocks until
    every kernel already queued on the stream has finished. Pinned + non_blocking
    keeps the CPU running ahead (the caching host allocator keeps the pinned
    source alive until the copy's stream event completes).
    """
    if torch.device(device).type not in ("cuda", "npu") or t.device.type != "cpu":
        return t.to(device=device)
    return t.pin_memory().to(device=device, non_blocking=True)


class ARDiffusionPagedLayerInputs(NamedTuple):
    """Compiled-region payload for one layer's paged self-attention.

    A NamedTuple of plain tensors + ints so ``torch.compile`` treats every field
    as a pytree graph input (no object-attribute guards, no recompiles when only
    tensor *values* change). All layers of one KV branch forward share the same
    metadata tensor objects, built once by ``prepare()``.

    ``layer_idx`` is a 0-dim CPU tensor, NOT a python int: all 40 DiT blocks
    share one compiled code object, and an int here becomes a per-layer dynamo
    value guard (``layer_idx == k``) — 40 cache variants that blow the
    recompile limit. A tensor input guards on shape/dtype only, so one graph
    serves every layer.
    """

    layer_idx: torch.Tensor
    key_pool: torch.Tensor
    value_pool: torch.Tensor
    block_size: int
    seq_len: int
    video_slots: torch.Tensor
    action_slots: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    max_query_len: int
    max_seq_len: int
    # Contiguous K/V staging for this layer, when reuse_history_staging is on. ``reuse_history`` is a host
    # bool decided once per forward (``_prepare_history_staging``): it selects one of two compiled variants.
    stage_key: torch.Tensor | None = None
    stage_value: torch.Tensor | None = None
    reuse_history: bool = False
    # Index of the current chunk's first block within the staged window: right after the visible history,
    # which is not the end of the padded block table (that carries at least one action-capacity block).
    stage_first_block: int = 0


@dataclass
class ARDiffusionPagedForwardContext:
    """Mutable KV-branch state shared by all layer contexts in one forward."""

    kv_cache: Any
    adapter: Any
    kv_branch: str
    history_block_ids: list[int]
    seq_len: int
    commit_current: bool
    max_video_tokens: int
    current_video_block_ids: list[int] = field(default_factory=list)
    current_video_slot_mapping: torch.Tensor | None = None
    action_scratch_block_ids: list[int] = field(default_factory=list)
    action_slot_mapping: torch.Tensor | None = None
    query_len: int = 0
    kv_len: int = 0
    _allocated_video: bool = False
    _committed: bool = False
    _action_len: int = 0
    # Set once by prepare(); shared by all layers of the KV branch forward.
    block_table: torch.Tensor | None = None
    query_start_loc: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    max_query_len: int = 0
    max_seq_len: int = 0
    _prepared: bool = False

    @property
    def block_size(self) -> int:
        return int(self.kv_cache.block_size)

    @property
    def num_current_video_blocks(self) -> int:
        if self.seq_len % self.block_size != 0:
            raise AssertionError(
                "AR-Diffusion paged attention expects frame-aligned seq_len "
                f"(multiple of block_size={self.block_size}), got {self.seq_len}"
            )
        return self.seq_len // self.block_size

    def ensure_video_slots(self, device: torch.device) -> None:
        """Allocate/write targets for the current video tokens, once per KV branch."""
        if self._allocated_video:
            return

        n_blocks = self.num_current_video_blocks
        if self.commit_current:
            start = int(self.adapter.num_computed_tokens)
            self.kv_cache.allocate_token_slots(self.adapter, self.seq_len)
            table = self.kv_cache.block_table(self.adapter)
            start_block = start // self.block_size
            self.current_video_block_ids = [int(b) for b in table[start_block : start_block + n_blocks]]
            positions = torch.arange(start, start + self.seq_len, dtype=torch.long)
            self.current_video_slot_mapping = _to_device_async(
                compute_slot_mapping(table, positions, self.block_size), device
            )
        else:
            self.current_video_block_ids = self.kv_cache.scratch_block_ids(self.kv_branch, 0, n_blocks)
            positions = torch.arange(self.seq_len, dtype=torch.long)
            self.current_video_slot_mapping = _to_device_async(
                compute_slot_mapping(self.current_video_block_ids, positions, self.block_size), device
            )
        self._allocated_video = True

    def ensure_action_slots(self, action_len: int, device: torch.device) -> None:
        """Reserve scratch slots for action/state K/V, if present."""
        if action_len <= 0:
            self.action_scratch_block_ids = []
            self.action_slot_mapping = torch.empty(0, dtype=torch.long, device=device)
            self._action_len = 0
            return

        self.ensure_video_slots(device)
        if self.action_slot_mapping is not None and self._action_len == action_len:
            return

        action_blocks = (action_len + self.block_size - 1) // self.block_size
        scratch_offset = 0 if self.commit_current else len(self.current_video_block_ids)
        self.action_scratch_block_ids = self.kv_cache.scratch_block_ids(
            self.kv_branch,
            scratch_offset,
            action_blocks,
        )
        positions = torch.arange(action_len, dtype=torch.long)
        self.action_slot_mapping = _to_device_async(
            compute_slot_mapping(self.action_scratch_block_ids, positions, self.block_size), device
        )
        self._action_len = action_len

    def video_block_table(self, device: torch.device) -> tuple[list[int], int]:
        self.ensure_video_slots(device)
        if self.max_video_tokens % self.block_size != 0:
            raise AssertionError(
                "AR-Diffusion paged attention requires max_video_tokens to be block-aligned, "
                f"got max_video_tokens={self.max_video_tokens}, block_size={self.block_size}"
            )
        all_video_blocks = self.history_block_ids + self.current_video_block_ids
        max_video_blocks = self.max_video_tokens // self.block_size
        sink_blocks = int(self.kv_cache.spec.sink_chunks)
        if len(all_video_blocks) <= max_video_blocks:
            visible_video_blocks = all_video_blocks
        else:
            tail_blocks = max_video_blocks - sink_blocks
            visible_video_blocks = all_video_blocks[:sink_blocks]
            if tail_blocks:
                visible_video_blocks += all_video_blocks[-tail_blocks:]
        video_len = len(visible_video_blocks) * self.block_size
        return visible_video_blocks, video_len

    def build_block_table(
        self,
        *,
        action_len: int,
        query_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Build FlashAttention block-table metadata for one self-attn call.

        The block table is tail-padded to a fixed width and ``max_seq_len`` is a
        constant upper bound, so across window growth only tensor *values* change
        — tensor shapes and the int consts stay stable for ``torch.compile``
        (and, later, CUDA-graph capture). The kernel only dereferences the first
        ``ceil(seq_lens/block_size)`` entries, so padding is never read.
        """
        video_blocks, video_len = self.video_block_table(device)
        self.ensure_action_slots(action_len, device)
        action_blocks = self.action_scratch_block_ids if action_len > 0 else []
        block_ids = video_blocks + action_blocks
        if not block_ids:
            raise RuntimeError("AR-Diffusion paged attention needs at least current video KV blocks")

        # Fixed capacity: full visible video window + one action-capacity block.
        action_capacity_blocks = max(1, (action_len + self.block_size - 1) // self.block_size)
        width = max(self.max_video_tokens // self.block_size + action_capacity_blocks, len(block_ids))
        padded = block_ids + [0] * (width - len(block_ids))

        self.query_len = int(query_len)
        self.kv_len = int(video_len + action_len)
        max_seq_len = int(self.max_video_tokens + action_capacity_blocks * self.block_size)
        # Built on the host once per AR block; the copies go through the same
        # pinned, non-blocking path as the slot mappings above so the CPU does
        # not wait for the stream to drain three times before the first layer.
        block_table = _to_device_async(torch.tensor([padded], dtype=torch.int32), device)
        query_start_loc = _to_device_async(torch.tensor([0, self.query_len], dtype=torch.int32), device)
        seq_lens = _to_device_async(torch.tensor([self.kv_len], dtype=torch.int32), device)
        return block_table, query_start_loc, seq_lens, self.query_len, max_seq_len

    def prepare(self, device: torch.device, action_len: int, query_len: int) -> None:
        """Host-side, once-per-KV-branch setup (called OUTSIDE torch.compile).

        Allocates the current video/action slots (still lazy: only the KV branch a
        CFG-parallel rank actually runs reaches its ``_forward_blocks``), builds
        the padded block-table metadata ONCE for all layers, and publishes the
        pool registry for the fused custom op. The compiled per-layer code then
        only consumes prebuilt tensors via ``ARDiffusionPagedLayerInputs``.
        """
        if getattr(self, "_prepared", False):
            return
        self.ensure_video_slots(device)
        (
            self.block_table,
            self.query_start_loc,
            self.seq_lens,
            self.max_query_len,
            self.max_seq_len,
        ) = self.build_block_table(action_len=action_len, query_len=query_len, device=device)
        if self.action_slot_mapping is None:
            self.action_slot_mapping = torch.empty(0, dtype=torch.long, device=device)
        self._prepare_history_staging(action_len)
        self._prepared = True

    def _prepare_history_staging(self, action_len: int) -> None:
        """Decide whether this forward can keep the history already staged, and from where.

        The staged window is only reusable when nothing about it moved: the same session adapter, the same
        visible history blocks, the same sequence bounds. Any difference -- a new AR block, a slid window, a
        different request -- and the buffer is rebuilt from the pools.

        The decision is a host bool, deliberately. It has to be, for the gather to actually be skipped: a
        device flag could only select between two results already computed. It is the same value for all
        layers of a forward, so it costs two compiled variants in total rather than one per layer.
        """
        self.staging_enabled = False
        self.reuse_history = False
        self.stage_first_block = 0
        # The manager allocates the staging pairs only when reuse_history_staging is on and the contiguous
        # gather path -- their only consumer -- is switched on.
        if not self.kv_cache.history_staging:
            return
        assert self.block_table is not None
        if action_len or self.block_table.shape[0] != 1:
            # Action tokens live in scratch blocks outside the video window; staging them is not modelled.
            return
        capacity = int(self.kv_cache.history_staging[0][0].shape[0])
        if int(self.max_seq_len) > capacity:
            raise RuntimeError(
                f"history staging buffers hold {capacity} tokens but this forward stages {int(self.max_seq_len)}"
            )
        state = self.kv_cache.history_staging_state
        signature = (
            int(self.adapter.num_computed_tokens),
            tuple(self.history_block_ids),
            int(self.max_seq_len),
            int(self.seq_len),
        )
        same_session = state.adapter is not None and state.adapter() is self.adapter
        reuse = same_session and state.signature == signature
        state.adapter = weakref.ref(self.adapter)
        state.signature = signature
        self.staging_enabled = True
        self.reuse_history = bool(reuse)
        # ``kv_len`` is the visible video window here (no action tokens on this path): the current chunk's
        # blocks are its last ``num_current_video_blocks`` entries, and the table's trailing padding blocks
        # come after them.
        self.stage_first_block = self.kv_len // self.block_size - self.num_current_video_blocks

    def history_staging(self, layer_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return the manager-owned tensors marked static for CUDA Graph input mutation.

        Fresh prefix views lose the bases' static-address annotation. Narrow the
        window inside the attention custom op instead of at the compiled boundary.
        """
        if not self.staging_enabled:
            # NPU eager dispatch of the fused write+attend custom op rejects
            # None for a tensor argument declared mutable, so hand back
            # zero-size buffers there; the op treats an empty stage buffer
            # like None (fresh allocation). Other devices keep the upstream
            # None contract.
            k_pool = self.kv_cache._k_pools[layer_idx]
            if k_pool.device.type == "npu":
                v_pool = self.kv_cache._v_pools[layer_idx]
                return (
                    torch.empty(0, device=k_pool.device, dtype=k_pool.dtype),
                    torch.empty(0, device=v_pool.device, dtype=v_pool.dtype),
                )
            return None, None
        return self.kv_cache.history_staging[layer_idx]

    def layer_inputs(self, layer_idx: int) -> ARDiffusionPagedLayerInputs:
        if not getattr(self, "_prepared", False):
            raise RuntimeError("ARDiffusionPagedForwardContext.layer_inputs() before prepare()")
        key_pool = self.kv_cache._k_pools[layer_idx]
        value_pool = self.kv_cache._v_pools[layer_idx]
        stage_key, stage_value = self.history_staging(layer_idx)
        return ARDiffusionPagedLayerInputs(
            layer_idx=_layer_idx_tensor(layer_idx),
            key_pool=key_pool,
            value_pool=value_pool,
            block_size=int(self.kv_cache.block_size),
            seq_len=int(self.seq_len),
            video_slots=self.current_video_slot_mapping,
            action_slots=self.action_slot_mapping,
            block_table=self.block_table,
            query_start_loc=self.query_start_loc,
            seq_lens=self.seq_lens,
            max_query_len=int(self.max_query_len),
            max_seq_len=int(self.max_seq_len),
            stage_key=stage_key,
            stage_value=stage_value,
            reuse_history=self.reuse_history,
            stage_first_block=int(self.stage_first_block),
        )

    def mark_committed(self) -> None:
        self._committed = True


@dataclass
class ARDiffusionPagedLayerContext:
    """Layer-specific handle passed through a model's ``kv_cache`` slot."""

    is_ar_diffusion_paged_context: ClassVar[bool] = True
    layer_idx: int
    forward_ctx: ARDiffusionPagedForwardContext

    @property
    def kv_cache(self):
        return self.forward_ctx.kv_cache

    @property
    def adapter(self):
        return self.forward_ctx.adapter

    @property
    def kv_branch(self) -> str:
        return self.forward_ctx.kv_branch

    @property
    def history_block_ids(self) -> list[int]:
        return self.forward_ctx.history_block_ids

    @property
    def current_video_block_ids(self) -> list[int]:
        return self.forward_ctx.current_video_block_ids

    @property
    def current_video_slot_mapping(self) -> torch.Tensor | None:
        return self.forward_ctx.current_video_slot_mapping

    @property
    def action_scratch_block_ids(self) -> list[int]:
        return self.forward_ctx.action_scratch_block_ids

    @property
    def action_slot_mapping(self) -> torch.Tensor | None:
        return self.forward_ctx.action_slot_mapping

    @property
    def seq_len(self) -> int:
        return self.forward_ctx.seq_len

    @property
    def query_len(self) -> int:
        return self.forward_ctx.query_len

    @property
    def kv_len(self) -> int:
        return self.forward_ctx.kv_len

    @property
    def commit_current(self) -> bool:
        return self.forward_ctx.commit_current

    def to_layer_inputs(self) -> ARDiffusionPagedLayerInputs:
        """Compiled-region payload; requires ``forward_ctx.prepare()`` first."""
        return self.forward_ctx.layer_inputs(self.layer_idx)


def is_ar_diffusion_paged_context(value: object) -> bool:
    return isinstance(value, ARDiffusionPagedLayerContext)


def _reference_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float,
    *,
    causal: bool,
) -> torch.Tensor:
    if causal:
        raise NotImplementedError("AR-Diffusion paged self-attention uses causal=False")
    outs: list[torch.Tensor] = []
    block_size = key_cache.shape[1]
    for i in range(seq_lens.shape[0]):
        q_start = int(query_start_loc[i].item())
        q_end = int(query_start_loc[i + 1].item())
        kv_len = int(seq_lens[i].item())
        q = query[q_start:q_end]
        positions = torch.arange(kv_len, device=query.device)
        logical_blocks = torch.div(positions, block_size, rounding_mode="floor")
        offsets = positions % block_size
        physical_blocks = block_table[i, logical_blocks].long()
        k = key_cache[physical_blocks, offsets]
        v = value_cache[physical_blocks, offsets]
        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * float(softmax_scale)
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        outs.append(torch.einsum("hqk,khd->qhd", probs, v))
    return torch.cat(outs, dim=0)


_FA_VERSION_BY_HEAD_SIZE: dict[int, int] = {}


def _stage_window(stage_key, stage_value, key_cache, value_cache, block_ids, n_blocks, block_size, *, first_block):
    """Gather visible blocks from ``first_block`` onward into the staging buffers.

    ``first_block`` is 0 for a full restage and the index of the current chunk's first block when the
    history is already staged. Skipping the leading blocks is the whole point: they were gathered by an
    earlier forward of the same AR block and cannot have changed since.
    """
    ids = block_ids[first_block:]
    # Write directly into the caller-owned window, avoiding an intermediate
    # gathered tensor and its copy on every layer and denoising step.
    torch.index_select(key_cache, 0, ids, out=stage_key.view(n_blocks, block_size, *key_cache.shape[2:])[first_block:])
    torch.index_select(
        value_cache, 0, ids, out=stage_value.view(n_blocks, block_size, *value_cache.shape[2:])[first_block:]
    )


def _resolve_fa_version(head_size: int) -> int:
    # get_flash_attn_version -> current_platform.get_device_capability() is not
    # dynamo-traceable, and the answer is fixed per head size for the process.
    version = _FA_VERSION_BY_HEAD_SIZE.get(head_size)
    if version is None:
        try:
            from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

            version = int(get_flash_attn_version(requires_alibi=False, head_size=head_size) or 2)
        except Exception:
            version = 2
        _FA_VERSION_BY_HEAD_SIZE[head_size] = version
    return version


def _rocm_flash_attn_varlen_func():
    """Resolve a ROCm varlen kernel, preferring AITER when available.

    The caller gathers paged KV into packed tensors before invoking this
    function, avoiding AITER releases whose ``block_table`` kernel is broken.
    """
    try:
        from aiter import flash_attn_varlen_func

        return flash_attn_varlen_func
    except ImportError:
        from flash_attn import flash_attn_varlen_func

        return flash_attn_varlen_func


def ar_diffusion_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    softmax_scale: float,
    causal: bool = False,
    stage_key: torch.Tensor | None = None,
    stage_value: torch.Tensor | None = None,
    reuse_history: bool = False,
    current_tokens: int = 0,
    stage_first_block: int = 0,
) -> torch.Tensor:
    """Run non-causal paged attention over a vLLM block table.

    ``query`` may be ``(B, L, H, D)`` or already flattened as ``(T, H, D)``.
    ``key_cache`` / ``value_cache`` are ``(num_blocks, block_size, H, D)``.
    """
    batched = query.dim() == 4
    if batched:
        batch, q_len = query.shape[:2]
        query_flat = query.reshape(batch * q_len, *query.shape[2:])
    else:
        query_flat = query

    if query_flat.is_npu:
        # NPU: gather the visible blocks on device, then run the fused
        # npu_fusion_attention kernel over the packed (contiguous) window.
        # Mirrors the ROCm gather path; the reference implementation below is
        # the correctness oracle but pays per-block gathers and fp32 einsum.
        import torch_npu

        positions = torch.arange(int(max_seq_len), device=query_flat.device)
        block_size = key_cache.shape[1]
        logical_blocks = torch.div(positions, block_size, rounding_mode="floor")
        offsets = positions % block_size
        physical_blocks = block_table[0, logical_blocks].long()
        gathered_k = key_cache[physical_blocks, offsets]
        gathered_v = value_cache[physical_blocks, offsets]
        # Flatten to (max_seq_len, H, D) so the kv-length mask packs rows.
        gathered_k = gathered_k.reshape(int(max_seq_len), *key_cache.shape[2:])
        gathered_v = gathered_v.reshape(int(max_seq_len), *value_cache.shape[2:])
        kv_len = int(seq_lens[0].item())
        packed_k = gathered_k[:kv_len]
        packed_v = gathered_v[:kv_len]
        # TND layout: per-sequence lengths and their cu_seqlens prefixes.
        q_lens_per_seq = query_start_loc[1:] - query_start_loc[:-1]
        actual_seq_q = q_lens_per_seq.to(torch.int32).cpu().tolist()
        actual_seq_kv = seq_lens.to(torch.int32).cpu().tolist()
        out = torch_npu.npu_fusion_attention(
            query_flat.contiguous(),
            packed_k.contiguous(),
            packed_v.contiguous(),
            head_num=query_flat.shape[1],
            input_layout="TND",
            actual_seq_qlen=actual_seq_q,
            actual_seq_kvlen=actual_seq_kv,
            scale=float(softmax_scale),
            keep_prob=1.0,
        )[0]
        if batched:
            return out.reshape(query.shape)
        return out

    if not query_flat.is_cuda:
        out = _reference_paged_attention(
            query_flat,
            key_cache,
            value_cache,
            block_table,
            query_start_loc,
            seq_lens,
            softmax_scale,
            causal=causal,
        )
    elif torch.version.hip is not None:
        # vllm.vllm_flash_attn contains CUDA-only extensions. ROCm's AITER and
        # upstream flash-attn expose the standard cu_seqlens_k API instead of
        # vLLM's seqused_k/fa_version API. The ROCm flash-attn paged kernel also
        # requires 128-token blocks, while AR-Diffusion uses frame-aligned
        # 16-token blocks, so gather the visible blocks on-device first.
        rocm_flash_attn_varlen_func = _rocm_flash_attn_varlen_func()
        cu_seqlens_k = torch.cat([seq_lens.new_zeros(1), torch.cumsum(seq_lens, dim=0, dtype=torch.int32)])
        positions = torch.arange(int(max_seq_len), device=query_flat.device)
        logical_blocks = torch.div(positions, key_cache.shape[1], rounding_mode="floor")
        offsets = positions % key_cache.shape[1]
        physical_blocks = block_table[:, logical_blocks].long()
        gathered_k = key_cache[physical_blocks, offsets]
        gathered_v = value_cache[physical_blocks, offsets]
        valid = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
        packed_k = gathered_k[valid]
        packed_v = gathered_v[valid]
        out = rocm_flash_attn_varlen_func(
            q=query_flat,
            k=packed_k,
            v=packed_v,
            cu_seqlens_q=query_start_loc,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=int(max_query_len),
            max_seqlen_k=int(max_seq_len),
            softmax_scale=float(softmax_scale),
            causal=causal,
        )
    elif contiguous_kv_gather_enabled() and block_table.shape[0] == 1:
        # Experimental (VLLM_OMNI_AR_DIFFUSION_KV_GATHER=1): FA3's paged-KV path with the
        # frame-sized block (1560 tokens, not a multiple of the FA3 K/V tile)
        # runs ~15-17% slower than the same kernel on contiguous K/V. Gather the
        # visible blocks into a contiguous buffer once per layer and attend
        # without a block table; ``seqused_k`` masks the tail-padding block.
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        fa_version = _resolve_fa_version(query_flat.shape[-1])
        block_size = key_cache.shape[1]
        n_blocks = int(max_seq_len) // block_size
        if n_blocks * block_size != int(max_seq_len):
            raise ValueError("the contiguous K/V gather path requires max_seq_len to be block-aligned")
        block_ids = block_table[0, :n_blocks].to(torch.long)
        if stage_key is None or stage_value is None or stage_key.numel() == 0:
            # Fresh allocations on purpose: a module-level cached buffer that is first
            # allocated inside a CUDA-graph-trees warm-up run lives in the graph pool
            # untracked ("tensor(s) in the cudagraph pool not tracked as outputs").
            k_buf = key_cache.index_select(0, block_ids)
            v_buf = value_cache.index_select(0, block_ids)
            k_flat = k_buf.view(n_blocks * block_size, *key_cache.shape[2:])
            v_flat = v_buf.view(n_blocks * block_size, *value_cache.shape[2:])
        else:
            # Caller-owned staging, allocated outside any capture and marked static, so the buffer this
            # writes into is the same address every forward and the graph pool never owns it.
            #
            # Only the current chunk moved. The block table is the visible history followed by the
            # current chunk's blocks and then padding (at least the action-capacity block, plus unused
            # window capacity while the window is still growing), so the rows this forward changed start
            # at ``stage_first_block`` -- the caller's count of visible history blocks -- and NOT at
            # ``n_blocks - current_blocks``, which would refresh padding and leave the current K/V stale.
            # Everything before them was staged by an earlier forward of the same AR block and is still
            # byte-for-byte what a full gather would produce. When the history did move -- new block,
            # slid window, different session -- reuse_history is 0 and the whole window is re-gathered,
            # which is the same work the unstaged path always does.
            # Keep the marked base tensors as graph inputs, but only stage the
            # active prefix when the manager allocated a larger capacity.
            k_flat = stage_key[:max_seq_len]
            v_flat = stage_value[:max_seq_len]
            if reuse_history:
                # The caller's metadata must describe a block-aligned current chunk inside the staged
                # window; anything else is a wrong offset, not a reason to quietly restage everything.
                current_blocks = current_tokens // block_size
                assert current_tokens > 0 and current_blocks * block_size == current_tokens, (
                    f"staged reuse needs a block-aligned current chunk, got {current_tokens} tokens"
                )
                assert 0 <= stage_first_block and stage_first_block + current_blocks <= n_blocks, (
                    f"stage_first_block={stage_first_block} + {current_blocks} blocks "
                    f"exceeds the {n_blocks}-block window"
                )
            _stage_window(
                k_flat,
                v_flat,
                key_cache,
                value_cache,
                block_ids,
                n_blocks,
                block_size,
                first_block=stage_first_block if reuse_history else 0,
            )
        out = torch.empty_like(query_flat)
        # Varlen K/V: cu_seqlens_k = [0, kv_len] built on device from seq_lens
        # (no host sync). Rows past kv_len (the tail-padding block) are never
        # read, exactly like the ROCm gather path above.
        cu_seqlens_k = torch.cat((seq_lens.new_zeros(1), seq_lens.to(torch.int32)))
        flash_attn_varlen_func(
            q=query_flat,
            k=k_flat,
            v=v_flat,
            out=out,
            cu_seqlens_q=query_start_loc,
            max_seqlen_q=int(max_query_len),
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=int(max_seq_len),
            softmax_scale=float(softmax_scale),
            causal=causal,
            fa_version=fa_version,
        )
    else:
        from vllm.vllm_flash_attn import flash_attn_varlen_func as paged_flash_attn_varlen_func

        fa_version = _resolve_fa_version(query_flat.shape[-1])

        out = torch.empty_like(query_flat)
        paged_flash_attn_varlen_func(
            q=query_flat,
            k=key_cache,
            v=value_cache,
            out=out,
            cu_seqlens_q=query_start_loc,
            max_seqlen_q=int(max_query_len),
            seqused_k=seq_lens,
            max_seqlen_k=int(max_seq_len),
            softmax_scale=float(softmax_scale),
            causal=causal,
            block_table=block_table,
            fa_version=fa_version,
        )

    if batched:
        return out.reshape(query.shape)
    return out


# ── Fused write+attend custom op (torch.compile-safe) ──────────────────────
#
# One opaque op per layer keeps the compiled DiT block fullgraph: dynamo treats
# it as a single graph node (no eager island, no graph breaks), and the K/V slot
# writes happen inside the op so write→read ordering with the block-table kernel
# is internal. The flat pools are explicit mutable inputs: Inductor/CUDA Graph
# must track their storage lifetime instead of observing an undeclared mutation
# through a process-global registry.
def _paged_write_attn_impl(
    query: torch.Tensor,
    k_curr: torch.Tensor,
    v_curr: torch.Tensor,
    k_act: torch.Tensor | None,
    v_act: torch.Tensor | None,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    block_size: int,
    video_slots: torch.Tensor,
    action_slots: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    softmax_scale: float,
    stage_key: torch.Tensor | None = None,
    stage_value: torch.Tensor | None = None,
    reuse_history: bool = False,
    stage_first_block: int = 0,
) -> torch.Tensor:
    key_pool[video_slots] = k_curr.to(key_pool.dtype)
    value_pool[video_slots] = v_curr.to(value_pool.dtype)
    if k_act is not None and v_act is not None and k_act.shape[0] > 0:
        key_pool[action_slots] = k_act.to(key_pool.dtype)
        value_pool[action_slots] = v_act.to(value_pool.dtype)
    key_cache = key_pool.unflatten(0, (-1, block_size))
    value_cache = value_pool.unflatten(0, (-1, block_size))
    return ar_diffusion_paged_attention(
        query,
        key_cache,
        value_cache,
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        softmax_scale=softmax_scale,
        causal=False,
        stage_key=stage_key,
        stage_value=stage_value,
        reuse_history=reuse_history,
        current_tokens=int(k_curr.shape[0]),
        stage_first_block=stage_first_block,
    )


# hasattr guard keeps registration idempotent across test re-imports that pop
# the module from sys.modules (same as sage_attn3.py).
if not hasattr(torch.ops.vllm_omni, "ar_diffusion_paged_write_attn"):
    # Keep staging arguments required even when their value is None. The dispatcher
    # strips trailing defaults, and older PyTorch mutation handlers index the
    # positional arguments without restoring them before bumping version counters.
    @torch.library.custom_op(
        "vllm_omni::ar_diffusion_paged_write_attn",
        mutates_args=("key_pool", "value_pool", "stage_key", "stage_value"),
    )
    def _paged_write_attn_op(
        query: torch.Tensor,
        k_curr: torch.Tensor,
        v_curr: torch.Tensor,
        k_act: torch.Tensor | None,
        v_act: torch.Tensor | None,
        key_pool: torch.Tensor,
        value_pool: torch.Tensor,
        block_size: int,
        video_slots: torch.Tensor,
        action_slots: torch.Tensor,
        block_table: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        max_query_len: int,
        max_seq_len: int,
        softmax_scale: float,
        stage_key: torch.Tensor | None,
        stage_value: torch.Tensor | None,
        reuse_history: bool,
        stage_first_block: int,
    ) -> torch.Tensor:
        return _paged_write_attn_impl(
            query,
            k_curr,
            v_curr,
            k_act,
            v_act,
            key_pool,
            value_pool,
            block_size,
            video_slots,
            action_slots,
            block_table,
            query_start_loc,
            seq_lens,
            max_query_len,
            max_seq_len,
            softmax_scale,
            stage_key,
            stage_value,
            reuse_history,
            stage_first_block,
        )

    @_paged_write_attn_op.register_fake
    def _(
        query,
        k_curr,
        v_curr,
        k_act,
        v_act,
        key_pool,
        value_pool,
        block_size,
        video_slots,
        action_slots,
        block_table,
        query_start_loc,
        seq_lens,
        max_query_len,
        max_seq_len,
        softmax_scale,
        stage_key=None,
        stage_value=None,
        reuse_history=False,
        stage_first_block=0,
    ):
        return torch.empty_like(query)


def paged_write_attn(
    inputs: ARDiffusionPagedLayerInputs, query, k_curr, v_curr, k_act, v_act, softmax_scale: float
) -> torch.Tensor:
    """Model-facing entry: routes through the custom op (traceable in fullgraph)."""
    return torch.ops.vllm_omni.ar_diffusion_paged_write_attn(
        query,
        k_curr,
        v_curr,
        k_act,
        v_act,
        inputs.key_pool,
        inputs.value_pool,
        inputs.block_size,
        inputs.video_slots,
        inputs.action_slots,
        inputs.block_table,
        inputs.query_start_loc,
        inputs.seq_lens,
        inputs.max_query_len,
        inputs.max_seq_len,
        softmax_scale,
        inputs.stage_key,
        inputs.stage_value,
        inputs.reuse_history,
        inputs.stage_first_block,
    )
