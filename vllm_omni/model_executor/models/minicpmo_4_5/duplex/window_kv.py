# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native paged-KV window for MiniCPM-o 4.5 duplex Stage 0.

``window_plan.py`` holds the window arithmetic in plain integers; this file
holds the three things that need an engine: the KV-cache spec/manager pair the
scheduler resolves, the layer change that produces that spec, and the in-place
re-RoPE that follows a trim. The scheduler helper also keeps a barge-in's
window for the session's next Stage-0 request (see ``snapshot_window``).

One mechanism, matching the reference implementation
----------------------------------------------------
A trim and a renumber are the same event here, which is what MiniCPM's HF
implementation does: cut the cached K/V, re-RoPE the survivors onto dense
positions. #7631 gets the same observable state from the other direction, by
rebuilding the prompt and letting the model recompute every retained row -- a
full forward pass over the window, per roll.

Here the cut is a block-table edit and the re-RoPE is one rotation per retained
key, because :class:`~vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_plan.PositionReanchor`
guarantees the shift is one uniform translation of a whole number of pages. Two
consequences are worth stating plainly:

* No physical row moves. After ``del table[sink:sink + n]``, logical index ``j``
  resolves to the page that ``j + n`` named before the deletion -- which is the
  page the row is already in -- and a block-aligned shift leaves ``pos %
  block_size`` alone. So the rotation reads and writes the same slots, and V is
  not touched at all, because only K carries a RoPE phase.
* The window needs no mask, because there is nothing to skip: the retained span
  is contiguous and plain causal attention over it is what the model was trained
  to do. See ``window_plan.py`` for why a *masked* window cannot express a duplex
  session, and why R-SWA -- the one upstream type that can -- is not usable on a
  FlashAttention-3 Stage 0.

What the sliding-window spec is then for is sizing. ``SlidingWindowSpec`` is the
only attention type in vLLM whose ``max_admission_blocks_per_request`` bounds a
request at its window rather than at ``max_model_len``, which is what lets
``max_sessions`` concurrent streams share a Stage 0 that advertises a 40k-token
context. Its own eviction is overridden to stay inert: a masked free and a
compaction must not race for the same pages, so the gap is freed by
:meth:`MiniCPMO45DuplexWindowManager.reanchor_block_table` and nothing else.

Reuse, not reimplementation
---------------------------
The base of everything below is the repo's own
:class:`~vllm_omni.experimental.ar_diffusion.kv_cache.paged.ChunkWindowSpec` /
:class:`~vllm_omni.experimental.ar_diffusion.kv_cache.paged.ChunkWindowManager`,
which already keeps a sink, already evicts on chunk boundaries, and already
compacts a block table. This project sets ``chunk_size`` to the cache block size,
so "chunk-aligned" and "page-aligned" are the same statement and the inherited
:meth:`compact_block_table` indexing applies as written.

Two constraints come with this
------------------------------
*Prefix caching stays off.* Not only because a compacted table is no longer the
table the block hashes were computed against, but because duplex appends are
prompts of ``[filler_id] * token_budget`` (``duplex/plugin.py``) whose real
content arrives as worker-side ``inputs_embeds``: the hashes would be identical
across sessions while the KV behind them is not. ``OmniTensorPrefixCache`` has
the same exposure through its block/slot mirroring, and it and async output
materialization are mutually exclusive anyway.
*Block size.* On CUDA, kernels typically use ``block_size == 16`` for a sliding window,
while Ascend NPU uses 128. The window geometry and compaction algorithms support any
positive block size. The conversion below reads that size back off the spec the
parent produced, so the kernel's choice -- not this module's -- is the one the
chunk arithmetic uses.

The arithmetic it delegates to is CPU-covered by
``tests/model_executor/models/minicpmo_4_5/duplex/test_window_plan.py``, which
also checks the RoPE identity behind :func:`rotate_cached_keys` numerically.
``tests/worker_v2/test_minicpmo_duplex_block_table.py`` exercises MRv2's actual
GPU block-table gather across the allocation bound and after compaction. The
spec disables ordinary sliding-window masking: that bound sizes admission,
but does not authorize the worker to hide the manager's retained pages.

Wiring, in the order the pieces get used
----------------------------------------
1. Model module: import this module for its spec registration, call
   :func:`install_duplex_window_layers` on the Stage 0 backbone right after
   ``init_vllm_registered_model`` so the profiler sees the windowed spec, and run
   :func:`validate_duplex_window_install` as the startup assertion.
2. Scheduler: an append becomes an ordinary session *extension*, so
   ``_update_request_as_session`` runs instead of ``_replace_streaming_session``
   and the request's token count grows rather than restarting. Nothing else
   changes until
   :func:`~vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_plan.plan_position_reanchor`
   returns a plan, which is the single place a trim -- or a
   ``context_length_exceeded`` finish -- is decided.
3. Worker: for that step, check :func:`assert_uniform_position_shift` on the
   request's real positions, call
   :meth:`MiniCPMO45DuplexWindowManager.reanchor_block_table` once, then
   :func:`rotate_cached_keys` per layer, and shift the request's positions,
   ``mrope_positions`` and ``mrope_position_delta`` in the same step so the next
   forward reads the compacted layout.
4. Deploy config: ``enable_prefix_caching`` stays false, ``block_size`` matches
   the engine's cache configuration, and the window's watermarks come from
   the model's duplex policy rather than from a scheduler constant.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowSpec
from vllm.v1.kv_cache_spec_registry import register_kv_cache_spec

from vllm_omni.core.sched.window_policy import WindowCut, WindowOperation
from vllm_omni.engine.duplex.contracts import parse_duplex_resource_request_id
from vllm_omni.experimental.ar_diffusion.kv_cache.paged import (
    ChunkWindowManager,
    ChunkWindowSpec,
)
from vllm_omni.model_executor.common.duplex.payload import payload_audio
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import MiniCPMO45TurnState, record_turn_sample
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_plan import (
    CARRIED_WINDOW_MAX_SESSIONS,
    CARRIED_WINDOW_OFFER_TTL_S,
    DuplexWindowGeometry,
    PositionReanchor,
    align_up,
    cdiv,
    plan_position_reanchor,
    prepare_stage0_window,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request, StreamingUpdate

    from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler


#: Default block size for sliding window on CUDA. Hardware like Ascend NPU may
#: use larger page sizes (e.g. 128).
DUPLEX_WINDOW_BLOCK_SIZE = 16


def duplex_stage0_session_id(request_id: str) -> str | None:
    """Session id of a duplex Stage-0 resource request (``duplex-s.….r.stage0``), else None."""
    fields = parse_duplex_resource_request_id(request_id)
    return fields.session_id if fields is not None and fields.role == "stage0" else None


def duplex_window_geometry(
    *,
    prefix_tokens: int,
    window_tokens: int,
    block_size: int,
    max_model_len: int,
    sample_room: int = 1,
    high_watermark_tokens: int | None = None,
) -> DuplexWindowGeometry:
    """Bind the model's window policy to the engine's paging geometry."""
    return DuplexWindowGeometry(
        prefix_tokens=int(prefix_tokens),
        window_tokens=int(window_tokens),
        block_size=int(block_size),
        max_model_len=int(max_model_len),
        sample_room=int(sample_room),
        high_watermark_tokens=None if high_watermark_tokens is None else int(high_watermark_tokens),
    )


def validate_duplex_window_install(
    cache_config,
    model_config,
    geometry: DuplexWindowGeometry,
) -> None:
    """Refuse a configuration the window cannot be enforced under.

    Called from the model module once both configs exist and before the
    KV-cache profile, so a wrong deploy config fails at startup rather than
    leaving a session that grows without bound.
    """
    if getattr(cache_config, "enable_prefix_caching", False):
        raise ValueError("MiniCPM-o 4.5 duplex KV window requires enable_prefix_caching=False")
    if geometry.block_size <= 0:
        raise ValueError(f"duplex KV window needs positive block_size, got {geometry.block_size}")
    configured_block_size = getattr(cache_config, "block_size", None)
    if configured_block_size is not None and geometry.block_size != configured_block_size:
        raise ValueError(
            f"duplex KV window geometry block_size ({geometry.block_size}) "
            f"does not match cache_config.block_size ({configured_block_size})"
        )
    needed = geometry.prefix_tokens + geometry.trigger_tokens + geometry.sample_room
    max_model_len = getattr(model_config, "max_model_len", None) if model_config is not None else None
    if max_model_len is not None and needed > max_model_len:
        raise ValueError(
            f"duplex window does not fit max_model_len={max_model_len}: "
            f"prefix={geometry.prefix_tokens} + trigger={geometry.trigger_tokens} "
            f"+ sample_room={geometry.sample_room} = {needed}"
        )


def resident_bound(geometry: DuplexWindowGeometry) -> int:
    """Tokens a windowed session can hold at once, rounded to whole pages."""
    return min(
        geometry.prefix_tokens + geometry.trigger_tokens + geometry.sample_room,
        geometry.max_model_len,
    )


class MiniCPMO45DuplexWindowManager(ChunkWindowManager):
    """Chunk-window paging with the trim made explicit: free, delete, re-RoPE.

    Inherited as-is: ``req_to_blocks`` for the table edit and
    :meth:`compact_block_table` for the deletion itself. Overridden: the
    automatic eviction, which for this model would free a head the session still
    has to attend. It is inert rather than removed so that admission, which reads
    the same spec, still sees a bounded request.
    """

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """Never free behind the compaction's back."""
        del num_computed_tokens
        return 0

    def compact_block_table(self, request_id: str, sink_blocks: int | None = None) -> int:
        if getattr(self, "enable_caching", False):
            raise RuntimeError("duplex block-table compaction requires prefix caching to be disabled")
        blocks = getattr(self, "req_to_blocks", {}).get(request_id)
        if blocks is None:
            blocks = getattr(self, "blocks", None)
        if blocks is None:
            return 0
        spec_sink = getattr(getattr(self, "kv_cache_spec", None), "sink_chunks", 0)
        start = spec_sink if sink_blocks is None else int(sink_blocks)
        end = start
        null_block = getattr(self, "_null_block", None)
        while end < len(blocks) and blocks[end] == null_block:
            end += 1
        if end == start:
            return 0
        del blocks[start:end]
        if hasattr(self, "num_cached_block") and request_id in self.num_cached_block:
            cached = self.num_cached_block[request_id]
            self.num_cached_block[request_id] = min(cached, start) + max(0, cached - end)
        return (end - start) * getattr(self, "block_size", 16)

    def compact_request_blocks(
        self,
        request_id: str,
        *,
        sink_blocks: int,
        num_blocks: int,
    ) -> int:
        """Free and compact a gap of blocks after an explicit request-local sink."""
        if getattr(self, "enable_caching", False):
            raise RuntimeError("duplex block-table compaction requires prefix caching to be disabled")
        blocks = getattr(self, "req_to_blocks", {}).get(request_id)
        if blocks is None:
            blocks = getattr(self, "blocks", None)
        if blocks is None or num_blocks <= 0:
            return 0
        start = int(sink_blocks)
        end = start + int(num_blocks)
        null_block = getattr(self, "_null_block", None)
        if (
            start < 0
            or end > len(blocks)
            or (null_block is not None and any(block == null_block for block in blocks[start:end]))
        ):
            return 0
        if hasattr(self, "_remove_blocks_in_range"):
            self._remove_blocks_in_range(request_id, start, end)
        elif hasattr(self, "free_blocks_range"):
            self.free_blocks_range(request_id, start, end)
        else:
            del blocks[start:end]
            return num_blocks * getattr(self, "block_size", 16)
        return self.compact_block_table(request_id, sink_blocks=start)

    def reanchor_block_table(self, request_id: str, plan: PositionReanchor) -> int:
        """Free and delete one request's gap; return the tokens it held.

        Returns 0 when the table disagrees with the plan -- the gap was already
        removed, or the request is gone -- and a caller that asked for a non-zero
        delta must then skip the rotation as well, since nothing moved.
        """
        gap_blocks = plan.delta // self.block_size
        return self.compact_request_blocks(
            request_id,
            sink_blocks=plan.sink_blocks,
            num_blocks=gap_blocks,
        )


@register_kv_cache_spec(manager_class=MiniCPMO45DuplexWindowManager, uniform_type_base_spec=None)
class MiniCPMO45DuplexWindowSpec(ChunkWindowSpec):
    """A :class:`ChunkWindowSpec` that resolves to the duplex manager above.

    Adds no fields. Registration exists only because dispatch walks the spec's
    MRO, so an unregistered subclass silently keeps the parent manager and the
    trim entry point is never reachable.
    """

    @property
    def num_retained_tokens(self) -> int:
        """Disable MRv2's ordinary sliding-window block-table masking.

        Zero tells BlockTables to preserve the manager's complete table. Our
        sliding_window is an admission bound, not an attention mask: only the
        duplex manager may remove history, retaining sink pages and reanchoring
        the remaining positions. Sessions with windowing off can also exceed
        that bound. Masking their leading blocks silently loses live context.
        """
        return 0


def duplex_window_spec(geometry: DuplexWindowGeometry, layer_spec: SlidingWindowSpec) -> MiniCPMO45DuplexWindowSpec:
    """Re-express a layer's sliding-window spec as the duplex one.

    ``layer_spec`` is what an upstream ``Attention`` reports once it has a
    sliding window. Its ``block_size`` was chosen by the kernel, not by
    ``--block-size``, so every count below is in the pages the cache will
    actually use -- and it has to be the pages the planner counts in too, or
    ``plan.sink_blocks`` and ``spec.sink_chunks`` describe different tables.
    """
    if layer_spec.block_size != geometry.block_size:
        raise ValueError(
            f"duplex window plans in {geometry.block_size}-token pages but the kernel gave this "
            f"group {layer_spec.block_size}-token pages; align cache_config.block_size with the geometry"
        )
    block_size = layer_spec.block_size
    # The mask never bites (see the manager), so the window is set to the
    # resident bound: long enough that a compacted session is always fully
    # attended, short enough that the allocator admits a session by what it
    # holds rather than by max_model_len.
    bound = align_up(resident_bound(geometry), block_size)
    return MiniCPMO45DuplexWindowSpec(
        block_size=block_size,
        num_kv_heads=layer_spec.num_kv_heads,
        head_size=layer_spec.head_size,
        head_size_v=layer_spec.head_size_v,
        dtype=layer_spec.dtype,
        kv_quant_mode=layer_spec.kv_quant_mode,
        page_size_padded=layer_spec.page_size_padded,
        sliding_window=bound,
        chunk_size=block_size,
        window_chunks=bound // block_size,
        sink_chunks=cdiv(geometry.prefix_tokens, block_size),
        reset_at_boundary=False,
    )


def install_duplex_window_layers(model: torch.nn.Module, *, geometry: DuplexWindowGeometry) -> list[str]:
    """Point a decoder's attention layers at :class:`MiniCPMO45DuplexWindowSpec`.

    Stage 0 builds its backbone through ``init_vllm_registered_model`` on a
    Qwen3/Qwen2 text config, so its layers are stock vLLM ``Attention`` modules
    reporting ``FullAttentionSpec``. ``RSWAAttention`` shows the way to change
    that without a model file: the only difference is what ``get_kv_cache_spec``
    returns. So each layer is re-classed and given the two attributes its
    override reads -- the window length that makes the parent emit a
    sliding-window spec at all, and the geometry that turns it into this one.

    The window length is ``resident_bound(geometry)``, and it exists for sizing
    only. The layer's ``impl`` was built before this call with no window, and
    that is the point: the kernel stays causal over the block table, which is
    correct precisely because the trim keeps the table dense.

    Returns the modified layer names, for the startup log and for a test that the
    spec really did change.
    """
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backend import AttentionType

    windowed = _duplex_window_attention_class()
    sliding_window = resident_bound(geometry)
    renamed: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, Attention) or isinstance(module, windowed):
            continue
        if module.attn_type != AttentionType.DECODER:
            # Encoder-side attention (the vision/audio towers) keeps its own
            # cache policy; only the decoder's KV is windowed.
            continue
        object.__setattr__(module, "__class__", windowed)
        object.__setattr__(module, "sliding_window", int(sliding_window))
        object.__setattr__(module, "_duplex_window_geometry", geometry)
        renamed.append(name)
    return renamed


@functools.lru_cache(maxsize=1)
def _duplex_window_attention_class():
    """``Attention`` that reports the duplex spec instead of a plain one.

    Subclassing at call time leaves upstream in charge of the parts that are the
    backend's business -- the kernel-chosen block size, page padding,
    quantisation mode -- and converts only the class of the result.
    """
    from vllm.model_executor.layers.attention import Attention

    class MiniCPMO45DuplexWindowAttention(Attention):
        _duplex_window_geometry: DuplexWindowGeometry

        def get_kv_cache_spec(self, vllm_config) -> KVCacheSpec | None:
            spec = super().get_kv_cache_spec(vllm_config)
            if spec is None or not isinstance(spec, SlidingWindowSpec) or isinstance(spec, ChunkWindowSpec):
                return spec
            return duplex_window_spec(self._duplex_window_geometry, spec)

    return MiniCPMO45DuplexWindowAttention


def assert_uniform_position_shift(positions: torch.Tensor, moved_from: int) -> None:
    """Guard the one case a single rotation cannot express.

    MRoPE scores against three position rows, and a translation of the tail is
    one rotation only while all three shift by the same amount. A retained span
    carrying vision tokens with distinct height/width rows does not, so the
    caller must re-prefill that session instead. Duplex audio and text advance
    every row together, so the streaming tail satisfies this.
    """
    if positions.dim() == 1:
        return
    if positions.dim() != 2:
        raise ValueError(f"expected a (rows, tokens) position tensor, got {tuple(positions.shape)}")
    tail = positions[:, int(moved_from) :]
    if tail.shape[1] == 0:
        return
    if not bool(torch.all(tail == tail[0])):
        raise RuntimeError(
            "duplex re-anchor needs one position row across the retained tail; re-prefill this "
            "session rather than rotating split MRoPE positions"
        )


def rotate_keys(keys: torch.Tensor, delta: int, inv_freq: torch.Tensor) -> torch.Tensor:
    """Undo ``delta`` positions of rotation on keys that are already rotated.

    ``keys`` is ``(tokens, heads, head_dim)`` with NeoX-style pairing
    (``rotate_half``), which is what the Qwen3-family RoPE Stage 0 registers
    writes. RoPE composes, so multiplying a key's cached pair by
    ``exp(-i * delta * inv_freq)`` yields exactly the key that would have been
    written ``delta`` positions earlier, for every frequency at once and without
    reference to where the token came from.
    """
    half = keys.shape[-1] // 2
    if inv_freq.numel() != half:
        raise ValueError(f"expected {half} RoPE frequencies for head_dim={keys.shape[-1]}, got {inv_freq.numel()}")
    # Angle and trigonometric computations must stay in float32 to avoid catastrophic
    # quantization error: delta * inv_freq can exceed 1024, where bfloat16 has ULP = 8
    # (quantization error ~4-5.7 rad, completely randomizing trig values).
    angle = float(delta) * inv_freq.to(device=keys.device, dtype=torch.float32)
    cos = torch.cos(angle).to(dtype=keys.dtype).unsqueeze(0).unsqueeze(1)
    sin = torch.sin(angle).to(dtype=keys.dtype).unsqueeze(0).unsqueeze(1)
    k1, k2 = keys[..., :half], keys[..., half:]
    # Complex multiply by exp(-i*angle): the inverse of the forward rotation.
    return torch.cat([k1 * cos + k2 * sin, k2 * cos - k1 * sin], dim=-1)


def flattened_key_cache_view(
    key: torch.Tensor,
    block_size: int | None,
    head_dim: int | None,
    rotary_dim: int,
) -> torch.Tensor:
    """Return a storage-sharing view or reject an unsupported flattened layout."""
    if head_dim is None or head_dim <= 0 or head_dim != rotary_dim:
        raise ValueError("flattened key cache requires explicit head_dim and full rotary")
    if block_size is None or key.shape[1] != block_size or key.shape[2] % head_dim:
        raise ValueError(f"unsupported flattened key cache shape {tuple(key.shape)}")
    try:
        return key.view(key.shape[0], block_size, -1, head_dim)
    except RuntimeError as exc:
        raise ValueError(
            f"cannot view flattened key cache in place: shape={tuple(key.shape)}, "
            f"stride={key.stride()}, head_dim={head_dim}"
        ) from exc


def rotate_cached_keys(
    k_pool: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
    *,
    block_ids: list[int],
    positions: torch.Tensor,
    plan: PositionReanchor,
    inv_freq: torch.Tensor,
    block_size: int | None = None,
    head_dim: int | None = None,
) -> int:
    """Re-RoPE the retained tail where it stands; return the rows touched.

    Supports:
      - vLLM 0.30+ packed FlashAttention KV layout:
        (num_blocks, num_kv_heads, block_size, 2 * head_dim)
      - Standard PagedAttention KV layouts:
        (num_blocks, block_size, num_kv_heads, head_dim) and
        (num_blocks, num_kv_heads, block_size, head_dim)
      - 5D split layouts (2, num_blocks, ...) or (num_blocks, 2, ...)

    Uses direct tensor advanced indexing assignment to write rotated keys in-place
    into the underlying storage, preserving V cache, non-retained blocks, and avoiding
    temporary copy allocation on non-contiguous strides.
    """
    if positions.numel() == 0:
        return 0

    # Ascend keeps K and V in separate tensors. Operate on the original K
    # storage only; stacking the pair would rotate a copy and leave KV stale.
    if isinstance(k_pool, (tuple, list)):
        if len(k_pool) != 2:
            raise ValueError("split KV cache must contain exactly K and V")
        k_pool = k_pool[0]

    # Ascend's standard paged layout can flatten heads into the last axis:
    # (num_blocks, block_size, num_kv_heads * head_dim).
    if k_pool.dim() == 3:
        k_pool = flattened_key_cache_view(k_pool, block_size, head_dim, inv_freq.numel() * 2)

    if k_pool.dim() == 5:
        if k_pool.shape[0] == 2:
            k_pool = k_pool[0]
        elif k_pool.shape[1] == 2:
            k_pool = k_pool[:, 0]
        else:
            raise ValueError(f"unsupported 5D cache shape {tuple(k_pool.shape)}")

    if k_pool.dim() != 4:
        raise ValueError(f"rotate_cached_keys requires a 4D key cache, got shape {tuple(k_pool.shape)}")

    expected_head_dim = inv_freq.numel() * 2
    last_dim = k_pool.shape[-1]
    if last_dim not in (expected_head_dim, 2 * expected_head_dim):
        raise ValueError(
            f"expected head_dim={expected_head_dim} or {2 * expected_head_dim} for "
            f"{inv_freq.numel()} RoPE frequencies, got {last_dim} (shape {tuple(k_pool.shape)})"
        )

    if block_size is None:
        if last_dim == 2 * expected_head_dim:
            # Packed FlashAttention: (num_blocks, num_kv_heads, block_size, 2 * head_dim)
            block_size = k_pool.shape[2]
        else:
            block_size = k_pool.shape[1]

    if k_pool.shape[2] == block_size:
        layout = "BHND"
        num_kv_heads = k_pool.shape[1]
    elif k_pool.shape[1] == block_size:
        layout = "BNHD"
        num_kv_heads = k_pool.shape[2]
    else:
        raise ValueError(f"cannot resolve layout: block_size={block_size} not in shape {tuple(k_pool.shape)}")

    positions = positions.to(device=k_pool.device, dtype=torch.long)
    if bool((positions < plan.moved_from).any()):
        raise RuntimeError(f"re-anchor given {int((positions < plan.moved_from).sum())} positions below moved_from")

    shifted = positions - plan.delta
    block_table = torch.as_tensor(block_ids, dtype=torch.long, device=k_pool.device)
    target_blocks = block_table[shifted // block_size]
    offsets = shifted % block_size

    b_idx = target_blocks.unsqueeze(1)
    h_idx = torch.arange(num_kv_heads, device=k_pool.device).unsqueeze(0)
    o_idx = offsets.unsqueeze(1)

    if layout == "BHND":
        keys = k_pool[b_idx, h_idx, o_idx, :expected_head_dim]
        rotated = rotate_keys(keys, plan.delta, inv_freq)
        k_pool[b_idx, h_idx, o_idx, :expected_head_dim] = rotated
    else:
        keys = k_pool[b_idx, o_idx, h_idx, :expected_head_dim]
        rotated = rotate_keys(keys, plan.delta, inv_freq)
        k_pool[b_idx, o_idx, h_idx, :expected_head_dim] = rotated

    return int(positions.numel())


class MiniCPMO45DuplexWindowPolicy:
    """Model-side retention policy and watermark geometry calculation."""

    @classmethod
    def plan_reanchor(
        cls,
        session: Any,
        update: Any,
        block_size: int,
        max_model_len: int,
        unit_tokens: list[int] | None = None,
        segment_output_ids: list[int] | None = None,
    ) -> PositionReanchor | None:
        """Evaluate model retention decision and return PositionReanchor if due."""
        info = getattr(update, "model_intermediate_buffer", None)
        if not isinstance(info, dict):
            return None
        duplex = info.get("duplex")
        if not isinstance(duplex, dict) or duplex.get("data_plane") is not True:
            return None
        runtime_config = duplex.get("runtime_config")
        runtime_config = runtime_config if isinstance(runtime_config, dict) else {}
        window = runtime_config.get("duplex_window_config")
        if not isinstance(window, dict):
            return None
        mode = window.get("sliding_window_mode", "off")
        if mode != "basic":
            # Only basic mode uses zero-copy KV reuse with in-place Re-RoPE.
            # Context mode relies on previous-text accumulation and partial forward,
            # so it routes explicitly through the official fallback (prepare_stage0_window).
            return None

        base_len = int(getattr(session, "num_computed_tokens", 0) or 0)
        prefix_tokens = int(runtime_config.get("duplex_window_prefix_tokens", 96) or 96)
        high_watermark = int(window.get("basic_window_high_tokens", 8000) or 8000)
        low_watermark = int(window.get("basic_window_low_tokens", 6000) or 6000)

        # Normalize total-sequence watermarks to content-only budgets for DuplexWindowGeometry
        # so that trigger_total == high_watermark and target_total == low_watermark,
        # perfectly matching the fallback path (prepare_stage0_window).
        content_low = max(block_size, low_watermark - prefix_tokens)
        content_high = max(content_low, high_watermark - prefix_tokens)

        geometry = DuplexWindowGeometry(
            prefix_tokens=prefix_tokens,
            window_tokens=content_low,
            block_size=block_size,
            max_model_len=max_model_len,
            high_watermark_tokens=content_high,
        )

        pending_tokens = len(getattr(update, "prompt_token_ids", []) or [])

        # Derive unit_tokens from session._minicpmo45_window_units if not explicitly given
        if unit_tokens is None and hasattr(session, "_minicpmo45_window_units"):
            raw_units = getattr(session, "_minicpmo45_window_units", ()) or ()
            if raw_units:
                unit_tokens = [int(u["length"]) for u in raw_units if isinstance(u, dict) and "length" in u]
                seq = int(duplex.get("seq", 0) or 0)
                if seq > 1 and segment_output_ids is not None:
                    preserve_len = int(runtime_config.get("duplex_first_append_context_tokens", 0) or 0)
                    old_prompt_tokens = int(getattr(session, "num_prompt_tokens", 0) or 0)
                    boundary_cur = old_prompt_tokens + len(segment_output_ids) + 2
                    recorded_open = getattr(session, "_minicpmo45_window_open_start", None)
                    open_start = preserve_len if recorded_open is None else int(recorded_open)
                    closing_len = boundary_cur - open_start
                    if closing_len > 0:
                        unit_tokens = [*unit_tokens, closing_len]

        return plan_position_reanchor(
            geometry,
            computed_tokens=base_len,
            pending_tokens=pending_tokens,
            unit_tokens=unit_tokens,
        )


class MiniCPMO45SchedulerWindowPolicy:
    """Scheduler-side window planning and request compaction helper for MiniCPM-o 4.5 duplex."""

    @classmethod
    def find_duplex_window_manager(cls, scheduler: Any) -> MiniCPMO45DuplexWindowManager | None:
        coordinator = getattr(getattr(scheduler, "kv_cache_manager", None), "coordinator", None)
        if coordinator is None:
            return None
        for mgr in getattr(coordinator, "single_type_managers", ()):
            if isinstance(mgr, (MiniCPMO45DuplexWindowManager, ChunkWindowManager)):
                return mgr
        return None

    @classmethod
    def apply_session_window(
        cls,
        scheduler: Any,
        session: Any,
        update: Any,
        *,
        segment_output_ids: list[int] | None = None,
        completed_terminator: int | None = None,
        unit_tokens: list[int] | None = None,
    ) -> PositionReanchor | None:
        """Evaluate watermark policy, compact KV block table and session tokens if needed."""
        cache_config = getattr(scheduler, "cache_config", None)
        block_size = int(getattr(cache_config, "block_size", 16) or 16)
        model_config = getattr(scheduler, "model_config", None)
        max_model_len = int(getattr(model_config, "max_model_len", 40960) or 40960)

        plan = MiniCPMO45DuplexWindowPolicy.plan_reanchor(
            session=session,
            update=update,
            block_size=block_size,
            max_model_len=max_model_len,
            unit_tokens=unit_tokens,
            segment_output_ids=segment_output_ids,
        )
        if plan is None:
            return None

        duplex_mgr = cls.find_duplex_window_manager(scheduler)
        if duplex_mgr is None:
            return None

        # Free and compact blocks via explicit request-local sink boundary on KV manager.
        # Rejection before state mutation: if compaction returns 0, no state is changed.
        gap_blocks = plan.delta // block_size
        if hasattr(duplex_mgr, "compact_request_blocks"):
            freed_tokens = duplex_mgr.compact_request_blocks(
                session.request_id,
                sink_blocks=plan.sink_blocks,
                num_blocks=gap_blocks,
            )
        else:
            freed_tokens = duplex_mgr.reanchor_block_table(session.request_id, plan)

        if freed_tokens == 0:
            return None

        old_computed = session.num_computed_tokens
        sink_end = plan.sink_end
        moved_from = plan.moved_from

        # Scheduler owns corresponding token-history and counter updates derived from applied compaction result
        old_prompt_tokens = (
            len(session.prompt_token_ids)
            if getattr(session, "prompt_token_ids", None) is not None
            else int(getattr(session, "num_prompt_tokens", 0) or old_computed)
        )
        if getattr(session, "prompt_token_ids", None) is not None and len(session.prompt_token_ids) >= moved_from:
            session.prompt_token_ids = session.prompt_token_ids[:sink_end] + session.prompt_token_ids[moved_from:]
            session.num_prompt_tokens = len(session.prompt_token_ids)

        if getattr(session, "_all_token_ids", None) is not None and len(session._all_token_ids) >= moved_from:
            session._all_token_ids = session._all_token_ids[:sink_end] + session._all_token_ids[moved_from:]

        session.num_computed_tokens -= freed_tokens

        # Worker instruction derived from the same applied compaction result
        info = update.model_intermediate_buffer
        duplex = info.setdefault("duplex", {})
        runtime_config = duplex.get("runtime_config")
        runtime_config = runtime_config if isinstance(runtime_config, dict) else {}
        preserve_len = int(runtime_config.get("duplex_first_append_context_tokens", 0) or 0)
        seq = int(duplex.get("seq", 0) or 0)

        reanchor_count = getattr(session, "_minicpmo45_reanchor_count", 0) + 1
        session._minicpmo45_reanchor_count = reanchor_count
        reanchor_id = f"{session.request_id}-r{reanchor_count}-{plan.moved_from}-{plan.delta}"
        duplex["stage0_reanchor"] = {
            "reanchor_id": reanchor_id,
            "delta": plan.delta,
            "moved_from": plan.moved_from,
            "sink_blocks": plan.sink_blocks,
            "sink_end": plan.sink_end,
            "prefix_tokens": preserve_len,
            "block_size": block_size,
            "old_computed_tokens": old_computed,
        }

        # Synchronize stage0_window with worker so worker finalizes the completed unit
        generated_ids = list(segment_output_ids) if segment_output_ids is not None else []
        stage0_window = duplex.setdefault("stage0_window", {})
        stage0_window["completed_token_ids"] = generated_ids
        if completed_terminator is not None:
            stage0_window["completed_terminator_token_id"] = int(completed_terminator)

        # Synchronize scheduler window units and coordinates atomically with compaction
        boundary_before = old_prompt_tokens + len(generated_ids) + 2
        recorded_open_start = getattr(session, "_minicpmo45_window_open_start", None)
        open_start = preserve_len if recorded_open_start is None else int(recorded_open_start)
        unit_len = boundary_before - open_start

        special_ids = {
            int(token_id)
            for token_id in runtime_config.get("duplex_window_special_token_ids", ())
            if isinstance(token_id, int)
        }
        units = list(getattr(session, "_minicpmo45_window_units", ()))
        if seq > 1 and unit_len > 0:
            units.append(
                {
                    "length": unit_len,
                    "generated_token_ids": [int(t) for t in generated_ids if t not in special_ids],
                }
            )

        # Prune units dropped by compaction
        drop_start = max(0, plan.sink_end - preserve_len)
        drop_end = drop_start + freed_tokens

        new_units: list[dict[str, Any]] = []
        curr_offset = 0
        for u in units:
            u_len = int(u.get("length", 0))
            u_start = curr_offset
            u_end = curr_offset + u_len
            curr_offset = u_end

            local_drop_start = max(0, drop_start - u_start)
            local_drop_end = min(u_len, drop_end - u_start)

            if local_drop_start >= local_drop_end:
                new_units.append(u)
            elif local_drop_start == 0 and local_drop_end == u_len:
                continue
            else:
                new_len = u_len - (local_drop_end - local_drop_start)
                gen_ids = u.get("generated_token_ids", [])
                kept_gen_ids = gen_ids if local_drop_end < u_len else []
                new_units.append(
                    {
                        "length": new_len,
                        "generated_token_ids": kept_gen_ids,
                    }
                )
        session._minicpmo45_window_units = new_units

        # Rebase open_start into compacted coordinates
        if seq > 1:
            session._minicpmo45_window_open_start = boundary_before - freed_tokens
        elif recorded_open_start is not None:
            session._minicpmo45_window_open_start = max(0, recorded_open_start - freed_tokens)

        return plan

    def __init__(self) -> None:
        self.carry: MiniCPMO45CarriedWindows | None = None

    @staticmethod
    def observe_progress(request: Request) -> None:
        info = getattr(request, "model_intermediate_buffer", None)
        duplex = info.get("duplex") if isinstance(info, dict) else None
        if not isinstance(duplex, dict) or not duplex.get("data_plane"):
            return
        cfg = duplex.get("runtime_config", {})
        ids = cfg.get("duplex_boundary_token_ids")
        if not ids or duplex.get("session_window_owned") is False:
            return
        # Only returned frames count. num_computed_tokens already includes outstanding dispatches.
        settled = min(request.num_computed_tokens - request.num_in_flight_tokens, request.num_tokens)
        seq = duplex["seq"]
        prior = getattr(request, "_minicpmo45_settled", None)
        start = getattr(request, "_minicpmo45_window_open_start", None)
        if start is None:
            start = cfg["duplex_first_append_context_tokens"]
        if settled <= start:
            return
        payload = duplex.get("payload") or {}
        if prior is None or prior["seq"] != seq:
            seeded = bool(cfg.get("initial_user_text")) and not getattr(request, "_minicpmo45_replay_source", None)
            turn = (
                replace(prior["turn"])
                if prior
                else MiniCPMO45TurnState(
                    current_turn_ended=not seeded, pending_speech_context=seeded, pending_speech_response_open=seeded
                )
            )
            if payload.get("is_speech") or (payload.get("video_frames") and payload_audio(payload) is None):
                turn.pending_speech_context = True
            request._minicpmo45_turn_at_start = turn
        turn = replace(request._minicpmo45_turn_at_start)
        generated = list(request._all_token_ids[request.num_prompt_tokens :])
        terminators = {ids["listen_token_id"], ids["chunk_eos_token_id"], ids["chunk_tts_eos_token_id"]}
        # A parked segment's last decision is known, but its terminator is
        # fed with </unit> on the next prefill. Other samples require a settled decoder row.
        consumed = min(len(generated), max(0, settled - request.num_prompt_tokens))
        parked = bool(generated and generated[-1] in terminators)
        active_ids = generated[:-1] if parked else generated[:consumed]
        accepted = generated if parked else active_ids
        if accepted and payload.get("open_assistant_turn") is True and payload.get("is_speech") is not True:
            if turn.current_turn_ended:
                turn.current_turn_ended = False
                turn.pending_speech_response_open = True
                turn.explicit_response_awaiting_content = True
        for token in accepted:
            record_turn_sample(turn, token, ids, payload, terminator=token in terminators)
        ended = turn.current_turn_ended
        if parked:
            tail = [generated[-1], ids["unit_end_token_id"]]
        else:
            tail = ([] if ended else [ids["turn_eos_token_id"]]) + [ids["listen_token_id"], ids["unit_end_token_id"]]
        special = cfg["duplex_window_special_token_ids"]
        units = list(getattr(request, "_minicpmo45_window_units", ()))
        rows = min(settled, request.num_prompt_tokens) - start
        units.append(
            {
                "length": rows + len(active_ids) + len(tail),
                "generated_token_ids": [t for t in active_ids if t not in special],
            }
        )
        request._minicpmo45_settled = {
            "seq": seq,
            "units": units,
            "previous": list(getattr(request, "_minicpmo45_window_previous_token_ids", ())),
            "start": start,
            "rows": rows,
            "tail": [*active_ids, *tail],
            "turn": turn,
            "close_on_next_input": parked and not ended,
            "parked": parked,
            "accepted": accepted,
        }

    def superseded_by(self, request: Request, live: Mapping[str, Request]) -> list[str]:
        sid = duplex_stage0_session_id(request.request_id)
        if sid is None:
            return []
        _, epoch, _ = _stage0_order(request.request_id)
        return [
            rid
            for rid, previous in live.items()
            if duplex_stage0_session_id(rid) == sid and _stage0_order(rid)[1] < epoch and not previous.is_finished()
        ]

    def prepare_admission(self, request: Request, max_model_len: int) -> None:
        sid = duplex_stage0_session_id(request.request_id)
        if sid is None:
            return
        if self.carry is None:
            self.carry = MiniCPMO45CarriedWindows()
        carry = self.carry
        window = carry.windows.get(sid)
        # Abort may have completed without another scheduler dispatch in between.
        for old_id, pending in reversed(carry.pending.items()):
            if duplex_stage0_session_id(old_id) == sid:
                window = pending
                break
        info = getattr(request, "model_intermediate_buffer", None)
        duplex = info.get("duplex") if isinstance(info, dict) else None
        if not isinstance(duplex, dict) or duplex.get("seq") != 1:
            return
        if duplex.get("session_window_owned") is False:
            return
        cfg = duplex["runtime_config"]
        if "duplex_first_append_context_tokens" in cfg:
            duplex["open_input_start"] = cfg["duplex_first_append_context_tokens"]
        if not window or "seq" not in window:
            return
        if _stage0_order(request.request_id)[1] <= _stage0_order(window["source"])[1]:
            return
        mode = cfg["duplex_window_config"]["sliding_window_mode"]
        units = list(window["units"])
        previous = list(window["previous"])
        marker = cfg.get("duplex_window_previous_marker_token_ids", [])
        previous_len = len(marker) + len(previous) if previous and mode == "context" else 0
        carried = sum(unit["length"] for unit in units)
        extra = carried + previous_len + int(window["close_on_next_input"])
        length = request.num_prompt_tokens + extra
        config = cfg["duplex_window_config"]
        # If the next prefill crosses the same window watermark as an ordinary
        # append, evict whole oldest units before admission. Keep at least one
        # decode position within the deployed model's context length.
        high = int(config.get("basic_window_high_tokens", max_model_len)) if mode == "basic" else max_model_len
        cap = max_model_len - 1
        if mode == "basic" and length > high:
            cap = min(cap, int(config.get("basic_window_low_tokens", high)))
        max_units = int(config.get("context_max_units", len(units))) if mode == "context" else len(units)
        skipped = 0
        while units and (length > cap or len(units) > max_units):
            unit = units.pop(0)
            skipped += 1
            carried -= unit["length"]
            if mode == "context":
                limit = max(0, int(config.get("context_previous_max_tokens", 500)))
                previous = (previous + unit["generated_token_ids"])[-limit:] if limit else []
                previous_len = len(marker) + len(previous) if previous else 0
            length = request.num_prompt_tokens + carried + previous_len + int(window["close_on_next_input"])
        if length >= max_model_len:
            # The original input is already at the deployment limit. Leave the
            # request's usual admission/overflow handling in control.
            return
        prompt_ids = [cfg["duplex_scheduler_token_id"]] * length
        request.prompt_token_ids = prompt_ids
        del request._all_token_ids[:]
        request._all_token_ids.extend(prompt_ids)
        request.num_prompt_tokens = length
        request.block_hashes.clear()
        request.update_block_hashes()
        request._minicpmo45_window_units = units.copy()
        request._minicpmo45_window_previous_token_ids = list(previous)
        request._minicpmo45_window_previous_len = previous_len
        request._minicpmo45_window_open_start = cfg["duplex_first_append_context_tokens"] + carried + previous_len
        carry.admitted[request.request_id] = window["source"]
        request._minicpmo45_replay_source = window["source"]
        duplex["open_input_start"] = request._minicpmo45_window_open_start
        duplex["stage0_window"] = {
            "replace": True,
            "mode": mode,
            "replay_cut": request.request_id,
            "carried_units": len(units),
            "evicted_at_cut": skipped,
            "carried_unit_lengths": [unit["length"] for unit in units],
            "replacement_prompt_len": length,
            "previous_token_ids": previous,
            "previous_marker_token_ids": marker,
            "close_on_next_input": window["close_on_next_input"],
        }
        if window["close_on_next_input"]:
            duplex["payload"] = {**duplex["payload"], "force_listen": True}

    def prepare_append(
        self,
        scheduler: OmniARScheduler,
        request: Request,
        update: StreamingUpdate,
        *,
        segment_output_ids: list[int],
        completed_terminator: int | None,
    ) -> int | None:
        self.snapshot_window(request)
        # A carried window must be placed by replacing the prompt, never by an in-place trim.
        reanchored = None
        if (
            self.find_duplex_window_manager(scheduler) is not None
            and getattr(request, "_minicpmo45_carried_window", None) is None
        ):
            reanchored = self.apply_session_window(
                scheduler,
                request,
                update,
                segment_output_ids=segment_output_ids,
                completed_terminator=completed_terminator,
            )
        info = getattr(update, "model_intermediate_buffer", None)
        duplex = info.get("duplex") if isinstance(info, dict) else None
        if isinstance(duplex, dict) and isinstance(duplex.get("stage0_window"), dict):
            request._minicpmo45_window_planned = True
        replaced = reanchored is None and prepare_stage0_window(
            request,
            update,
            segment_output_ids=segment_output_ids,
            completed_terminator=completed_terminator,
        )
        if isinstance(duplex, dict) and (start := getattr(request, "_minicpmo45_window_open_start", None)) is not None:
            duplex["open_input_start"] = start
        if replaced:
            return len(update.prompt_token_ids)
        return None

    def release_session(self, session_id: str) -> None:
        if (carry := self.carry) is None:
            return
        carry.windows.pop(session_id, None)
        for records in (carry.pending, carry.admitted, carry.snapshot_ids, carry.unstarted):
            for rid in list(records):
                if duplex_stage0_session_id(rid) == session_id:
                    if isinstance(records, dict):
                        records.pop(rid, None)
                    else:
                        records.discard(rid)

    def snapshot_window(self, session: Any) -> None:
        """Remember a Stage-0 request's window as the worker has it, before an append changes it.

        The window plan runs when the append arrives, but the worker updates its
        own units only when it runs that append. Until the append is scheduled,
        an abort must keep the window as it was before the plan.
        """
        if duplex_stage0_session_id(session.request_id) is None:
            return
        carry = self.carry
        if carry is None:
            carry = MiniCPMO45CarriedWindows()
            self.carry = carry
        if getattr(session, "_minicpmo45_window_snapshot", None) is not None:
            # An earlier append is still waiting to be booked: the worker has the window from before that one.
            return
        # Taken even when the window is empty: the worker has no closed unit before the first plan either.
        session._minicpmo45_window_snapshot = {
            "units": [dict(unit) for unit in getattr(session, "_minicpmo45_window_units", None) or ()],
            "previous": list(getattr(session, "_minicpmo45_window_previous_token_ids", ()) or ()),
            "carried": getattr(session, "_minicpmo45_carried_window", None),
        }
        carry.snapshot_ids.add(session.request_id)

    def before_book(self, requests: Mapping[str, Request], output: SchedulerOutput) -> None:
        """Drop the snapshots of requests whose append this step hands to the worker."""
        carry = self.carry
        if carry is None or not carry.snapshot_ids:
            return
        for request_id in carry.snapshot_ids.intersection(output.num_scheduled_tokens):
            carry.snapshot_ids.discard(request_id)
            request = requests.get(request_id)
            if request is not None:
                request._minicpmo45_window_snapshot = None

    def before_finish(self, requests: Mapping[str, Request], request_ids: Iterable[str], *, aborted: bool) -> None:
        """Note what finished duplex Stage-0 requests leave; the worker learns of them at the next step."""
        carry = self.carry
        if carry is None:
            return
        for request_id in request_ids:
            request = requests.get(request_id)
            # A request that already finished (freed later) never reaches the worker's finished set again.
            live = request is not None and not request.is_finished()
            if live and duplex_stage0_session_id(request_id) is not None:
                if not getattr(request, "_minicpmo45_started", False):
                    # Never scheduled: it neither holds nor replaces the session's kept window.
                    carry.unstarted.add(request_id)
                elif (
                    aborted
                    and request_id not in carry.pending
                    and (window := self._window_left_by(request)) is not None
                ):
                    if "source" not in window and window["time"] is None:
                        window["time"] = time.monotonic()
                    carry.pending[request_id] = window
            carry.snapshot_ids.discard(request_id)

    @staticmethod
    def _window_left_by(request: Any) -> dict[str, Any] | None:
        """The completed units an aborted request leaves, or None when it has none."""
        if (settled := getattr(request, "_minicpmo45_settled", None)) is not None:
            return {**settled, "source": request.request_id, "time": None}
        # An append the worker never ran leaves the window as it was before that append's plan.
        window = getattr(request, "_minicpmo45_window_snapshot", None) or {
            "units": getattr(request, "_minicpmo45_window_units", ()),
            "previous": getattr(request, "_minicpmo45_window_previous_token_ids", ()),
            "carried": getattr(request, "_minicpmo45_carried_window", None),
        }
        unplaced = window["carried"] or {}
        units = [*unplaced.get("units", ()), *(window["units"] or ())]
        if not units:
            return None
        previous = list(window["previous"] or ()) or list(unplaced.get("previous", ()))
        # Units carried from an earlier abort keep that abort's time, so the offer never outlives the first abort.
        kept_since = unplaced.get("time") if unplaced.get("units") else None
        return {"units": units, "previous": previous, "time": kept_since}

    def on_schedule(self, requests: Mapping[str, Request], output: SchedulerOutput) -> list[WindowOperation]:
        """Decide this step's kept and handed windows, and queue the same steps for the worker.

        The worker applies ``duplex_carry_ops`` in order at the start of the
        step, before it frees the finished requests' state:
          * ``("keep", session_id, request_id)``: keep the finished request's units for the session;
          * ``("drop", session_id, "")``: drop the session's kept units;
          * ``("hand", session_id, request_id)``: give them to the session's new request.
        """
        carry = self.carry
        if carry is None:
            return []
        now = time.monotonic()
        ops: list[WindowOperation] = []
        windows = carry.windows
        # Only pack bounded offers that will reach a receiver or survive this step.
        pending_keeps: dict[str, WindowOperation] = {}
        # A queued append can overflow and finish before its snapshot is committed.
        carry.snapshot_ids.difference_update(output.finished_req_ids)
        for request_id in sorted(output.finished_req_ids, key=_stage0_order):
            session_id = duplex_stage0_session_id(request_id)
            if session_id is None:
                continue
            if request_id in carry.unstarted:
                carry.unstarted.discard(request_id)
                continue
            window = carry.pending.pop(request_id, None)
            pending_keeps.pop(session_id, None)
            if window is not None:
                windows.pop(session_id, None)
                windows[session_id] = window
                op = WindowOperation("keep", session_id, request_id)
                if "seq" in window:
                    op = WindowOperation(
                        "keep",
                        session_id,
                        request_id,
                        WindowCut(
                            window["seq"],
                            window["start"],
                            window["rows"],
                            tuple(window["accepted"]),
                            tuple(window["tail"]),
                        ),
                    )
                    ops.append(op)
                else:
                    pending_keeps[session_id] = op
            elif windows.get(session_id, {}).get("source"):
                # A request which was cancelled before it settled any new rows
                # leaves its predecessor intact. Close/expiry releases the session.
                continue
            elif windows.pop(session_id, None) is not None:
                # The session's later request left nothing to keep, so its earlier window goes too.
                ops.append(WindowOperation("drop", session_id))
        # Highest epoch first: if two of a session's requests start in one step, the newer one gets the window.
        new_reqs = getattr(output, "scheduled_new_reqs", ()) or ()
        new_ids = sorted((req.req_id for req in new_reqs), key=_stage0_order, reverse=True)
        for request_id in new_ids:
            session_id = duplex_stage0_session_id(request_id)
            if session_id is None:
                continue
            if (started := requests.get(request_id)) is not None:
                started._minicpmo45_started = True
            if session_id not in windows:
                continue
            if "source" in windows[session_id]:
                # The receiver was bound before its prompt was sized. Cancelled or
                # superseded waiters cannot take another request's replay buffer.
                continue
            window = windows.pop(session_id)
            request = requests.get(request_id)
            if (
                request is None
                or now - window["time"] > CARRIED_WINDOW_OFFER_TTL_S
                # The worker drops carried units at the request's first window plan, so a carry that
                # arrives after that plan can no longer be placed.
                or getattr(request, "_minicpmo45_window_planned", False)
            ):
                ops.append(WindowOperation("drop", session_id))
                continue
            request._minicpmo45_carried_window = window
            if (keep := pending_keeps.pop(session_id, None)) is not None:
                ops.append(keep)
            ops.append(WindowOperation("hand", session_id, request_id))
        for target, source in carry.admitted.items():
            if target not in output.finished_req_ids:
                ops.append(
                    WindowOperation("restore", duplex_stage0_session_id(target) or "", target, source_request_id=source)
                )
        carry.admitted.clear()
        for session_id in [
            sid
            for sid, kept in windows.items()
            if "source" not in kept and now - kept["time"] > CARRIED_WINDOW_OFFER_TTL_S
        ]:
            windows.pop(session_id)
            ops.append(WindowOperation("drop", session_id))
        unowned = [sid for sid, kept in windows.items() if "source" not in kept]
        for oldest in unowned[:-CARRIED_WINDOW_MAX_SESSIONS]:
            windows.pop(oldest)
            ops.append(WindowOperation("drop", oldest))

        ops.extend(op for sid, op in pending_keeps.items() if sid in windows)
        return ops


def _stage0_order(request_id: str) -> tuple[str, int, str]:
    """Order finished or new Stage-0 requests by session, then by epoch as a number (``e.9`` before ``e.10``)."""
    fields = parse_duplex_resource_request_id(request_id)
    if fields is None or fields.role != "stage0":
        return "", -1, request_id
    return fields.session_id, fields.epoch, request_id


class MiniCPMO45CarriedWindows:
    """Scheduler-side barge-in carry state, one per scheduler (see ``MiniCPMO45SchedulerWindowPolicy``)."""

    def __init__(self) -> None:
        #: session id -> the window its aborted Stage-0 request left; the worker keeps the same ones.
        self.windows: dict[str, dict[str, Any]] = {}
        #: request id -> window of an aborted request whose finish has not reached the worker yet.
        self.pending: dict[str, dict[str, Any]] = {}
        #: Stage-0 requests whose window snapshot waits for their append to be booked.
        self.snapshot_ids: set[str] = set()
        #: Stage-0 requests that finished before their first scheduled step.
        self.unstarted: set[str] = set()
        self.admitted: dict[str, str] = {}


class MiniCPMO45DuplexWorkerHelper:
    """Worker-side KV cache rotation and position metadata helper for MiniCPM-o 4.5 duplex."""

    @classmethod
    def get_reanchor_parameters(cls, runner: Any) -> tuple[int, int | None, torch.Tensor]:
        """Resolve the same cache geometry for startup validation and rotation."""
        block_size = int(getattr(getattr(runner, "cache_config", None), "block_size", 16) or 16)
        model_config = getattr(runner, "model_config", None)
        head_dim = model_config.get_head_size() if model_config is not None else None
        return block_size, head_dim, cls.get_rope_inv_freq(runner)

    @classmethod
    def get_rope_inv_freq(cls, runner: Any) -> torch.Tensor:
        inv_freq = getattr(runner, "_duplex_inv_freq", None)
        if inv_freq is None:
            model_config = getattr(runner, "model_config", None)
            head_dim = (
                model_config.get_head_size()
                if model_config is not None and hasattr(model_config, "get_head_size")
                else 128
            )
            hf_config = getattr(model_config, "hf_config", None) if model_config is not None else None
            base = float(getattr(hf_config, "rope_theta", 1000000.0) or 1000000.0)
            inv_freq = 1.0 / (
                base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=runner.device) / head_dim)
            )
            runner._duplex_inv_freq = inv_freq
        return inv_freq

    @classmethod
    def resolve_group_block_ids(
        cls,
        runner: Any,
        req_id: str,
        req_idx: int,
        group_idx: int = 0,
    ) -> list[int]:
        """Resolve the flat block ID table for a specific KV cache group of a request.

        In production, req_state.block_ids is a tuple of lists, one per KV cache group.
        This helper unpacks the appropriate group to ensure compute_slot_mapping receives
        a flat 1D list of integer block IDs.
        """
        req_state = getattr(runner, "requests", {}).get(req_id) if hasattr(runner, "requests") else None
        if req_state is not None and getattr(req_state, "block_ids", None):
            raw_blocks = req_state.block_ids
            if isinstance(raw_blocks, (tuple, list)) and len(raw_blocks) > 0:
                if isinstance(raw_blocks[0], (tuple, list)):
                    actual_group = group_idx if group_idx < len(raw_blocks) else 0
                    return [int(b) for b in raw_blocks[actual_group]]
                return [int(b) for b in raw_blocks]
        # MRv2 owns request slots and block tables on the runner, rather than
        # on its transient InputBatch. Read after staged writes are applied.
        req_states = getattr(runner, "req_states", None)
        block_tables = getattr(runner, "block_tables", None)
        if req_states is not None and block_tables is not None:
            slot = req_states.req_id_to_index.get(req_id)
            if slot is not None:
                num_blocks = int(block_tables.num_blocks.np[group_idx, slot])
                return block_tables.block_tables[group_idx].gpu[slot, :num_blocks].tolist()
        # Fallback to input_batch block_table
        bt = getattr(getattr(runner, "input_batch", None), "block_table", None)
        if bt is not None:
            bt_tables = getattr(bt, "block_tables", None)
            if bt_tables and len(bt_tables) > 0:
                actual_group = group_idx if group_idx < len(bt_tables) else 0
                bt_row = bt_tables[actual_group]
            else:
                bt_row = bt
            num_blocks = int(bt_row.num_blocks_per_row[req_idx])
            return [int(b) for b in bt_row.block_table.np[req_idx, :num_blocks]]
        return []

    @classmethod
    def maybe_apply_reanchor(cls, runner: Any, scheduler_output: Any = None) -> None:
        """Apply in-place KV reanchor and rotation on worker before model forward."""
        req_states = getattr(runner, "req_states", None)
        if req_states is not None and scheduler_output is not None:
            req_ids = list(scheduler_output.num_scheduled_tokens)
        else:
            if not hasattr(runner, "input_batch") or runner.input_batch is None:
                return
            num_reqs = getattr(runner.input_batch, "num_reqs", len(runner.input_batch.req_ids))
            req_ids = runner.input_batch.req_ids[:num_reqs]
        applied_reanchors = getattr(runner, "_applied_stage0_reanchor_ids", None)
        if applied_reanchors is None:
            applied_reanchors = runner._applied_stage0_reanchor_ids = set()

        for req_idx, req_id in enumerate(req_ids):
            if req_states is not None:
                slot = req_states.req_id_to_index.get(req_id)
                if slot is None:
                    continue
                info = runner.model_state.intermediate_buffer.buffers[slot]
            else:
                info = runner.model_intermediate_buffer.get(req_id)
            if not isinstance(info, dict):
                continue
            duplex = info.get("duplex")
            if not isinstance(duplex, dict):
                continue
            reanchor = duplex.pop("stage0_reanchor", None)
            if reanchor is None:
                continue

            # Sanitize scheduled_new_reqs in scheduler_output so subsequent runner
            # metadata refreshes (_update_additional_information in _preprocess)
            # do not re-inject this already popped reanchor command.
            if scheduler_output is not None and hasattr(scheduler_output, "scheduled_new_reqs"):
                for new_req in scheduler_output.scheduled_new_reqs:
                    if getattr(new_req, "req_id", None) == req_id:
                        buf = getattr(new_req, "model_intermediate_buffer", None)
                        if isinstance(buf, dict) and isinstance(buf.get("duplex"), dict):
                            buf["duplex"].pop("stage0_reanchor", None)

            reanchor_sig = (
                f"{req_id}:{reanchor.get('moved_from')}:{reanchor.get('delta')}:{reanchor.get('old_computed_tokens')}"
            )
            reanchor_id = reanchor.get("reanchor_id") or reanchor_sig
            if reanchor_id in applied_reanchors:
                continue
            applied_reanchors.add(reanchor_id)
            reanchor["reanchor_id"] = reanchor_id

            plan = PositionReanchor(
                delta=reanchor["delta"],
                moved_from=reanchor["moved_from"],
                sink_blocks=reanchor["sink_blocks"],
            )

            # Scheduler is authoritative for logical state (block_table and computed tokens).
            # The parent runner's _update_states() already installed the post-compaction block IDs
            # and decremented num_computed_tokens_cpu. We do NOT double-compact or double-decrement here.
            old_computed = reanchor.get("old_computed_tokens")
            if old_computed is None:
                if req_states is not None:
                    slot = req_states.req_id_to_index[req_id]
                    old_computed = int(req_states.num_computed_tokens_np[slot]) + plan.delta
                else:
                    old_computed = int(runner.input_batch.num_computed_tokens_cpu[req_idx]) + plan.delta
            old_computed = int(old_computed)

            req_state = runner.requests.get(req_id) if hasattr(runner, "requests") else None
            mrope_pos = getattr(req_state, "mrope_positions", None) if req_state is not None else None
            if mrope_pos is not None:
                assert_uniform_position_shift(mrope_pos, plan.moved_from)
                block_size = int(getattr(getattr(runner, "cache_config", None), "block_size", 16) or 16)
                sink_tokens = plan.sink_blocks * block_size
                if mrope_pos.shape[1] >= old_computed:
                    req_state.mrope_positions = torch.cat(
                        [
                            mrope_pos[:, :sink_tokens],
                            mrope_pos[:, plan.moved_from : old_computed] - plan.delta,
                        ],
                        dim=1,
                    )
                if getattr(req_state, "mrope_position_delta", None) is not None:
                    req_state.mrope_position_delta = max(0, req_state.mrope_position_delta - plan.delta)

            positions = torch.arange(plan.moved_from, old_computed, dtype=torch.long, device=runner.device)
            if mrope_pos is None:
                assert_uniform_position_shift(positions, plan.moved_from)

            if positions.numel() > 0 and hasattr(runner, "kv_caches") and runner.kv_caches:
                block_size, head_dim, inv_freq = cls.get_reanchor_parameters(runner)
                kv_groups = getattr(runner, "kv_cache_group_ids", None)
                # Layers share their KV group's table; on MRv2 each lookup is a device read.
                group_block_ids: dict[int, list[int]] = {}
                for layer_idx, kv_cache in enumerate(runner.kv_caches):
                    group_idx = kv_groups[layer_idx] if (kv_groups and layer_idx < len(kv_groups)) else 0
                    layer_block_ids = group_block_ids.get(group_idx)
                    if layer_block_ids is None:
                        layer_block_ids = cls.resolve_group_block_ids(runner, req_id, req_idx, group_idx=group_idx)
                        group_block_ids[group_idx] = layer_block_ids
                    rotate_cached_keys(
                        kv_cache,
                        block_ids=layer_block_ids,
                        positions=positions,
                        plan=plan,
                        inv_freq=inv_freq,
                        block_size=block_size,
                        head_dim=head_dim,
                    )

            # Evict worker-held unit history so state.window_units remains bounded
            model = getattr(runner, "model", None)
            helper = getattr(model, "_minicpmo45_duplex_data_plane_helper", None)
            if helper is not None and isinstance(getattr(helper, "sessions", None), dict):
                req_sessions = getattr(model, "_minicpmo45_duplex_request_sessions", {})
                session_key = duplex.get("session_id") or (
                    req_sessions.get(req_id) if isinstance(req_sessions, dict) else None
                )
                session_state = helper.sessions.get(session_key) if session_key else None
                if session_state is not None and hasattr(session_state, "window_units"):
                    if hasattr(helper, "_evict_window_units_for_reanchor"):
                        helper._evict_window_units_for_reanchor(session_state, reanchor)
                    else:
                        session_state.window_units.clear()
