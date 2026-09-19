# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native paged-KV window for MiniCPM-o 4.5 duplex Stage 0.

``window_plan.py`` holds the window arithmetic in plain integers; this file
holds the three things that need an engine: the KV-cache spec/manager pair the
scheduler resolves, the layer change that produces that spec, and the in-place
re-RoPE that follows a trim.

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
*Block size 16.* FlashAttention, FlashInfer and Triton all require
``block_size == 16`` for a sliding window (``AGENTS.md``), and upstream lets the
kernel choose a sliding-window layer's block size rather than taking
``--block-size``. The conversion below reads that size back off the spec the
parent produced, so the kernel's choice -- not this module's -- is the one the
chunk arithmetic uses.

Nothing here has been executed: it needs an engine, a checkpoint and a GPU. The
arithmetic it delegates to is CPU-covered by
``tests/model_executor/models/minicpmo_4_5/duplex/test_window_plan.py``, which
also checks the RoPE identity behind :func:`rotate_cached_keys` numerically.

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
4. Deploy config: ``enable_prefix_caching`` stays false, ``block_size`` is 16,
   and the window's watermarks come from the model's duplex policy rather than
   from a scheduler constant.
"""

from __future__ import annotations

import functools
from typing import Any

import torch
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowSpec
from vllm.v1.kv_cache_spec_registry import register_kv_cache_spec

from vllm_omni.experimental.ar_diffusion.kv_cache.paged import (
    ChunkWindowManager,
    ChunkWindowSpec,
    compute_slot_mapping,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_plan import (
    DuplexWindowGeometry,
    PositionReanchor,
    align_up,
    cdiv,
    plan_position_reanchor,
)

#: FlashAttention, FlashInfer and Triton all reject other page sizes for a
#: sliding-window group.
DUPLEX_WINDOW_BLOCK_SIZE = 16


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
    if cache_config.enable_prefix_caching:
        raise ValueError("MiniCPM-o 4.5 duplex KV window requires enable_prefix_caching=False")
    if geometry.block_size != DUPLEX_WINDOW_BLOCK_SIZE:
        raise ValueError(
            f"duplex KV window needs block_size={DUPLEX_WINDOW_BLOCK_SIZE} "
            f"(sliding-window pages are fixed there), got {geometry.block_size}"
        )
    needed = geometry.prefix_tokens + geometry.trigger_tokens + geometry.sample_room
    if needed > model_config.max_model_len:
        raise ValueError(
            f"duplex window does not fit max_model_len={model_config.max_model_len}: "
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

    def plan_reanchor(
        self,
        geometry: DuplexWindowGeometry,
        *,
        computed_tokens: int,
        pending_tokens: int,
        unit_tokens: list[int] | None = None,
    ) -> PositionReanchor | None:
        """Whether the next append pushes the session past its window."""
        return plan_position_reanchor(
            geometry,
            computed_tokens=computed_tokens,
            pending_tokens=pending_tokens,
            unit_tokens=unit_tokens,
        )

    def reanchor_block_table(self, request_id: str, plan: PositionReanchor) -> int:
        """Free and delete one request's gap; return the tokens it held.

        Returns 0 when the table disagrees with the plan -- the gap was already
        removed, or the request is gone -- and a caller that asked for a non-zero
        delta must then skip the rotation as well, since nothing moved.
        """
        if self.enable_caching:
            raise RuntimeError("duplex block-table compaction requires prefix caching to be disabled")
        spec = self.kv_cache_spec
        if spec.chunk_size != self.block_size:
            raise RuntimeError(
                f"duplex trim assumes chunk_size == block_size; "
                f"got chunk_size={spec.chunk_size} block_size={self.block_size}"
            )
        blocks = self.req_to_blocks.get(request_id)
        if blocks is None:
            return 0
        start = plan.sink_blocks
        gap_blocks = plan.delta // self.block_size
        end = start + gap_blocks
        if (
            start < 0
            or gap_blocks <= 0
            or end > len(blocks)
            or any(block == self._null_block for block in blocks[start:end])
        ):
            return 0
        self._remove_blocks_in_range(request_id, start, end)
        return self.compact_block_table(request_id)


@register_kv_cache_spec(manager_class=MiniCPMO45DuplexWindowManager, uniform_type_base_spec=None)
class MiniCPMO45DuplexWindowSpec(ChunkWindowSpec):
    """A :class:`ChunkWindowSpec` that resolves to the duplex manager above.

    Adds no fields. Registration exists only because dispatch walks the spec's
    MRO, so an unregistered subclass silently keeps the parent manager and the
    trim entry point is never reachable.
    """


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


def rotate_cached_keys(
    k_pool: torch.Tensor,
    *,
    block_ids: list[int],
    positions: torch.Tensor,
    plan: PositionReanchor,
    inv_freq: torch.Tensor,
) -> int:
    """Re-RoPE the retained tail where it stands; return the rows touched.

    Args:
        k_pool: ``(num_blocks, block_size, num_kv_heads, head_dim)``, the layer's
            key cache as the attention backend sees it.
        block_ids: the request's block table *after*
            :meth:`MiniCPMO45DuplexWindowManager.reanchor_block_table`.
        positions: the retained tail's positions *before* the shift -- what the
            caller's slot bookkeeping still reports until it applies
            ``reanchor_positions`` itself.
        plan: the re-anchor, i.e. the uniform ``-delta`` translation.
        inv_freq: this layer group's RoPE frequencies, ``(head_dim // 2,)``.

    Each row is read and written at the slot it already occupies: ``delta`` is a
    whole number of pages, so a tail token's ``(page, offset)`` pair survives the
    renumbering and only its angle changes. That is the entire reason a trim can
    cost a memory pass instead of a forward pass. V needs nothing, because V was
    never rotated.

    The caller checks :func:`assert_uniform_position_shift` on the request's real
    position tensor first: this function is per layer and cannot see whether
    MRoPE split the tail across rows.
    """
    if positions.numel() == 0:
        return 0
    positions = positions.to(dtype=torch.long)
    if bool((positions < plan.moved_from).any()):
        raise RuntimeError(f"re-anchor given {int((positions < plan.moved_from).sum())} positions below moved_from")
    block_size = k_pool.shape[1]
    # Post-trim table and post-trim position resolve to the pre-trim slot: the
    # deletion shifted the index by exactly as many pages as the position moved,
    # so this is the row the token was written into, not a copy of it.
    slots = compute_slot_mapping(block_ids, positions - plan.delta, block_size)
    flat = k_pool.reshape(-1, *k_pool.shape[2:])
    # Advanced indexing copies on the way out, so the write below does not alias
    # the rows still being read.
    flat[slots] = rotate_keys(flat[slots], plan.delta, inv_freq)
    return int(slots.numel())


class MiniCPMO45DuplexSchedulerHelper:
    """Scheduler-side window planning and request compaction helper for MiniCPM-o 4.5 duplex."""

    @classmethod
    def find_duplex_window_manager(cls, scheduler: Any) -> MiniCPMO45DuplexWindowManager | None:
        coordinator = getattr(getattr(scheduler, "kv_cache_manager", None), "coordinator", None)
        if coordinator is None:
            return None
        for mgr in getattr(coordinator, "single_type_managers", ()):
            if isinstance(mgr, MiniCPMO45DuplexWindowManager):
                return mgr
        return None

    @classmethod
    def maybe_reanchor_session(
        cls,
        scheduler: Any,
        session: Any,
        update: Any,
    ) -> PositionReanchor | None:
        """Evaluate watermark policy, compact KV block table and session tokens if needed."""
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
        if mode not in {"basic", "context"}:
            return None

        base_len = int(getattr(session, "num_computed_tokens", 0) or 0)
        prefix_tokens = int(runtime_config.get("duplex_window_prefix_tokens", 96) or 96)
        if mode == "basic":
            high_watermark = int(window.get("basic_window_high_tokens", 8000) or 8000)
            low_watermark = int(window.get("basic_window_low_tokens", 6000) or 6000)
        else:
            max_units = int(window.get("context_max_units", 24) or 24)
            low_watermark = prefix_tokens + max_units * 12
            high_watermark = low_watermark + 500

        block_size = int(getattr(scheduler.cache_config, "block_size", 16) or 16)
        max_model_len = int(scheduler.model_config.max_model_len)

        geometry = DuplexWindowGeometry(
            prefix_tokens=prefix_tokens,
            window_tokens=low_watermark,
            block_size=block_size,
            max_model_len=max_model_len,
            high_watermark_tokens=high_watermark,
        )

        pending_tokens = len(getattr(update, "prompt_token_ids", []) or [])
        plan = plan_position_reanchor(
            geometry,
            computed_tokens=base_len,
            pending_tokens=pending_tokens,
        )
        if plan is None:
            return None

        duplex_mgr = cls.find_duplex_window_manager(scheduler)
        if duplex_mgr is None:
            return None

        freed_tokens = duplex_mgr.reanchor_block_table(session.request_id, plan)
        if freed_tokens == 0:
            return None

        old_computed = session.num_computed_tokens
        sink_end = plan.sink_end
        moved_from = plan.moved_from

        # Compact session token sequences and counts consistently
        if getattr(session, "prompt_token_ids", None) is not None and len(session.prompt_token_ids) >= moved_from:
            session.prompt_token_ids = session.prompt_token_ids[:sink_end] + session.prompt_token_ids[moved_from:]
            session.num_prompt_tokens = len(session.prompt_token_ids)

        if getattr(session, "_all_token_ids", None) is not None and len(session._all_token_ids) >= moved_from:
            session._all_token_ids = session._all_token_ids[:sink_end] + session._all_token_ids[moved_from:]

        session.num_computed_tokens -= plan.delta

        duplex["stage0_reanchor"] = {
            "delta": plan.delta,
            "moved_from": plan.moved_from,
            "sink_blocks": plan.sink_blocks,
            "old_computed_tokens": old_computed,
        }
        return plan


class MiniCPMO45DuplexWorkerHelper:
    """Worker-side KV cache rotation and position metadata helper for MiniCPM-o 4.5 duplex."""

    @classmethod
    def get_rope_inv_freq(cls, runner: Any) -> torch.Tensor:
        inv_freq = getattr(runner, "_duplex_inv_freq", None)
        if inv_freq is None:
            head_dim = runner.model_config.get_head_size()
            base = float(getattr(runner.model_config.hf_config, "rope_theta", 1000000.0) or 1000000.0)
            inv_freq = 1.0 / (
                base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=runner.device) / head_dim)
            )
            runner._duplex_inv_freq = inv_freq
        return inv_freq

    @classmethod
    def maybe_apply_reanchor(cls, runner: Any) -> None:
        """Apply in-place KV reanchor and rotation on worker before model forward."""
        if not hasattr(runner, "input_batch") or runner.input_batch is None:
            return
        num_reqs = runner.input_batch.num_reqs
        req_ids = runner.input_batch.req_ids[:num_reqs]
        for req_idx, req_id in enumerate(req_ids):
            info = runner.model_intermediate_buffer.get(req_id)
            if not isinstance(info, dict):
                continue
            duplex = info.get("duplex")
            if not isinstance(duplex, dict):
                continue
            reanchor = duplex.pop("stage0_reanchor", None)
            if reanchor is None:
                continue

            plan = PositionReanchor(
                delta=reanchor["delta"],
                moved_from=reanchor["moved_from"],
                sink_blocks=reanchor["sink_blocks"],
            )

            # Scheduler is authoritative for logical state (block_table and computed tokens).
            # The parent runner's _update_states() already installed the post-compaction block IDs
            # and decremented num_computed_tokens_cpu. We do NOT double-compact or double-decrement here.
            old_computed = int(
                reanchor.get(
                    "old_computed_tokens",
                    int(runner.input_batch.num_computed_tokens_cpu[req_idx]) + plan.delta,
                )
            )

            req_state = runner.requests.get(req_id) if hasattr(runner, "requests") else None
            if req_state is not None and getattr(req_state, "block_ids", None):
                compacted_block_ids = list(req_state.block_ids)
            else:
                bt = runner.input_batch.block_table
                bt_row = getattr(bt, "block_tables", [bt])[0]
                num_blocks = int(bt_row.num_blocks_per_row[req_idx])
                compacted_block_ids = list(bt_row.block_table.np[req_idx, :num_blocks])

            mrope_pos = getattr(req_state, "mrope_positions", None) if req_state is not None else None
            if mrope_pos is not None:
                assert_uniform_position_shift(mrope_pos, plan.moved_from)
                sink_tokens = plan.sink_blocks * runner.cache_config.block_size
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
                inv_freq = cls.get_rope_inv_freq(runner)
                for kv_cache in runner.kv_caches:
                    k_pool = kv_cache[0] if getattr(kv_cache, "dim", lambda: 0)() == 5 else kv_cache
                    rotate_cached_keys(
                        k_pool,
                        block_ids=compacted_block_ids,
                        positions=positions,
                        plan=plan,
                        inv_freq=inv_freq,
                    )
