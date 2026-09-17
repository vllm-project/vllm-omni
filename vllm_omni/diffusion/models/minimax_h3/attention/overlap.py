# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3 VSA execution scope: Q preparation, coarse attention, and reverse-output overlap.

All producer tickets share one per-invocation lifetime. Layout and numerical
contracts remain VSA-specific; the Ulysses transport is provided by shared code.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
import torch.distributed as dist
from vllm.logger import init_logger

logger = init_logger(__name__)
_BUNDLE_LOGGED = False
_GATE_LOGGED = False

ACTIVE: ContextVar[dict[str, Any] | None] = ContextVar("minimax_h3_overlap", default=None)


@lru_cache(None)
def streams(device):
    return torch.get_device_module().Stream(device=device, priority=-1), torch.get_device_module().Stream(device=device)


def run(
    attention,
    query,
    key,
    value,
    metadata,
    projector,
    *,
    gate_input=None,
    gate_projector=None,
    vsplit_input=None,
    vsplit_projection=None,
    vsplit_layer_index=None,
):
    assert ACTIVE.get() is None, "overlap forward cannot be nested"
    impl = attention.attention
    assert impl.h3_kernel_backend == "flashinfer" and impl.topk == 162
    assert query.shape == key.shape == value.shape == (1, 11992, 56, 128)
    assert query.dtype == key.dtype == value.dtype == torch.bfloat16
    caller = torch.get_device_module().current_stream()
    comm, side = streams(query.device)
    comm.wait_stream(caller)
    state = dict(comm=comm, side=side, projector=projector, impl=impl, prepared=None, reverse=None, projected=False)
    token = ACTIVE.set(state)
    try:
        with torch.get_device_module().stream(comm), torch.cuda.nvtx.range("h3.overlap.attention"):
            if vsplit_input is not None:
                from vllm_omni.diffusion.models.minimax_h3.attention.qkv_overlap import begin

                begin(state, attention, vsplit_projection, vsplit_input, value, vsplit_layer_index, metadata)
            if gate_input is not None:
                assert gate_input.shape == (11992, 5376) and gate_input.dtype == torch.bfloat16
                assert not gate_input.requires_grad and gate_input.is_contiguous()
                assert gate_projector is not None and "gate_compress" not in metadata.extra
                assert "vsa_h3_prefix_segments" in metadata.extra
                # Q/K norm+RoPE have already been submitted to caller. Gate
                # uses exactly the original GEMM on the same side stream as
                # QK preparation and reverse gate/O. It has no dependency on
                # received Q/K; preprocessing only inspects its shape/dtype.
                side.wait_stream(comm)
                ticket = state.get("vsplit_ticket")
                if ticket is not None:
                    assert ticket.launched and ticket.done is not None
                    with torch.cuda.nvtx.range("h3.vsplit.gate_after_v_ready"):
                        side.wait_event(ticket.done)
                with torch.get_device_module().stream(side), torch.cuda.nvtx.range("h3.lossless.gate_gemm"):
                    gate_input.record_stream(side)
                    gate_result = gate_projector(gate_input)
                    gate = gate_result[0] if isinstance(gate_result, tuple) else gate_result
                    assert gate.shape == (11992, 7168) and gate.dtype == torch.bfloat16
                    metadata.extra["gate_compress"] = gate.view(1, 11992, 56, 128)
                global _GATE_LOGGED
                if not _GATE_LOGGED:
                    logger.info(
                        "H3_LOSSLESS_GATE rank=%d M=11992 N=7168 K=5376 dtype=bf16 "
                        "shape_unchanged=1 stream=overlap_side",
                        dist.get_rank(),
                    )
                    _GATE_LOGGED = True
            out = attention(query, key, value, metadata)
            assert state["projected"], "chunked reverse/projector did not run"
            if vsplit_input is not None:
                from vllm_omni.diffusion.models.minimax_h3.attention.qkv_overlap import finish

                finish(state)
        caller.wait_stream(comm)
        out.record_stream(caller)
        return out
    finally:
        # Also cover exceptions before the normal preparation/reverse joins.
        if vsplit_input is not None:
            from vllm_omni.diffusion.models.minimax_h3.attention.qkv_overlap import cleanup

            cleanup(state, caller)

        drain_output_producer(state, caller)
        caller.wait_stream(side)
        caller.wait_stream(comm)
        ACTIVE.reset(token)


def before_v(query, key, value, metadata, group):
    active = ACTIVE.get()
    if active is None:
        return value
    if active.get("vsplit_ticket") is not None:
        from vllm_omni.diffusion.models.minimax_h3.attention.qkv_overlap import before_v as split_before_v

        return split_before_v(active, query, key, value, metadata, group)
    from vllm_omni.diffusion.distributed.flashinfer_ulysses import _state_for
    from vllm_omni.diffusion.models.minimax_h3.attention.vsa import (
        _build_h3_ordered_q2k_indices,
        _get_h3_layout,
        _get_h3_tile_metadata,
        _get_h3_tiled_source_rows,
        _pool_h3_tiles,
        h3_vsa_tile_pack,
    )

    state = _state_for(value, group, 8)
    assert state.communicator is not None
    out = state.output(value, op="scatter_heads", slot="v")
    landing = state.communicator.input_buffer(out, value.shape)
    landing.copy_(value)
    layout = _get_h3_layout(metadata)
    assert layout == ((558, 1206), (107, 22, 40), 1764), layout
    prefix, shape, _ = layout
    active["side"].wait_stream(active["comm"])
    with torch.get_device_module().stream(active["side"]), torch.cuda.nvtx.range("h3.overlap.qk_prepare"):
        _, sizes, _, _, prefix_blocks, video_blocks = _get_h3_tile_metadata(prefix, shape, query.device)
        maps = _get_h3_tiled_source_rows(prefix, shape, 105472, query.device)
        q_ready = prepared_q()
        q = h3_vsa_tile_pack(query[:, :95924], maps) if q_ready is None else q_ready[0]
        k = h3_vsa_tile_pack(key[:, :95924], maps)
        qp = _pool_h3_tiles(q, sizes) if q_ready is None else q_ready[1]
        kp = _pool_h3_tiles(k, sizes)
        scores = torch.matmul(qp, kp.transpose(-2, -1)) * active["impl"].softmax_scale
        idx, num = _build_h3_ordered_q2k_indices(scores, prefix_blocks, video_blocks, 162)
        active["prepared"] = (q, k, qp, kp, scores, idx, num)
    return landing


def prepared():
    active = ACTIVE.get()
    if active is None:
        return None
    result = active["prepared"]
    assert result is not None, "forward QK preparation hook did not run"
    active["comm"].wait_stream(active["side"])
    for tensor in result:
        tensor.record_stream(active["comm"])
    return result


def publish_reverse(fine, coarse, plan):
    active = ACTIVE.get()
    if active is None:
        return False

    join_coarse(coarse)
    assert active["reverse"] is None
    active["reverse"] = (fine, coarse, plan)
    return True


def finish_reverse(attn_output, ctx):
    active = ACTIVE.get()
    if active is None:
        return None
    from vllm_omni.diffusion.distributed.flashinfer_ulysses import _state_for
    from vllm_omni.diffusion.models.minimax_h3.ops.attention.chunks import gate_chunk
    from vllm_omni.diffusion.models.minimax_h3.ops.attention.owner_route import h3_vsa_owner_route_plan_is_trusted

    fine, coarse, plan = active["reverse"]
    assert fine is attn_output
    assert set(ctx.o_bundle_state) == {"plan"} and ctx.o_bundle_state["plan"] is plan
    assert ctx.strict_a2a_backend == "flashinfer-pcie" and ctx.joint_len == 0 and not ctx.use_uaa
    assert h3_vsa_owner_route_plan_is_trusted(plan)
    assert plan.local_rows == 11992 and plan.kmax == 270 and plan.sp_world_size == 8
    gate = ctx.o_bundle_gate_local
    assert gate is not None and gate.shape == (1, 11992, 56, 128)
    rank = dist.get_rank(ctx.ulysses_pg)
    state = _state_for(fine, ctx.ulysses_pg, 8)
    comm = state.communicator
    assert comm is not None
    # Cache these maps on the trusted, geometry-bound route's CPU tensors.
    tiles, row_map = route_maps(plan, rank, fine.device)
    count = 2998
    result = torch.empty(1, 11992, 5376, device=fine.device, dtype=torch.bfloat16)

    prepare_output_chunks(active, fine, coarse, tiles, state)
    for index in range(4):
        start = index * count
        out, landing = consume_output_chunk(active, index)
        with torch.cuda.nvtx.range(f"h3.overlap.o_rdma.{index}"):
            comm.gather_heads(landing, out=out)
        active["side"].wait_stream(active["comm"])
        with (
            torch.get_device_module().stream(active["side"]),
            torch.cuda.nvtx.range(f"h3.overlap.o_gate_project.{index}"),
        ):
            view = gate_chunk(out, gate[:, start : start + count], row_map[start : start + count], count)
            projected = active["projector"](view)
            result[:, start : start + count].copy_(projected)
    active["comm"].wait_stream(active["side"])
    result.record_stream(active["side"])
    gate.record_stream(active["side"])
    finish_output_chunks(active)
    active["projected"] = True
    # Separate markers bind the validator to the chunked protocol and its
    # actual volume; they do not impersonate the original whole-bundle path.
    if not state.o_producer_direct_logged:
        logger.info(
            "H3_OVERLAP_EXCHANGE rank=%d group=%s chunks=4 rows=2998 tail=270 "
            "remote_bytes=163975168 direct=1 completed=1",
            rank,
            ctx.ulysses_pg.group_name,
        )
        state.o_producer_direct_logged = True
    state.o_producer_direct_expected = None
    global _BUNDLE_LOGGED
    if not _BUNDLE_LOGGED:
        logger.info(
            "H3_OVERLAP_BUNDLE rank=%d fine_rows=%d local_rows=11992 coarse_rows=1648 "
            "chunks=4 tail=270 gate=bf16 projector=active",
            rank,
            fine.shape[1],
        )
        _BUNDLE_LOGGED = True
    return result


_ROUTE_MAPS = {}


def route_maps(plan, rank, device):
    key = (id(plan), rank, device)
    if key not in _ROUTE_MAPS:
        rows = plan.local_row_to_tail[rank]
        _ROUTE_MAPS[key] = (plan, plan.owner_tile_ids.to(device), rows.to(device))
    saved, tiles, row_map = _ROUTE_MAPS[key]
    assert saved is plan
    return tiles, row_map


_LOGGED: set[tuple[int, str]] = set()
_PREFIX = "VLLM_OMNI_H3_LOSSLESS_"


def _enabled(feature: str) -> bool:
    return os.environ.get(_PREFIX + feature, "0") == "1"


def _log(feature: str) -> None:
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    key = (rank, feature)
    if key not in _LOGGED:
        logger.info("H3_SCHEDULE_V3_FEATURE rank=%d feature=%s enabled=1", rank, feature)
        _LOGGED.add(key)


def layout_view_enabled() -> bool:
    enabled = _enabled("LAYOUT_VIEW")
    if enabled:
        _log("layout_view")
    return enabled


def after_q(query: torch.Tensor, metadata: Any) -> None:
    active = ACTIVE.get()
    if not _enabled("EARLY_Q") or active is None:
        return
    # The V-first candidate submits Q-only and K-dependent preparation
    # together after K exchange, leaving K transport to gate projection.
    if active.get("vsplit_ticket") is not None and not active.get("vsplit_k_ready", False):
        return
    from vllm_omni.diffusion.attention.ops.sage_quantization import quantize_sage_q_sm120
    from vllm_omni.diffusion.models.minimax_h3.attention.vsa import (
        _get_h3_layout,
        _get_h3_tile_metadata,
        _get_h3_tiled_source_rows,
        _pool_h3_tiles,
        h3_vsa_tile_pack,
    )

    assert query.shape == (1, 95936, 7, 128) and query.dtype == torch.bfloat16
    assert query.is_cuda and not query.requires_grad
    assert torch.get_device_module().current_stream(query.device) == active["comm"]
    assert "v3_q_ready" not in active, "Q preparation must run once per layer"
    layout = _get_h3_layout(metadata)
    assert layout == ((558, 1206), (107, 22, 40), 1764), layout
    prefix, shape, _ = layout
    side = active["side"]
    side.wait_stream(active["comm"])
    with torch.get_device_module().stream(side), torch.cuda.nvtx.range("h3.lossless.v3.early_q"):
        query.record_stream(side)
        _, sizes, _, _, _, _ = _get_h3_tile_metadata(prefix, shape, query.device)
        maps = _get_h3_tiled_source_rows(prefix, shape, 105472, query.device)
        q = h3_vsa_tile_pack(query[:, :95924], maps)
        qp = _pool_h3_tiles(q, sizes)
        q_int8, q_scale = quantize_sage_q_sm120(q.transpose(1, 2))
        active["v3_q_ready"] = dict(q=q, qp=qp, q_int8=q_int8, q_scale=q_scale)
    _log("early_q")


def prepared_q():
    active = ACTIVE.get()
    if active is None:
        return None
    ready = active.get("v3_q_ready")
    if ready is None:
        return None
    return ready["q"], ready["qp"]


def quantized_q(query: torch.Tensor):
    active = ACTIVE.get()
    if active is None:
        return None
    ready = active.get("v3_q_ready")
    if ready is None:
        return None
    q = ready["q"]
    assert (
        query.data_ptr() == q.data_ptr()
        and query.shape == q.shape
        and query.stride() == q.stride()
        and query.dtype == q.dtype
        and query.device == q.device
    ), "prepared Q identity changed before Sage quantization"
    assert torch.get_device_module().current_stream(query.device) == active["comm"]
    # prepared() already enqueued the producer-stream join. Keep the tensors
    # in this invocation's state and protect their use on the consumer stream.
    q_int8, q_scale = ready["q_int8"], ready["q_scale"]
    q_int8.record_stream(active["comm"])
    q_scale.record_stream(active["comm"])
    return q_int8, q_scale


@dataclass
class _CoarseTicket:
    active_id: int
    ready: Any
    inputs: tuple[torch.Tensor, torch.Tensor]
    used: bool = False


def coarse_ready(v_tiled: torch.Tensor, scores: torch.Tensor):
    active = ACTIVE.get()
    if not _enabled("COARSE_OVERLAP") or active is None:
        return None
    # ACTIVE is scoped to the fixed geometry, owner-local gate/O bundle route.
    prepared = active["prepared"]
    assert prepared is not None and prepared[4] is scores
    assert active["reverse"] is None
    assert v_tiled.shape == (1, 105472, 7, 128) and v_tiled.dtype == torch.bfloat16
    assert scores.shape == (1, 7, 1648, 1648) and scores.dtype == torch.float32
    assert v_tiled.device == scores.device
    assert torch.get_device_module().current_stream(v_tiled.device) == active["comm"]
    assert "v3_coarse_ticket" not in active, "coarse preparation must run once per layer"
    ready = torch.get_device_module().Event(enable_timing=False)
    ready.record(active["comm"])
    v_tiled.record_stream(active["side"])
    scores.record_stream(active["side"])
    ticket = _CoarseTicket(id(active), ready, (v_tiled, scores))
    active["v3_coarse_ticket"] = ticket
    _log("coarse_overlap")
    return ticket


@contextmanager
def coarse_scope(ticket):
    if ticket is None:
        yield
        return
    active = ACTIVE.get()
    assert isinstance(ticket, _CoarseTicket) and active is not None and ticket.active_id == id(active)
    assert not ticket.used, "coarse ticket must be consumed once"
    ticket.used = True
    side = active["side"]
    # The event was recorded before fine attention. Waiting on the current
    # comm stream here would also wait for fine and eliminate the overlap.
    side.wait_event(ticket.ready)
    with torch.get_device_module().stream(side), torch.cuda.nvtx.range("h3.lossless.v3.coarse"):
        try:
            yield
        finally:
            done = torch.get_device_module().Event(enable_timing=False)
            done.record(side)
            active["v3_coarse_done"] = done


def join_coarse(compressed: torch.Tensor) -> None:
    active = ACTIVE.get()
    if active is None:
        return
    ticket = active.get("v3_coarse_ticket")
    if ticket is None:
        return
    assert ticket.used and "v3_coarse_done" in active
    assert torch.get_device_module().current_stream(compressed.device) == active["comm"]
    active["comm"].wait_event(active["v3_coarse_done"])
    compressed.record_stream(active["comm"])


OUTPUT_LOOKAHEAD_ENV = "VLLM_OMNI_H3_O_PRODUCER_LOOKAHEAD"


@lru_cache(None)
def output_producer_stream(device):
    return torch.get_device_module().Stream(device=device, priority=-1)


def prepare_output_chunks(active, fine, coarse, tiles, state):
    from vllm_omni.diffusion.models.minimax_h3.ops.attention.chunks import produce

    if os.environ.get(OUTPUT_LOOKAHEAD_ENV) != "1" or "o_lookahead" in active:
        raise RuntimeError("This arm requires the fixed four-chunk O lookahead route")
    assert torch.get_device_module().current_stream(fine.device) == active["comm"]
    assert fine.dtype == coarse.dtype == torch.bfloat16
    assert tuple(fine.shape) == (1, 95936, 7, 128)
    assert tuple(coarse.shape) == (1, 1648, 7, 128)
    assert dist.get_world_size() == 8 and state.communicator is not None
    count = 2998
    shape = (1, 8 * (count + 270), 7, 128)
    entries = []
    for index in range(4):
        prototype = state.registered_shape_prototype(shape, dtype=fine.dtype, device=fine.device)
        out = state.output(prototype, op="gather_heads", slot=f"h3_o_chunk4_{index}")
        landing = state.communicator.input_buffer(out, shape)
        assert landing.is_contiguous() and out.is_contiguous()
        entries.append(dict(out=out, landing=landing, ready=None))
    # Concurrent writes must use separate registered source slots and must not
    # overlap any receive buffer consumed by gate/projection on the side stream.
    ranges = sorted(
        (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for e in entries for t in (e["landing"], e["out"])
    )
    assert all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:])), "O registered ranges overlap"
    stream = output_producer_stream(fine.device)
    ticket = dict(stream=stream, entries=entries, consumed=0)
    active["o_lookahead"] = ticket  # Install before submitting anything: cleanup also covers exceptions.
    stream.wait_stream(active["comm"])  # fine and the original coarse join precede this event.
    for tensor in (fine, coarse, tiles):
        tensor.record_stream(stream)
    with torch.get_device_module().stream(stream):
        for index, e in enumerate(entries):
            e["landing"].record_stream(stream)
            with torch.cuda.nvtx.range(f"h3.oproducer.lookahead.produce.{index}"):
                produce(fine, coarse, tiles, e["landing"], index * count, count)
            ready = torch.get_device_module().Event(enable_timing=False)
            ready.record(stream)
            e["ready"] = ready
    return ticket


def consume_output_chunk(active, index):
    ticket = active["o_lookahead"]
    assert index == ticket["consumed"] and 0 <= index < 4
    entry = ticket["entries"][index]
    assert entry["ready"] is not None
    active["comm"].wait_event(entry["ready"])
    entry["landing"].record_stream(active["comm"])
    ticket["consumed"] += 1
    return entry["out"], entry["landing"]


def finish_output_chunks(active):
    if active["o_lookahead"]["consumed"] != 4:
        raise RuntimeError("O producer must be consumed once per output chunk")


def drain_output_producer(active, caller):
    ticket = active.get("o_lookahead")
    if ticket is not None:
        caller.wait_stream(ticket["stream"])
        for entry in ticket["entries"]:
            entry["landing"].record_stream(caller)


H3_VSA_O_BUNDLE_ENV = "VLLM_OMNI_FASTVIDEO_VSA_O_BUNDLE"
H3_VSA_O_BUNDLE_ACTIVE_KEY = "vsa_h3_o_bundle_active"
H3_VSA_O_BUNDLE_STATE_KEY = "vsa_h3_o_bundle_state"


def h3_vsa_o_bundle_enabled() -> bool:
    """Return whether reverse-O compact-coarse piggybacking was requested."""
    raw = os.environ.get(H3_VSA_O_BUNDLE_ENV, "0").strip()
    if raw not in {"0", "1"}:
        raise ValueError(f"{H3_VSA_O_BUNDLE_ENV} must be exactly '0' or '1', got {raw!r}")
    return raw == "1"
