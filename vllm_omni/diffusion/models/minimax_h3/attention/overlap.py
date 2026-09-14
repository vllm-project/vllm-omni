# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in, geometry-bound real-model integration of the validated RDMA schedule."""

from contextvars import ContextVar
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
        from vllm_omni.diffusion.models.minimax_h3.attention.output_overlap import cleanup as o_cleanup

        o_cleanup(state, caller)
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
    from vllm_omni.diffusion.models.minimax_h3.attention.backend import (
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
        from vllm_omni.diffusion.models.minimax_h3.attention.schedule import prepared_q

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
    from vllm_omni.diffusion.models.minimax_h3.attention.schedule import join_coarse

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
    from vllm_omni.diffusion.models.minimax_h3.attention.output_overlap import complete as complete_o
    from vllm_omni.diffusion.models.minimax_h3.attention.output_overlap import consume as consume_o
    from vllm_omni.diffusion.models.minimax_h3.attention.output_overlap import prepare as prepare_o

    prepare_o(active, fine, coarse, tiles, state)
    for index in range(4):
        start = index * count
        out, landing = consume_o(active, index)
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
    complete_o(active)
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
