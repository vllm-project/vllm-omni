# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""QK/V producer scheduling for the supported H3 overlap geometry.

Only tensor arguments cross the regional-compile boundary. Tickets, streams,
events and producer state live entirely in the existing eager attention
island. The V placeholder aliases raw Q and must never be read as V: the
strictly guarded Ulysses path replaces it before its first V data access.
"""

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
import torch.distributed as dist
from vllm.logger import init_logger

logger = init_logger(__name__)
ROWS, HIDDEN, WIDTH = 11992, 5376, 7168
_AUDIT_CALLS: dict[int, int] = {}
_GLOBAL_AUDIT_CALL = 0
_ACTIVE_LOGGED = set()


def configured_mode(prefix: str) -> str:
    if os.getenv("VLLM_OMNI_H3_VSPLIT", "0") != "1" or not prefix.startswith("blocks."):
        return "off"
    mode = os.getenv("VLLM_OMNI_H3_VSPLIT_MODE", "split")
    if mode != "split" or os.environ.get("VLLM_OMNI_H3_MXFP8_VSPLIT") != "1":
        raise ValueError(f"Invalid H3 VSPLIT mode: {mode!r}")
    for flag in (
        "ATTENTION_OVERLAP",
        "LOSSLESS_GATE",
        "LOSSLESS_EARLY_Q",
        "LOSSLESS_LAYOUT_VIEW",
        "LOSSLESS_COARSE_OVERLAP",
    ):
        if os.getenv("VLLM_OMNI_H3_" + flag, "0") != "1":
            raise RuntimeError(f"H3 VSPLIT requires the existing v3 flag H3_{flag}=1")
    if os.getenv("VLLM_BATCH_INVARIANT", "0") != "0":
        raise RuntimeError("H3 VSPLIT does not replace batch-invariant GEMM")
    return mode


def validate_projection(projection, prepared) -> None:
    from vllm.distributed import get_tensor_model_parallel_world_size

    from vllm_omni.diffusion.layers.mxfp8 import _scale_numel
    from vllm_omni.diffusion.models.minimax_h3.mxfp8 import validate_split_projection

    if not dist.is_initialized() or dist.get_world_size() != 8 or get_tensor_model_parallel_world_size() != 1:
        raise RuntimeError("MXFP8 split requires initialized TP1/SP8")
    if not isinstance(prepared, tuple) or len(prepared) != 3:
        raise RuntimeError("MXFP8 split requires explicit quantized X, scale and QK tensors")
    x, scale, qk = prepared
    expected = (
        ((ROWS, HIDDEN), torch.float8_e4m3fn),
        ((_scale_numel(ROWS, HIDDEN),), torch.uint8),
        ((ROWS, 2 * WIDTH), torch.bfloat16),
    )
    for tensor, (shape, dtype) in zip(prepared, expected):
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or not tensor.is_cuda
            or tensor.device != projection.weight.device
            or not tensor.is_contiguous()
            or tensor.requires_grad
            or torch.is_grad_enabled()
        ):
            raise RuntimeError("MXFP8 split prepared tensor contract changed")
    validate_split_projection(projection)


@lru_cache(None)
def _v_stream(device_index: int):
    return torch.get_device_module().Stream(device=device_index)


@dataclass
class Ticket:
    x: torch.Tensor
    weight: torch.Tensor
    placeholder: torch.Tensor
    stream: Any
    x_ready: Any
    layer_index: int
    x_scale: torch.Tensor
    qk: torch.Tensor
    projection: Any
    result: Any = None
    done: Any = None
    launched: bool = False
    consumed: bool = False


def begin(active, attention, projection, prepared, placeholder, layer_index: int, metadata):
    validate_projection(projection, prepared)
    x, x_scale, qk = prepared
    validate_route(attention, metadata)
    if not 0 <= layer_index < 50 or tuple(placeholder.shape) != (1, ROWS, 56, 128):
        raise RuntimeError("H3 VSPLIT placeholder geometry changed")
    ready = torch.get_device_module().Event(enable_timing=False)
    ready.record(active["comm"])
    stream = _v_stream(x.device.index)
    for tensor in (x, x_scale, qk, projection.weight, projection.mxfp8_weight_scale):
        tensor.record_stream(stream)
    ticket = Ticket(x, projection.weight, placeholder, stream, ready, layer_index, x_scale, qk, projection)
    active["vsplit_ticket"] = ticket
    # V depends only on the prepared input. Launch before Q transport, then
    # hand the SMs to gate projection rather than running both large GEMMs
    # concurrently. Gate/early-Q/QK preparation are queued on the side stream.
    stream.wait_event(ready)
    ticket.launched = True  # Exception cleanup joins partial submissions.
    with torch.get_device_module().stream(stream):
        from vllm_omni.diffusion.models.minimax_h3.mxfp8 import project_split_v

        ticket.result = project_split_v(projection, x, x_scale, qk).view(1, ROWS, 56, 128)
        ticket.done = torch.get_device_module().Event(enable_timing=False)
        ticket.done.record(stream)
    key = (dist.get_rank(), layer_index)
    if key not in _ACTIVE_LOGGED:
        _ACTIVE_LOGGED.add(key)
        logger.info(
            "H3_VSPLIT_ACTIVE %s",
            json.dumps(
                dict(
                    schema=1,
                    rank=key[0],
                    layer_index=layer_index,
                    mode="split",
                    schedule="before_q_v_then_gate_qk_after_k",
                    v_stream="dedicated",
                    qk_shape=[ROWS, 2 * WIDTH],
                    v_shape=[ROWS, WIDTH],
                ),
                sort_keys=True,
            ),
        )


def validate_route(attention, metadata) -> None:
    from vllm_omni.diffusion.forward_context import get_ulysses_mode

    strategy = attention._get_active_parallel_strategy()
    if (
        strategy is not attention.parallel_strategy
        or strategy.name != "ulysses"
        or strategy._sp_group.ulysses_world_size != 8
        or strategy._sp_group.ring_world_size != 1
        or get_ulysses_mode(default="strict") != "strict"
        or strategy._ulysses_a2a_backend != "flashinfer-pcie"
        or not strategy._ulysses_a2a_permute
        or strategy._scatter_idx != 2
        or strategy._gather_idx != 1
        or not strategy._qk_input_landing_enabled
        or attention.use_ring
        or attention._active_paged_kv_adapter() is not None
        or any(getattr(metadata, name, None) is not None for name in ("joint_query", "joint_key", "joint_value"))
    ):
        raise RuntimeError("H3 VSPLIT requires the exact non-joint BF16 Q/K/V Ulysses SP8 route")


def after_q() -> None:
    from vllm_omni.diffusion.models.minimax_h3.attention.overlap import ACTIVE

    active = ACTIVE.get()
    ticket = None if active is None else active.get("vsplit_ticket")
    if ticket is None:
        return
    assert active is not None
    if not ticket.launched or ticket.done is None or "v3_q_ready" in active:
        raise RuntimeError("H3 VSPLIT requires V submitted before Q and Q preparation deferred until K")


def before_v(active, query, key, placeholder, metadata, group):
    """Schedule K-dependent work before joining the independent V producer."""
    from vllm_omni.diffusion.distributed.flashinfer_ulysses import _state_for
    from vllm_omni.diffusion.models.minimax_h3.attention.backend import (
        _build_h3_ordered_q2k_indices,
        _get_h3_layout,
        _get_h3_tile_metadata,
        _get_h3_tiled_source_rows,
        _pool_h3_tiles,
        h3_vsa_tile_pack,
    )
    from vllm_omni.diffusion.models.minimax_h3.attention.schedule import prepared_q

    ticket = active["vsplit_ticket"]
    if (
        not ticket.launched
        or ticket.done is None
        or ticket.consumed
        or placeholder.data_ptr() != ticket.placeholder.data_ptr()
        or placeholder.shape != ticket.placeholder.shape
        or placeholder.stride() != ticket.placeholder.stride()
    ):
        raise RuntimeError("H3 VSPLIT V ticket missing, repeated, or rebound")
    layout = _get_h3_layout(metadata)
    if layout != ((558, 1206), (107, 22, 40), 1764):
        raise RuntimeError("H3 VSPLIT layout changed")
    prefix, shape, _ = layout
    if active.get("vsplit_k_ready", False) or "v3_q_ready" in active:
        raise RuntimeError("Deferred Q preparation must be submitted once after K")
    active["vsplit_k_ready"] = True
    from vllm_omni.diffusion.models.minimax_h3.attention.schedule import after_q

    after_q(query, metadata)
    with torch.cuda.nvtx.range("h3.vsplit.qk_ready"):
        active["side"].wait_stream(active["comm"])
    with torch.get_device_module().stream(active["side"]), torch.cuda.nvtx.range("h3.overlap.qk_prepare"):
        _, sizes, _, _, prefix_blocks, video_blocks = _get_h3_tile_metadata(prefix, shape, query.device)
        maps = _get_h3_tiled_source_rows(prefix, shape, 105472, query.device)
        q_ready = prepared_q()
        if q_ready is None:
            raise RuntimeError("H3 VSPLIT requires prepared earlyQ")
        q, qp = q_ready
        k = h3_vsa_tile_pack(key[:, :95924], maps)
        kp = _pool_h3_tiles(k, sizes)
        scores = torch.matmul(qp, kp.transpose(-2, -1)) * active["impl"].softmax_scale
        idx, num = _build_h3_ordered_q2k_indices(scores, prefix_blocks, video_blocks, 162)
        active["prepared"] = (q, k, qp, kp, scores, idx, num)
    # This is the first V data access. Waiting here cannot delay the already
    # submitted side-stream QK preparation above.
    with torch.cuda.nvtx.range("h3.vsplit.v_ready_wait"):
        active["comm"].wait_event(ticket.done)
    value = ticket.result
    value.record_stream(active["comm"])
    state = _state_for(value, group, 8)
    if state.communicator is None:
        raise RuntimeError("H3 VSPLIT requires the active RDMA communicator")
    out = state.output(value, op="scatter_heads", slot="v")
    landing = state.communicator.input_buffer(out, value.shape)
    landing.copy_(value)
    ticket.consumed = True
    return landing


def finish(active) -> None:
    ticket = active.get("vsplit_ticket")
    if ticket is not None and not ticket.consumed:
        raise RuntimeError("H3 VSPLIT V hook was not consumed")


def cleanup(active, caller) -> None:
    ticket = active.get("vsplit_ticket")
    if ticket is not None:
        caller.wait_stream(ticket.stream)
        for tensor in (
            ticket.x,
            ticket.x_scale,
            ticket.qk,
            ticket.weight,
            ticket.projection.mxfp8_weight_scale,
            ticket.result,
        ):
            if tensor is not None:
                tensor.record_stream(caller)
