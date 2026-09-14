# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in scheduling helpers for the fixed H3 SP8 reverse-O bundle experiment."""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from vllm.logger import init_logger

logger = init_logger(__name__)
_LOGGED: set[tuple[int, str]] = set()
_PREFIX = "VLLM_OMNI_H3_LOSSLESS_"


def _active():
    from vllm_omni.diffusion.models.minimax_h3.attention.overlap import ACTIVE

    return ACTIVE.get()


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
    active = _active()
    if not _enabled("EARLY_Q") or active is None:
        return
    # The V-first candidate submits Q-only and K-dependent preparation
    # together after K exchange, leaving K transport to gate projection.
    if active.get("vsplit_ticket") is not None and not active.get("vsplit_k_ready", False):
        return
    from vllm_omni.diffusion.attention.ops.sage_quantization import quantize_sage_q_sm120
    from vllm_omni.diffusion.models.minimax_h3.attention.backend import (
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
    active = _active()
    if active is None:
        return None
    ready = active.get("v3_q_ready")
    if ready is None:
        return None
    return ready["q"], ready["qp"]


def quantized_q(query: torch.Tensor):
    active = _active()
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
    active = _active()
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
    active = _active()
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
    active = _active()
    if active is None:
        return
    ticket = active.get("v3_coarse_ticket")
    if ticket is None:
        return
    assert ticket.used and "v3_coarse_done" in active
    assert torch.get_device_module().current_stream(compressed.device) == active["comm"]
    active["comm"].wait_event(active["v3_coarse_done"])
    compressed.record_stream(active["comm"])
