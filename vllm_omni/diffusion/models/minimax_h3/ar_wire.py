# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Quantised TP all-reduce for the MiniMax-H3 diffusion lane.

Same wire idea as the Ulysses a2a (`h3_a2a_wire.py`) applied to the TP reduction, which is the
single biggest line in the step (918 ms/step, 106 calls x 214 MB/rank, at the PCIe wire limit).

Scheme (world == 2):

    y = x_local + dequant(quant(x_peer))        # each rank keeps its own contribution exact

so the error is only the peer's quantisation error: one rounding of the peer's tensor, on a path
that already runs quantised activations. The saving is in the collective - one of the largest lines
in the step at this shape - and it has to be measured on the target wire, because the in-situ
collective is cheaper than the isolated benchmark predicts.

Env:
    H3_AR_WIRE=int8|bf16        default bf16 (== the original behaviour, byte-identical)
    H3_AR_WIRE_STATS=1          log every call
    H3_AR_WIRE_RPP / _WARPS     launch geometry (defaults 32 / 8; measured flat)

Any failure, unexpected dtype/shape, or world != 2 falls back to the caller's original all-reduce
and logs once. Never breaks the lane.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist
from vllm.triton_utils import tl, triton

from . import a2a_wire as w

logger = logging.getLogger(__name__)

TAG = "[h3_ar_wire]"
_STATE = {"calls": 0, "failed": False, "logged_mode": None}


def _log(msg: str) -> None:
    logger.info("%s %s", TAG, msg)


def _wire() -> str:
    """All-reduce mode, from ``H3_AR_WIRE`` (default "bf16", the unchanged code path)."""
    return os.environ.get("H3_AR_WIRE", "bf16").strip().lower()


def enabled() -> bool:
    return _wire() == "int8" and not _STATE["failed"]


@triton.jit
def _decode_add_kernel(
    packet_ptr, local_ptr, out_ptr, records, RECORDS: tl.constexpr, PACKET: tl.constexpr, VALUE_OFFSET: tl.constexpr
):
    """(records, PACKET) uint8 + (records, 128) bf16 -> (records, 128) bf16 = local + decoded."""
    record = tl.program_id(0) * RECORDS + tl.arange(0, RECORDS)
    columns = tl.arange(0, 128)
    valid = record[:, None] < records
    base = record[:, None] * PACKET
    stored = tl.load(packet_ptr + base + columns[None, :], mask=valid, other=VALUE_OFFSET).to(tl.int32)
    group = columns // 32
    code = tl.load(packet_ptr + base + 128 + group[None, :], mask=valid, other=0).to(tl.int32)
    scale_bits = (((code >> 3) - 15 + 127) << 23) | ((code & 7) << 20)
    scale = scale_bits.to(tl.float32, bitcast=True)
    dec = (stored - VALUE_OFFSET).to(tl.float32) * scale
    loc = tl.load(local_ptr + record[:, None] * 128 + columns[None, :], mask=valid, other=0.0)
    tl.store(out_ptr + record[:, None] * 128 + columns[None, :], (loc.to(tl.float32) + dec).to(tl.bfloat16), mask=valid)


def _tensor_group():
    from vllm.distributed.parallel_state import get_tp_group

    return get_tp_group()


_WS: dict = {}


def _workspace(records: int, device, world: int):
    """One grow-to-max workspace per device, sliced per call: bounded resident memory.

    Keying this by shape leaked: each distinct all-reduce shape allocated its own packet and gather
    buffers on workers that have almost no headroom, and the 106 calls per step exhausted the device
    (a 274 MiB allocation failed with 111 MiB free). One set per device, sized to the largest shape
    seen, allocates nothing per call; the slices stay contiguous, so the flat all-gather is valid.
    """
    key = (device.index, world)
    st = _WS.get(key)
    if st is None:
        st = {"cap": 0, "pkt": None, "recv": None}
        _WS[key] = st
    if records > st["cap"]:
        # release the previous set before asking for a bigger one
        st["pkt"] = st["recv"] = None
        torch.accelerator.empty_cache()
        st["pkt"] = torch.empty(records * w.OUTPUT_PACKET, dtype=torch.uint8, device=device)
        st["recv"] = torch.empty(world * records * w.OUTPUT_PACKET, dtype=torch.uint8, device=device)
        st["cap"] = records
        _log(
            f"AR workspace for {records} records/world={world}: pkt={records * w.OUTPUT_PACKET} B "
            f"recv={world * records * w.OUTPUT_PACKET} B"
        )
    pkt = st["pkt"][: records * w.OUTPUT_PACKET].view(records, w.OUTPUT_PACKET)
    recv = st["recv"][: world * records * w.OUTPUT_PACKET]
    return pkt, recv


@torch.compiler.disable
def quant_reduce(x: torch.Tensor, device_group=None) -> torch.Tensor:
    if device_group is None:
        group = _tensor_group()
        device_group = getattr(group, "device_group", None) or group
    world = dist.get_world_size(group=device_group)

    hidden = x.shape[-1]
    if x.dtype != torch.bfloat16 or not x.is_contiguous() or hidden % w.VECTOR != 0 or world != 2:
        raise ValueError(f"shape/dtype/world not supported for int8 AR: {tuple(x.shape)} {x.dtype} world={world}")

    flat = x.view(-1, w.VECTOR)
    records = flat.shape[0]
    pkt, recv = _workspace(records, x.device, world)

    w._encode_pack_kernel[(triton.cdiv(records, w._rpp()),)](
        flat,
        pkt,
        records,
        RECORDS=w._rpp(),
        PACKET=w.OUTPUT_PACKET,
        VALUE_OFFSET=w.VALUE_BIAS,
        num_warps=w._warps(),
    )
    dist.all_gather_into_tensor(recv, pkt.reshape(-1), group=device_group)

    rank = dist.get_rank(group=device_group)
    peer_pkt = recv.view(world, records * w.OUTPUT_PACKET)[1 - rank].view(records, w.OUTPUT_PACKET)

    # in place: vllm's tensor_model_parallel_all_reduce also reduces in place, so this keeps the
    # caller's contract and costs no output allocation. Each element reads then writes itself.
    _decode_add_kernel[(triton.cdiv(records, w._rpp()),)](
        peer_pkt,
        flat,
        flat,
        records,
        RECORDS=w._rpp(),
        PACKET=w.OUTPUT_PACKET,
        VALUE_OFFSET=w.VALUE_BIAS,
        num_warps=w._warps(),
    )

    if _STATE["calls"] == 0 or os.environ.get("H3_AR_WIRE_STATS", "0") == "1":
        bf16_bytes = x.numel() * x.element_size()
        pkt_bytes = pkt.numel()
        _log(
            f"int8 AR call {_STATE['calls']}: {tuple(x.shape)} bf16={bf16_bytes} B "
            f"packet={pkt_bytes} B ({pkt_bytes / max(bf16_bytes, 1):.3f}x) world={world} "
            f"rpp={w._rpp()} warps={w._warps()} ws={list(_WS)}"
        )
    _STATE["calls"] += 1
    return x


@torch.compiler.disable
def tp_all_reduce(x: torch.Tensor, original) -> torch.Tensor:
    """Drop-in for `tensor_model_parallel_all_reduce(x)` on the DiT's TP group.

    `original` is the caller's own all-reduce, used unchanged whenever int8 is off, unsupported, or
    has failed - so the default path stays byte-identical to the stock behaviour.

    Both this and `quant_reduce` are `@torch.compiler.disable`: the call site lives inside the
    torch.compile'd MXFP8 linear, so without that Dynamo traces this module's Python (workspace
    dict, env lookup, stats f-string) and raises - measured as
    `InternalTorchDynamoError: ValueError: Unknown format code 'f' for object of type 'str'`,
    which latched the fallback and silently served plain bf16 for the whole run.
    """
    on = enabled()
    mode = "int8" if on else "bf16"
    if _STATE["logged_mode"] != mode:
        _STATE["logged_mode"] = mode
        _log(f"TP all-reduce mode -> {mode}")
    if not on:
        return original(x)
    try:
        return quant_reduce(x)
    except Exception as exc:
        _STATE["failed"] = True
        _log(f"int8 AR FAILED ({type(exc).__name__}: {exc}) -> falling back to the original all-reduce")
        return original(x)
