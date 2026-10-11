# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Env-gated int8 (UE5M3) transport for the H3 lane's Ulysses all-to-all. Default OFF.

Why: the Ulysses exchange is 632-648 ms/step (200 calls, 4 per block) at ~22 GB/s = PCIe Gen4 x16 wire
speed, i.e. it is byte-bound, so the only lever is fewer bytes. Sol-H3 ships `comm_quant` (int8/FP8
packets) for its own attention layout; this lane's `all_to_all_4D` exchanges a tensor whose last dim is
`hs`=128 with equal splits in both directions, so the same idea drops in as an encode -> exchange ->
decode around the existing `all_to_all_single`, with every surrounding permute untouched.

At this packet size, int8/UE5M3 packets are 0.5625x bf16 bytes at rel ~4-6e-3; raw FP8 is 0.5000x at
rel 5.2e-2, so int8 is the better trade and is the only mode wired here. Encoding and decoding add
two full-tensor passes per exchange (~81 ms/step of kernels), so the byte saving only pays when the
collective itself is wire-bound; the launch geometry is exposed for tuning.

    H3_A2A_WIRE=bf16|int8     (default bf16 -> unchanged code path, byte-identical clips)
    H3_A2A_WIRE_STATS=1       log byte/error stats on every call (default: first call only)
    H3_A2A_WIRE_BUFCACHE=1    reuse the intermediate buffers (default 0; see `_bufcache`)
    H3_A2A_WIRE_RPP=<int>     records per program for both kernels (default 32)
    H3_A2A_WIRE_WARPS=<int>   num_warps for both kernels (default 8)

The module never raises into the model: any failure logs once and falls back to the bf16 path.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist
from vllm.triton_utils import tl, triton

from .comm.comm_quant import (  # the ported Sol-H3 primitives (same packet format)
    OUTPUT_PACKET,
    VALUE_BIAS,
    VECTOR,
    _encode_ue5m3_int8,
)

logger = logging.getLogger(__name__)

TAG = "[h3_a2a_wire]"
_STATE = {"calls": 0, "fused_calls": 0, "failed": False}
_POOL: dict[tuple, torch.Tensor] = {}


def _wire() -> str:
    """Transport mode, from ``H3_A2A_WIRE`` (default "bf16", the unchanged code path)."""
    return os.environ.get("H3_A2A_WIRE", "bf16").strip().lower()


# Both the stock int8 mode and the fused variant keep the NON-fused directions (notably the
# reverse/o-path, which has no fused kernel) on int8. A mode string that only the qkv branch
# understood silently pushed the o-path back to bf16 - measured as 2400 bf16 transitions and a
# changed clip.
_INT8_MODES = ("int8", "int8-fused", "int8_fused", "fused")


def wire_enabled() -> bool:
    return _wire() in _INT8_MODES


def _bufcache() -> bool:
    # Default OFF. The pooled variant corrupts the payload (deterministic
    # 26,410,165 B colour mosaic while exit=0 and a2a_failed=0), so it must be opt-in;
    # serve_arwire.sh also exports 0 explicitly for the same reason.
    return os.environ.get("H3_A2A_WIRE_BUFCACHE", "0").strip().lower() in ("1", "true", "yes", "on")


def _rpp() -> int:
    return int(os.environ.get("H3_A2A_WIRE_RPP", "32"))


def _warps() -> int:
    return int(os.environ.get("H3_A2A_WIRE_WARPS", "8"))


def _log(msg: str) -> None:
    logger.info("%s %s", TAG, msg)


_SEEN_MODES: set = set()


def _log_mode_once(mode: str) -> None:
    """Log each distinct mode once.

    Not per transition: the qkv direction and the reverse/o direction can carry different mode
    labels, which made a transition log emit 2400 lines per boot and drown the real signal.
    """
    if mode not in _SEEN_MODES:
        _SEEN_MODES.add(mode)
        _log(f"transport mode -> {mode}")


def _buf(shape, dtype, device, tag: str = "x") -> torch.Tensor:
    """Pooled intermediate buffer, keyed by role (`tag`) as well as geometry.

    The role matters: encode output and the receive buffer can have identical shape/dtype, and aliasing
    them turns the exchange into a send-to-self (caught by the 5-iteration 2-rank test).
    """
    if not _bufcache():
        return torch.empty(shape, dtype=dtype, device=device)
    key = (tag, device.index, tuple(shape), dtype)
    buf = _POOL.get(key)
    if buf is None:
        buf = torch.empty(shape, dtype=dtype, device=device)
        _POOL[key] = buf
    return buf


@triton.jit
def _encode_pack_kernel(
    input_ptr, packet_ptr, records, RECORDS: tl.constexpr, PACKET: tl.constexpr, VALUE_OFFSET: tl.constexpr
):
    """(records, 128) bf16 -> (records, PACKET) uint8, order preserved (no rank merge)."""
    record = tl.program_id(0) * RECORDS + tl.arange(0, RECORDS)
    columns = tl.arange(0, 128)
    valid = record[:, None] < records
    values = tl.load(input_ptr + record[:, None] * 128 + columns[None, :], mask=valid, other=0.0)
    stored, scale_codes = _encode_ue5m3_int8(values, GROUPS=RECORDS * 4)
    stored = tl.reshape(stored, RECORDS, 128)
    scale_codes = tl.reshape(scale_codes, RECORDS, 4)
    base = record[:, None] * PACKET
    tl.store(packet_ptr + base + columns[None, :], stored, mask=valid)
    groups = tl.arange(0, 4)
    tl.store(packet_ptr + base + 128 + groups[None, :], scale_codes, mask=valid)
    padding = tl.arange(0, 16)
    tl.store(packet_ptr + base + 132 + padding[None, :], 0, mask=valid & (padding[None, :] < PACKET - 132))


@triton.jit
def _decode_pack_kernel(
    packet_ptr, output_ptr, records, RECORDS: tl.constexpr, PACKET: tl.constexpr, VALUE_OFFSET: tl.constexpr
):
    """(records, PACKET) uint8 -> (records, 128) bf16, order preserved."""
    record = tl.program_id(0) * RECORDS + tl.arange(0, RECORDS)
    columns = tl.arange(0, 128)
    valid = record[:, None] < records
    base = record[:, None] * PACKET
    stored = tl.load(packet_ptr + base + columns[None, :], mask=valid, other=VALUE_OFFSET).to(tl.int32)
    group = columns // 32
    code = tl.load(packet_ptr + base + 128 + group[None, :], mask=valid, other=0).to(tl.int32)
    scale_bits = (((code >> 3) - 15 + 127) << 23) | ((code & 7) << 20)
    scale = scale_bits.to(tl.float32, bitcast=True)
    decoded = (stored - VALUE_OFFSET).to(tl.float32) * scale
    tl.store(output_ptr + record[:, None] * 128 + columns[None, :], decoded, mask=valid)


def encode_pack(x: torch.Tensor) -> torch.Tensor:
    """(..., 128) bf16 -> (..., PACKET) uint8 (pooled buffer; consume before the next call)."""
    if x.dtype != torch.bfloat16 or not x.is_contiguous() or x.shape[-1] != VECTOR:
        raise ValueError(f"int8 transport needs a contiguous BF16 tensor with last dim {VECTOR}")
    records = x.numel() // VECTOR
    packet = _buf((*x.shape[:-1], OUTPUT_PACKET), torch.uint8, x.device, tag="pkt")
    if records:
        _encode_pack_kernel[(triton.cdiv(records, _rpp()),)](
            x,
            packet,
            records,
            RECORDS=_rpp(),
            PACKET=OUTPUT_PACKET,
            VALUE_OFFSET=VALUE_BIAS,
            num_warps=_warps(),
        )
    return packet


def decode_pack(packet: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """(..., PACKET) uint8 -> (..., 128) bf16 with `like`'s leading shape (pooled buffer)."""
    records = packet.numel() // OUTPUT_PACKET
    out = _buf((*like.shape[:-1], VECTOR), torch.bfloat16, packet.device, tag="dec")
    if records:
        _decode_pack_kernel[(triton.cdiv(records, _rpp()),)](
            packet,
            out,
            records,
            RECORDS=_rpp(),
            PACKET=OUTPUT_PACKET,
            VALUE_OFFSET=VALUE_BIAS,
            num_warps=_warps(),
        )
    return out


def round_trip_error(x: torch.Tensor) -> tuple[float, float]:
    """Local encode+decode error (no collective). Uses (and clobbers) the pooled buffers."""
    back = decode_pack(encode_pack(x), x)
    d = (back.float() - x.float()).abs().max().item()
    m = x.float().abs().max().item() or 1.0
    return d, d / m


def quant_all_to_all(input_t: torch.Tensor, group, world: int) -> torch.Tensor:
    """Encode -> equal-split uint8 all_to_all_single -> decode. Same shape/dtype as `input_t`."""
    _log_mode_once("int8")
    want_stats = _STATE["calls"] == 0 or os.environ.get("H3_A2A_WIRE_STATS", "0") == "1"
    d = rel = float("nan")
    if want_stats:
        # measure BEFORE the real encode: the pooled buffers are overwritten right after
        d, rel = round_trip_error(input_t)

    packet = encode_pack(input_t)
    recv = _buf(packet.shape, torch.uint8, packet.device, tag="recv")
    dist.all_to_all_single(recv.view(-1), packet.view(-1), group=group)
    out = decode_pack(recv, input_t)

    if want_stats:
        bf16_bytes = input_t.numel() * input_t.element_size()
        pkt_bytes = packet.numel()
        _log(
            f"int8 call {_STATE['calls']}: {tuple(input_t.shape)} bf16={bf16_bytes} B "
            f"packet={pkt_bytes} B ({pkt_bytes / max(bf16_bytes, 1):.3f}x) max|diff|={d:.3e} rel={rel:.2e} "
            f"rpp={_rpp()} warps={_warps()} pool={len(_POOL)}"
        )
    _STATE["calls"] += 1
    return out


def all_to_all_4d(input_t: torch.Tensor, group, seq_world_size: int) -> torch.Tensor:
    """Replacement for the `dist.all_to_all_single(output, input_t)` call in all_to_all_4D.

    Falls back to the original bf16 exchange on any failure (once, logged).
    """
    if not wire_enabled() or _STATE["failed"] or seq_world_size <= 1:
        _log_mode_once("bf16")
        out = torch.empty_like(input_t)
        dist.all_to_all_single(out, input_t, group=group)
        return out
    try:
        return quant_all_to_all(input_t, group, seq_world_size)
    except Exception as exc:  # never break the lane
        _STATE["failed"] = True
        _log(f"int8 transport FAILED ({type(exc).__name__}: {exc}) -> falling back to bf16 for this run")
        out = torch.empty_like(input_t)
        dist.all_to_all_single(out, input_t, group=group)
        return out


# ---------------------------------------------------------------------------
# Fused qkv transport: H3_A2A_WIRE=int8-fused (control file value "int8-fused")
# ---------------------------------------------------------------------------
# The stock int8 path performs two full-size bf16 layout passes around the wire
# (pre-transform into the transmitted layout, post-transform into (bs, seqlen, ...)).
# Both are pure reindexings of 128-element rows, so they can be folded into the
# encode/decode kernels: the encode reads the SOURCE rows directly and the decode
# writes the FINAL layout directly. Row contents and per-row scales are unchanged,
# so packets and decoded values are bit-identical to the stock path.


def fused_enabled() -> bool:
    """True when the wire mode asks for the fused qkv transport."""
    return _wire() in ("int8-fused", "int8_fused", "fused")


@triton.jit
def _encode_pack_qkv_fused_kernel(
    src_ptr,
    packet_ptr,
    records,
    SHARD_SEQLEN,
    BS,
    SHARD_HC,
    HC,
    RECORDS: tl.constexpr,
    PACKET: tl.constexpr,
    VALUE_OFFSET: tl.constexpr,
):
    """(bs, shard_seqlen, hc, hs) bf16 -> (P, shard_seqlen, bs, shard_hc, hs) packets.

    Records are walked in the TRANSMITTED order so the packet buffer matches the stock
    pre-transform exactly; the source row is looked up by inverting that mapping.
    """
    record = tl.program_id(0) * RECORDS + tl.arange(0, RECORDS)
    columns = tl.arange(0, 128)
    valid = record[:, None] < records
    head_local = record % SHARD_HC
    rest = record // SHARD_HC
    batch = rest % BS
    rest2 = rest // BS
    seq_local = rest2 % SHARD_SEQLEN
    peer = rest2 // SHARD_SEQLEN
    # source (bs, shard_seqlen, hc, hs) row for transmitted record (peer, seq_local, batch, head_local)
    source_row = ((batch * SHARD_SEQLEN + seq_local) * HC) + peer * SHARD_HC + head_local
    values = tl.load(src_ptr + source_row[:, None] * 128 + columns[None, :], mask=valid, other=0.0)
    stored, scale_codes = _encode_ue5m3_int8(values, GROUPS=RECORDS * 4)
    stored = tl.reshape(stored, RECORDS, 128)
    scale_codes = tl.reshape(scale_codes, RECORDS, 4)
    base = record[:, None] * PACKET
    tl.store(packet_ptr + base + columns[None, :], stored, mask=valid)
    groups = tl.arange(0, 4)
    tl.store(packet_ptr + base + 128 + groups[None, :], scale_codes, mask=valid)
    padding = tl.arange(0, 16)
    tl.store(packet_ptr + base + 132 + padding[None, :], 0, mask=valid & (padding[None, :] < PACKET - 132))


@triton.jit
def _decode_pack_qkv_fused_kernel(
    packet_ptr,
    output_ptr,
    records,
    SHARD_SEQLEN,
    BS,
    SHARD_HC,
    SEQLEN,
    RECORDS: tl.constexpr,
    PACKET: tl.constexpr,
    VALUE_OFFSET: tl.constexpr,
):
    """(P, shard_seqlen, bs, shard_hc, hs) packets -> (bs, seqlen, shard_hc, hs) bf16."""
    record = tl.program_id(0) * RECORDS + tl.arange(0, RECORDS)
    columns = tl.arange(0, 128)
    valid = record[:, None] < records
    head_local = record % SHARD_HC
    rest = record // SHARD_HC
    batch = rest % BS
    rest2 = rest // BS
    seq_local = rest2 % SHARD_SEQLEN
    peer = rest2 // SHARD_SEQLEN
    seq_global = peer * SHARD_SEQLEN + seq_local
    target_row = (batch * SEQLEN + seq_global) * SHARD_HC + head_local
    base = record[:, None] * PACKET
    stored = tl.load(packet_ptr + base + columns[None, :], mask=valid, other=VALUE_OFFSET).to(tl.int32)
    group = columns // 32
    code = tl.load(packet_ptr + base + 128 + group[None, :], mask=valid, other=0).to(tl.int32)
    scale_bits = (((code >> 3) - 15 + 127) << 23) | ((code & 7) << 20)
    scale = scale_bits.to(tl.float32, bitcast=True)
    decoded = (stored - VALUE_OFFSET).to(tl.float32) * scale
    tl.store(output_ptr + target_row[:, None] * 128 + columns[None, :], decoded, mask=valid)


def all_to_all_4d_qkv_fused(input_t: torch.Tensor, group, world: int):
    """int8-fused qkv exchange: (bs, shard_seqlen, hc, hs) -> (bs, seqlen, hc/world, hs).

    Returns None whenever the shape/layout is not the expected qkv one so the caller can run
    the stock path; never raises into the model.
    """
    try:
        if input_t.dim() != 4 or input_t.dtype != torch.bfloat16 or not input_t.is_contiguous() or world <= 1:
            return None
        bs, shard_seqlen, hc, hs = input_t.shape
        if hs != VECTOR or hc % world:
            return None
        if not fused_enabled() or _STATE["failed"]:
            return None
        shard_hc = hc // world
        records = bs * shard_seqlen * hc
        packet = _buf((world, shard_seqlen, bs, shard_hc, OUTPUT_PACKET), torch.uint8, input_t.device, tag="pktf")
        if records:
            _encode_pack_qkv_fused_kernel[(triton.cdiv(records, _rpp()),)](
                input_t,
                packet,
                records,
                shard_seqlen,
                bs,
                shard_hc,
                hc,
                RECORDS=_rpp(),
                PACKET=OUTPUT_PACKET,
                VALUE_OFFSET=VALUE_BIAS,
                num_warps=_warps(),
            )
        recv = _buf(packet.shape, torch.uint8, packet.device, tag="recvf")
        dist.all_to_all_single(recv.view(-1), packet.view(-1), group=group)
        seqlen = shard_seqlen * world
        # Fresh, not pooled: the caller returns this straight to attention, so two calls
        # (q then k) would otherwise alias the same buffer.
        out = torch.empty((bs, seqlen, shard_hc, hs), dtype=torch.bfloat16, device=input_t.device)
        if records:
            _decode_pack_qkv_fused_kernel[(triton.cdiv(records, _rpp()),)](
                recv,
                out,
                records,
                shard_seqlen,
                bs,
                shard_hc,
                seqlen,
                RECORDS=_rpp(),
                PACKET=OUTPUT_PACKET,
                VALUE_OFFSET=VALUE_BIAS,
                num_warps=_warps(),
            )
        _log_mode_once("int8-fused")
        _STATE["fused_calls"] += 1
        return out
    except Exception as exc:  # never break the lane
        _STATE["failed"] = True
        _log(f"int8-fused FAILED ({type(exc).__name__}: {exc}) -> bf16 for this run")
        return None


# ---------------------------------------------------------------------------
# Registration with the generic distributed layer
# ---------------------------------------------------------------------------
# comm.py ships only the stock exchange. The H3 model installs this transport from
# here during setup, so the generic layer carries no H3 knowledge.


def _exchange_adapter(input_t: torch.Tensor, group) -> torch.Tensor:
    """Stock-interface exchange; resolves the wire mode (bf16 by default) on every call."""
    return all_to_all_4d(input_t, group, dist.get_world_size(group))


def _fused_4d_adapter(input_t: torch.Tensor, group, seq_world_size: int, use_sync: bool):
    """Whole-call override for the qkv direction; None means "use the stock path"."""
    if not fused_enabled() or _STATE["failed"]:
        return None
    out = all_to_all_4d_qkv_fused(input_t.contiguous(), group, seq_world_size)
    if out is not None and use_sync:
        from vllm_omni.platforms import current_omni_platform

        current_omni_platform.synchronize()
    return out


def install() -> None:
    """Register this transport with ``vllm_omni.diffusion.distributed.comm``."""
    from vllm_omni.diffusion.distributed.comm import register_seq_all_to_all_backend

    register_seq_all_to_all_backend(exchange=_exchange_adapter, fused_4d=_fused_4d_adapter)


def stats() -> dict:
    """Transport counters: which mode is selected, how many exchanges ran, and any fallback.

    ``calls`` counts per-tensor quantised exchanges and ``fused_calls`` counts
    whole-call fused qkv exchanges; they are separate because the fused path can be dead
    while the plain one keeps the total moving. ``failed`` latches when the wire gave up
    and used bf16 for the rest of the run, so a caller can assert the fast path actually
    ran instead of inferring it from a result that would look fine either way.
    """
    return {
        "mode": _wire(),
        "calls": _STATE["calls"],
        "fused_calls": _STATE["fused_calls"],
        "failed": _STATE["failed"],
    }
