# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Grouped gfx950 AITER attention for the two-segment MiniMax-H3 SP4 layout."""

import ctypes
import ctypes.util
import hashlib
import os
import struct
from functools import lru_cache
from importlib import resources
from threading import Lock

import torch

from .rocm_h3_bounds import GROUPED_CUTS

SHAPE = (1, 37760, 14, 128)
# https://github.com/ROCm/aiter/blob/50bfa78d/hsa/gfx950/fmha_v3_fwd/fwd_hd128_bf16_group.co
_STOCK_SHA = "c64e27c2a14a22eac8d841dc16ea67d29a7bedfd674e4cfc81a67c5c892294c1"
_SCHEDULED_SHA = "c04afb7b5e38f0933fad59b797df3d3ea6cdbb85663938442e974b390b011880"
_STRIDES = ((1792, 128, 1), (5376, 128, 1), (5376, 384, 1))
_PLANS = {}
_LOAD_LOCK = Lock()


def _with_tile_first_grid(original: bytes) -> bytes | None:
    # The released grouped kernel uses s2/s3/s4 as (head, sequence, tile).
    # There are three self moves in the prologue before the first use of those
    # IDs. Swap X and Z there, and launch (tile, sequence, head). The file
    # digest also identifies the symbol, ISA, free temporary s40 and kernarg ABI.
    if hashlib.sha256(original).hexdigest() != _STOCK_SHA:
        return None

    def s_mov_b32(dst: int, src: int) -> bytes:
        return struct.pack("<I", 0xBE800000 | dst << 16 | src)

    previous = s_mov_b32(2, 2) + s_mov_b32(3, 3) + s_mov_b32(4, 4)
    moved = s_mov_b32(40, 2) + s_mov_b32(2, 4) + s_mov_b32(4, 40)
    scheduled = original.replace(previous, moved, 1)
    return scheduled if hashlib.sha256(scheduled).hexdigest() == _SCHEDULED_SHA else None


@lru_cache(maxsize=1)
def _code_object() -> bytes | None:
    # Wheels put the public code objects in aiter_meta. Source installs may
    # expose a different catalog; allow the regular AITER dispatcher there.
    try:
        image = resources.files("aiter_meta").joinpath("hsa/gfx950/fmha_v3_fwd/fwd_hd128_bf16_group.co")
        return _with_tile_first_grid(image.read_bytes())
    except (ImportError, OSError):
        return None


class _Slot(ctypes.Union):
    # AITER fmha_fwd_v3_args in csrc/include/mha_fwd.h pads each argument to 16 bytes.
    _fields_ = [
        ("pointer", ctypes.c_void_p),
        ("u32", ctypes.c_uint32),
        ("f32", ctypes.c_float),
        ("pad", ctypes.c_byte * 16),
    ]


class _Args(ctypes.Structure):
    _fields_ = [
        (name, _Slot)
        for name in (
            "o q k v lse scale seq_len seq_stride tile_stride head_stride batch_stride gqa "
            "k_seq_stride k_head_stride k_batch_stride options emit_lse kv_seq_len qk_dim v_dim heads "
            "v_seq_stride v_head_stride v_batch_stride o_seq_stride o_head_stride o_batch_stride "
            "q_bounds k_bounds lse_head_stride padded_q_bounds padded_k_bounds "
            "q_descale k_descale v_descale q_descale_bs q_descale_hs k_descale_bs "
            "k_descale_hs v_descale_bs v_descale_hs"
        ).split()
    ]


@lru_cache
def _layout_args(row_stride: int, head_stride: int) -> _Args:
    args = _Args()
    args.scale.f32 = 128**-0.5
    args.seq_len.u32 = args.kv_seq_len.u32 = SHAPE[1]
    args.heads.u32 = SHAPE[2]
    args.qk_dim.u32 = args.v_dim.u32 = SHAPE[3]
    args.seq_stride.u32 = args.k_seq_stride.u32 = args.v_seq_stride.u32 = row_stride * 2
    args.head_stride.u32 = args.k_head_stride.u32 = args.v_head_stride.u32 = head_stride * 2
    args.tile_stride.u32 = 256 * row_stride * 2
    args.o_seq_stride.u32, args.o_head_stride.u32 = 1792 * 2, 128 * 2
    args.gqa.u32 = 1
    args.options.u32 = 5  # grouped, no mask, no dropout
    return args


@lru_cache(maxsize=1)
def _hip():
    ptr = ctypes.c_void_p
    u = ctypes.c_uint
    try:
        lib = ctypes.CDLL(ctypes.util.find_library("amdhip64") or "libamdhip64.so")
        lib.hipModuleLoadData.argtypes = [ctypes.POINTER(ptr), ptr]
        lib.hipModuleGetFunction.argtypes = [ctypes.POINTER(ptr), ptr, ctypes.c_char_p]
        lib.hipModuleUnload.argtypes = [ptr]
        lib.hipModuleLaunchKernel.argtypes = [ptr, u, u, u, u, u, u, u, ptr, ptr, ctypes.POINTER(ptr)]
    except (OSError, AttributeError):
        return None
    return lib


def _load():
    image = _code_object()
    lib = _hip()
    if image is None or lib is None:
        return None
    mod = ctypes.c_void_p()
    fn = ctypes.c_void_p()
    if lib.hipModuleLoadData(ctypes.byref(mod), image):
        return None
    if lib.hipModuleGetFunction(ctypes.byref(fn), mod, b"_ZN5aiter25fmha_fwd_hd128_bf16_groupE"):
        lib.hipModuleUnload(mod)
        return None
    # Keep one module per device/context for the lifetime of graph replays.
    return lib, mod, fn


def try_grouped_fmha(q, k, v, cu_q, cu_k, *, max_seqlen):
    """Preconditions include two owner-validated packed segments and default softmax scale."""
    if torch.version.hip is None or torch.compiler.is_compiling() or q.device.type != "cuda":
        return None
    if max_seqlen not in GROUPED_CUTS or q.stride()[1:] not in _STRIDES:
        return None
    if any(
        t.shape != SHAPE
        or t.dtype != torch.bfloat16
        or t.device != q.device
        or t.is_neg()
        or t.is_conj()
        or t.requires_grad
        or t.stride()[1:] != q.stride()[1:]
        or t.data_ptr() % 16
        for t in (q, k, v)
    ):
        return None
    if any(
        t.device != q.device
        or t.dtype != torch.int32
        or t.is_neg()
        or t.is_conj()
        or t.shape != (3,)
        or t.stride() != (1,)
        for t in (cu_q, cu_k)
    ):
        return None
    with torch.accelerator.device_index(q.device.index):
        key = os.getpid(), q.device.index
        if key not in _PLANS:
            if torch.cuda.is_current_stream_capturing():
                return None
            with _LOAD_LOCK:
                if key not in _PLANS:
                    arch = torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0]
                    _PLANS[key] = _load() if arch == "gfx950" else None
        plan = _PLANS[key]
        if plan is None:
            return None
        lib, _, fn = plan
        stream = torch.cuda.current_stream(q.device)
        out = torch.empty(SHAPE, dtype=q.dtype, device=q.device)
        args = _Args.from_buffer_copy(_layout_args(q.stride(1), q.stride(2)))
        args.o.pointer = out.data_ptr()
        args.q.pointer, args.k.pointer, args.v.pointer = q.data_ptr(), k.data_ptr(), v.data_ptr()
        args.q_bounds.pointer = args.padded_q_bounds.pointer = cu_q.data_ptr()
        args.k_bounds.pointer = args.padded_k_bounds.pointer = cu_k.data_ptr()
        size = ctypes.c_size_t(ctypes.sizeof(args))
        # HIP_LAUNCH_PARAM_BUFFER_POINTER / BUFFER_SIZE / END from hip_runtime_api.h.
        extra = (ctypes.c_void_p * 5)(1, ctypes.addressof(args), 2, ctypes.addressof(size), 3)
        rc = lib.hipModuleLaunchKernel(fn, 148, 2, 14, 512, 1, 1, 0, stream.cuda_stream, None, extra)
        if rc:
            raise RuntimeError(f"Launching MiniMax-H3 grouped AITER attention failed with HIP status {rc}")
        for tensor in (q, k, v, cu_q, cu_k):
            tensor.record_stream(stream)
        return out
