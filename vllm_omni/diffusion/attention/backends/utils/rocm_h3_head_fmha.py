# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Launch grouped AITER FMHA on one received head group of the MiniMax-H3 Ulysses exchange."""

import ctypes
import ctypes.util
import hashlib
import os
import struct
from collections.abc import Callable
from functools import cache
from importlib import resources
from pathlib import Path
from threading import Lock

import torch

# ROCm/aiter 50bfa78d, hsa/gfx950/fmha_v3_fwd/fwd_hd128_bf16_group.co.
# Pin both versions so another instruction sequence or kernel argument ABI is never launched.
_STOCK_SHA = "c64e27c2a14a22eac8d841dc16ea67d29a7bedfd674e4cfc81a67c5c892294c1"
_TILE_FIRST_SHA = "c04afb7b5e38f0933fad59b797df3d3ea6cdbb85663938442e974b390b011880"
_SYMBOL = b"_ZN5aiter25fmha_fwd_hd128_bf16_groupE"
_HeadFmha = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]
_RUNNERS: dict[tuple[int, torch.device], _HeadFmha | None] = {}
_LOAD_LOCK = Lock()


def _tile_first(original: bytes) -> bytes | None:
    if hashlib.sha256(original).hexdigest() != _STOCK_SHA:
        return None
    # The first three s_mov_b32 instructions in the prologue are self moves.
    # Swap the workgroup X/Z IDs via unused s40 before their first use, so X
    # traverses 256-query tiles and Z traverses heads. The attention body is unchanged.
    original_moves = struct.pack("<III", 0xBE820002, 0xBE830003, 0xBE840004)
    swap_xz = struct.pack("<III", 0xBEA80002, 0xBE820004, 0xBE840028)
    image = original.replace(original_moves, swap_xz, 1)
    return image if hashlib.sha256(image).hexdigest() == _TILE_FIRST_SHA else None


@cache
def _code_object() -> bytes | None:
    relative = "gfx950/fmha_v3_fwd/fwd_hd128_bf16_group.co"
    try:
        image = (resources.files("aiter_meta") / "hsa" / relative).read_bytes()
    except (ImportError, OSError):
        # In a source install, AITER resolves its assembly catalog in jit.core.
        try:
            from aiter.jit.core import AITER_ASM_DIR

            image = (Path(AITER_ASM_DIR) / relative).read_bytes()
        except (ImportError, OSError):
            return None
    return _tile_first(image)


class _Slot(ctypes.Union):
    # AITER csrc/include/mha_fwd.h pads each argument of fmha_fwd_v3_args to 16 bytes.
    _fields_ = [
        ("ptr", ctypes.c_void_p),
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


@cache
def _layout_args(heads: int) -> _Args:
    if heads not in (4, 10):
        raise ValueError(f"Unsupported H3 head group size {heads}")
    args = _Args()
    args.scale.f32 = 128**-0.5
    args.seq_len.u32 = args.kv_seq_len.u32 = 37760
    args.heads.u32 = heads
    args.qk_dim.u32 = args.v_dim.u32 = 128
    # Each received row is [Q, K, V] x heads x 128 BF16. The output is packed.
    args.seq_stride.u32 = args.k_seq_stride.u32 = args.v_seq_stride.u32 = 3 * heads * 128 * 2
    args.tile_stride.u32 = 256 * args.seq_stride.u32
    args.head_stride.u32 = args.k_head_stride.u32 = args.v_head_stride.u32 = args.o_head_stride.u32 = 128 * 2
    args.o_seq_stride.u32 = heads * 128 * 2
    args.gqa.u32 = 1
    args.options.u32 = 5  # grouped, no mask or dropout
    return args


def _load(device: torch.device) -> _HeadFmha | None:
    image = _code_object()
    if image is None:
        return None
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
    module, kernel = ptr(), ptr()
    if lib.hipModuleLoadData(ctypes.byref(module), image):
        return None
    if lib.hipModuleGetFunction(ctypes.byref(kernel), module, _SYMBOL):
        lib.hipModuleUnload(module)
        return None

    def run(received: torch.Tensor, bounds: torch.Tensor, output: torch.Tensor) -> None:
        """Consume one completed receive and launch on the caller's current compute stream."""
        heads = received.size(3)
        if (
            received.shape != (4, 9440, 3, heads, 128)
            or output.shape != (4, 9440, heads, 128)
            or received.dtype != torch.bfloat16
            or output.dtype != torch.bfloat16
            or bounds.shape != (3,)
            or bounds.dtype != torch.int32
            or any(
                tensor.device != device or not tensor.is_contiguous() or tensor.is_neg() or tensor.requires_grad
                for tensor in (received, bounds, output)
            )
            or (received.data_ptr() | output.data_ptr()) % 16
        ):
            raise ValueError("Unexpected grouped H3 FMHA receive, output or bounds layout")
        args = _Args.from_buffer_copy(_layout_args(heads))
        args.o.ptr = output.data_ptr()
        args.q.ptr = received.data_ptr()
        args.k.ptr = args.q.ptr + heads * 128 * 2
        args.v.ptr = args.k.ptr + heads * 128 * 2
        args.q_bounds.ptr = args.k_bounds.ptr = args.padded_q_bounds.ptr = args.padded_k_bounds.ptr = bounds.data_ptr()
        size = ctypes.c_size_t(ctypes.sizeof(args))
        # HIP_LAUNCH_PARAM_BUFFER_POINTER / BUFFER_SIZE / END from hip_runtime_api.h.
        extra = (ptr * 5)(1, ctypes.addressof(args), 2, ctypes.addressof(size), 3)
        with torch.accelerator.device_index(device.index):
            stream = torch.get_device_module(device).current_stream(device)
            rc = lib.hipModuleLaunchKernel(kernel, 148, 2, heads, 512, 1, 1, 0, stream.cuda_stream, None, extra)
            if rc:
                raise RuntimeError(f"Launching grouped H3 head FMHA failed with HIP status {rc}")
            for tensor in (received, bounds, output):
                tensor.record_stream(stream)

    return run


def prepare_head_fmha(device: torch.device) -> _HeadFmha | None:
    """Resolve the optional gfx950 kernel before collectives; the caller owns the two validated segment bounds."""
    if torch.version.hip is None or device.type != "cuda" or torch.compiler.is_compiling():
        return None
    if device.index is None:
        device = torch.device("cuda", torch.accelerator.current_device_index())
    backend = torch.get_device_module(device)
    with torch.accelerator.device_index(device.index):
        if backend.is_current_stream_capturing():
            return None
        key = (os.getpid(), device)
        if key not in _RUNNERS:
            with _LOAD_LOCK:
                if key not in _RUNNERS:
                    arch = backend.get_device_properties(device).gcnArchName.split(":")[0]
                    _RUNNERS[key] = _load(device) if arch == "gfx950" else None
        return _RUNNERS[key]
