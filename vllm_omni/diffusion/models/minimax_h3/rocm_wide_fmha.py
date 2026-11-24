# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Eager ROCm module launcher for the two-segment MiniMax-H3 gfx950 kernel."""

import ctypes
import math
import os
import threading

import torch

_TOKENS, _HEADS, _DIM = 37_760, 14, 128
_SHAPE = (_TOKENS, _HEADS, _DIM)
_QKV_STRIDES = (5_376, _DIM, 1)
_CUTS = frozenset({37_746, 37_747, 37_748, 37_749, 37_750, 37_751, 37_753, 37_755, 37_757})
_LONG_TILES = 1_504
_V_SLOTS = 593
# Each stream owns about 130 MiB of V scratch. Additional streams use ordinary attention.
_MAX_WORKSPACES_PER_DEVICE = 4


def _worklist(cut: int) -> list[tuple[int, ...]]:
    """Partition each head's two independent segments exactly once."""
    if cut not in _CUTS:
        raise ValueError(f"Unsupported H3 segment boundary {cut}")
    units = (cut + 127) // 128
    kv_blocks = (cut + 63) // 64
    long = []
    short = []
    for head in range(_HEADS):
        tiles = _LONG_TILES // _HEADS + int(head < _LONG_TILES % _HEADS)
        # Cover the segment using 384-row and 256-row tiles. The first
        # tile starts before zero; the kernel masks those leading rows.
        wide_tiles = units - 2 * tiles
        first = cut - units * 128
        for tile in range(tiles):
            wide = tile < wide_tiles
            long.append((0, cut, 0, cut, first, 0, kv_blocks, head, 0, 8 if wide else 0, 0 if wide else 8))
            first += 384 if wide else 256
        tail = _TOKENS - cut
        short.append((cut, tail, cut, tail, tail - 384, 0, 1, head, 0, 8, 0))
    # Identity head order and the measured band schedule. The persistent
    # workgroups take new descriptors from a common device-side counter.
    long.sort(key=lambda row: (-row[9], row[7], -row[4]))
    return long[:512] + long[1122:1346] + long[512:1122] + long[1346:] + short


_Pointer = ctypes.c_void_p
_Int = ctypes.c_int32


class _AttentionArgs(ctypes.Structure):
    """By-value Args layout in h3_wide_attention's kernel argument metadata."""

    _fields_ = [
        ("q", _Pointer),
        ("k", _Pointer),
        ("v", _Pointer),
        ("out", _Pointer),
        ("strides", (ctypes.c_int64 * 3) * 4),
        ("work", _Pointer),
        ("partial_out", _Pointer),
        ("packed_v", _Pointer),
        ("heads", _Int),
        ("kv_heads", _Int),
        ("group", _Int),
        ("pack", _Int),
        ("causal", _Int),
        ("scale_log2e", ctypes.c_float),
        ("lse", _Pointer),
        ("lse_strides", ctypes.c_int64 * 3),
        ("pages", _Pointer),
        ("table_len", _Int),
        ("page_shift", _Int),
    ]


class _HipModule:
    def __init__(self, path: str):
        self.api = ctypes.CDLL("libamdhip64.so")
        self.api.hipGetErrorString.argtypes = [_Int]
        self.api.hipGetErrorString.restype = ctypes.c_char_p
        self.api.hipModuleLoad.argtypes = [ctypes.POINTER(_Pointer), ctypes.c_char_p]
        self.api.hipModuleGetFunction.argtypes = [ctypes.POINTER(_Pointer), _Pointer, ctypes.c_char_p]
        self.api.hipModuleUnload.argtypes = [_Pointer]
        self.api.hipMemsetAsync.argtypes = [_Pointer, _Int, ctypes.c_size_t, _Pointer]
        self.api.hipModuleLaunchKernel.argtypes = [
            _Pointer,
            *([ctypes.c_uint] * 7),
            _Pointer,
            ctypes.POINTER(_Pointer),
            ctypes.POINTER(_Pointer),
        ]
        self.handle = _Pointer()
        self.check(self.api.hipModuleLoad(ctypes.byref(self.handle), os.fsencode(path)), f"load {path}")
        try:
            self.pack = self.function(b"h3_pack_v")
            self.attention = self.function(b"h3_wide_attention")
        except RuntimeError:
            self.api.hipModuleUnload(self.handle)
            raise

    def check(self, error: int, operation: str) -> None:
        if error:
            message = self.api.hipGetErrorString(error).decode()
            raise RuntimeError(f"H3 BF16 attention failed to {operation}: {message} ({error})")

    def function(self, name: bytes):
        handle = _Pointer()
        self.check(self.api.hipModuleGetFunction(ctypes.byref(handle), self.handle, name), f"resolve {name.decode()}")
        return handle

    def launch(self, function, grid_x: int, grid_y: int, stream: int, *args) -> None:
        arg_pointers = (_Pointer * len(args))(*(ctypes.addressof(arg) for arg in args))
        self.check(
            self.api.hipModuleLaunchKernel(function, grid_x, grid_y, 1, 512, 1, 1, 0, stream, arg_pointers, None),
            "launch kernel",
        )


class _Workspace:
    def __init__(self, device: torch.device, path: str):
        self.packed_v = torch.empty((_HEADS, _V_SLOTS, 16, 64, 8), dtype=torch.bfloat16, device=device)
        self.queue = torch.empty(1, dtype=torch.int32, device=device)
        self.tables: dict[int, torch.Tensor] = {}
        self.module = _HipModule(path)
        self.lock = threading.Lock()

    def forward(self, q, k, v, cut, stream):
        # Serialize enqueueing all three operations if two host threads share a stream.
        with self.lock:
            if cut not in self.tables:
                self.tables[cut] = torch.tensor(_worklist(cut), dtype=torch.int32, device=q.device)
            table = self.tables[cut]
            out = torch.empty(_SHAPE, dtype=q.dtype, device=q.device)
            for tensor in (q, k, v):
                tensor.record_stream(stream)
            stream_ptr = stream.cuda_stream
            packed_v = self.packed_v.data_ptr()
            self.module.launch(
                self.module.pack,
                _V_SLOTS,
                _HEADS,
                stream_ptr,
                _Pointer(k.data_ptr()),
                _Pointer(v.data_ptr()),
                _Pointer(),
                _Pointer(packed_v),
                _Int(cut),
            )
            self.module.check(
                self.module.api.hipMemsetAsync(self.queue.data_ptr(), 0, 4, stream_ptr), "reset work queue"
            )
            args = _AttentionArgs(
                q=q.data_ptr(),
                k=k.data_ptr(),
                v=v.data_ptr(),
                out=out.data_ptr(),
                work=table.data_ptr(),
                packed_v=packed_v,
                heads=_HEADS,
                kv_heads=_HEADS,
                group=1,
                pack=1,
                scale_log2e=math.log2(math.e) / math.sqrt(_DIM),
            )
            for idx, row_stride in enumerate((_QKV_STRIDES[0], _QKV_STRIDES[0], _QKV_STRIDES[0], _HEADS * _DIM)):
                args.strides[idx][:] = (0, row_stride, _DIM)
            self.module.launch(
                self.module.attention, 256, 1, stream_ptr, args, _Int(len(table)), _Pointer(self.queue.data_ptr())
            )
            return out


_workspaces: dict[tuple[torch.device, int, str], _Workspace] = {}
_workspace_lock = threading.Lock()


def try_wide_fmha(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cut: int, hsaco_path: str) -> torch.Tensor | None:
    """Return dense attention when the ROCm code object and input geometry match."""
    if not hsaco_path or not torch.version.hip or q.device.type != "cuda" or cut not in _CUTS:
        return None
    if (
        torch.compiler.is_compiling()
        or torch.is_grad_enabled()
        or any(
            t.device != q.device
            or t.dtype != torch.bfloat16
            or t.layout != torch.strided
            or t.shape != _SHAPE
            or (t.stride() != _QKV_STRIDES and not t.is_contiguous())
            or t.requires_grad
            or t.is_neg()
            or t.is_conj()
            or t.data_ptr() % 16
            for t in (q, k, v)
        )
    ):
        return None
    with torch.accelerator.device_index(q.device.index):
        if torch.cuda.is_current_stream_capturing():
            return None
        if torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0] != "gfx950":
            return None
        stream = torch.cuda.current_stream(q.device)
        key = (q.device, stream.cuda_stream, hsaco_path)
        with _workspace_lock:
            workspace = _workspaces.get(key)
            if workspace is None:
                if sum(device == q.device for device, _, _ in _workspaces) >= _MAX_WORKSPACES_PER_DEVICE:
                    return None
                workspace = _Workspace(q.device, hsaco_path)
                _workspaces[key] = workspace
        if any(t.stride() != _QKV_STRIDES for t in (q, k, v)):
            # Default Ulysses returns contiguous Q/K/V. Defer their shared layout
            # allocation until the device and this particular stream can launch.
            for tensor in (q, k, v):
                tensor.record_stream(stream)
            q, k, v = torch.stack((q, k, v), dim=1).unbind(1)
        return workspace.forward(q, k, v, cut, stream)
