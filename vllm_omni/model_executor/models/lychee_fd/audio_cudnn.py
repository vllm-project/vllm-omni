# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Released Lychee adaptor convolution using the current cuDNN backend API.

cuDNN 9.20's heuristic changed engine23 SPLIT_K_BUF/SPLIT_K_SLC from 1/640
(the released cuDNN9.7 runtime) to 4/256. Those choices change BF16 rounding.
An explicit model-local plan restores the released accumulation on the fixed
400ms streaming shape. It uses Torch's existing library and current stream.
Other devices and cuDNN versions retain the ordinary Conv1d implementation.

Enums below follow the installed cuDNN9.20 cudnn_graph.h and public API:
https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-graph-library.html
"""

from __future__ import annotations

import ctypes
import threading
from enum import IntEnum
from pathlib import Path

import torch
from torch import nn

from vllm_omni.platforms import current_omni_platform


class _Descriptor(IntEnum):
    CONVOLUTION = 1
    ENGINE = 2
    ENGINE_CONFIG = 3
    EXECUTION_PLAN = 5
    KNOB_CHOICE = 7
    CONVOLUTION_FORWARD = 10
    OPERATION_GRAPH = 15
    VARIANT_PACK = 16
    TENSOR = 17


class _Type(IntEnum):
    HANDLE = 0
    DATA_TYPE = 1
    BOOLEAN = 2
    INT64 = 3
    FLOAT = 4
    VOID_PTR = 6
    CONVOLUTION_MODE = 7
    KNOB_TYPE = 9
    BACKEND_DESCRIPTOR = 15


class _Attribute(IntEnum):
    CONV_COMP_TYPE = 100
    CONV_MODE = 101
    CONV_DILATIONS = 102
    CONV_STRIDES = 103
    CONV_POST_PADDING = 104
    CONV_PRE_PADDING = 105
    CONV_SPATIAL_DIMS = 106
    ENGINE_CONFIG_ENGINE = 300
    ENGINE_CONFIG_KNOBS = 302
    PLAN_HANDLE = 400
    PLAN_ENGINE_CONFIG = 401
    PLAN_WORKSPACE_SIZE = 402
    KNOB_TYPE = 600
    KNOB_VALUE = 601
    FORWARD_ALPHA = 700
    FORWARD_BETA = 701
    FORWARD_CONV = 702
    FORWARD_WEIGHT = 703
    FORWARD_INPUT = 704
    FORWARD_OUTPUT = 705
    GRAPH_HANDLE = 800
    GRAPH_OPERATIONS = 801
    TENSOR_ALIGNMENT = 900
    TENSOR_DATA_TYPE = 901
    TENSOR_DIMENSIONS = 902
    TENSOR_STRIDES = 903
    TENSOR_UID = 906
    TENSOR_VIRTUAL = 907
    PACK_UIDS = 1000
    PACK_POINTERS = 1001
    PACK_WORKSPACE = 1003
    ENGINE_GRAPH = 1300
    ENGINE_INDEX = 1301


class _Knob(IntEnum):
    TILE_SIZE = 2
    SPLIT_K_BUF = 12
    TILEK = 13
    STAGES = 14
    REDUCTION_MODE = 15
    SPLIT_K_SLC = 17
    IDX_MODE = 18
    SPECFILT = 23


_RELEASED_KNOBS = {
    _Knob.TILE_SIZE: 5,
    _Knob.SPLIT_K_BUF: 1,
    _Knob.TILEK: 1,
    _Knob.STAGES: 3,
    _Knob.REDUCTION_MODE: 0,
    _Knob.SPLIT_K_SLC: 640,
    _Knob.IDX_MODE: 0,
    _Knob.SPECFILT: 0,
}
_CTYPE = {
    _Type.HANDLE: ctypes.c_void_p,
    _Type.DATA_TYPE: ctypes.c_int,
    _Type.BOOLEAN: ctypes.c_bool,
    _Type.INT64: ctypes.c_int64,
    _Type.FLOAT: ctypes.c_float,
    _Type.VOID_PTR: ctypes.c_void_p,
    _Type.CONVOLUTION_MODE: ctypes.c_int,
    _Type.KNOB_TYPE: ctypes.c_int,
    _Type.BACKEND_DESCRIPTOR: ctypes.c_void_p,
}


def _supports_released_profile(device: torch.device) -> bool:
    """Use the measured profile only on its validated CUDA/cuDNN runtime."""
    return (
        device.type == "cuda"
        and device.index is not None
        and current_omni_platform.is_cuda()
        and torch.backends.cudnn.version() == 92000
        and current_omni_platform.get_device_capability(device.index) == (8, 0)
    )


def _load_torch_cudnn():
    # Resolve the DSO Torch actually loaded, rather than searching another
    # toolkit/frontend package with a different cuDNN installation.
    version = torch.backends.cudnn.version()
    if version != 92000:
        raise RuntimeError(f"Lychee released adaptor plan requires cuDNN9.20.0, got {version}")
    mapped = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if line.split() and line.split()[-1].endswith("/libcudnn.so.9")
        }
    )
    if len(mapped) != 1:
        raise RuntimeError(f"Lychee expected one Torch cuDNN main library, got {mapped}")
    return ctypes.CDLL(mapped[0]), mapped[0]


class LycheeAudioCudnnPlan:
    """One device, handle and bounded workspace; serialize reuse across streams."""

    MAX_WORKSPACE_BYTES = 16 * 1024**2

    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("Lychee adaptor plan requires an explicit CUDA device")
        self._lock = threading.RLock()
        self._closed = False
        self._completion = None
        self._descriptors = []
        self._handle = ctypes.c_void_p()
        self.workspace = None
        self.workspace_bytes = 0
        self.lib, self.library_path = _load_torch_cudnn()
        self._bind_api()
        try:
            with torch.accelerator.device_index(self.device.index):
                if current_omni_platform.get_device_capability(self.device.index) != (8, 0):
                    raise RuntimeError("Lychee released adaptor cuDNN profile requires the validated SM80 device")
                self._check(self.lib.cudnnCreate(ctypes.byref(self._handle)), "create handle")
                self._build()
                self.workspace = torch.empty(self.workspace_bytes, device=self.device, dtype=torch.uint8)
        except BaseException:
            self.close(suppress_errors=True)
            raise

    def _bind_api(self):
        pointer = ctypes.c_void_p
        self.lib.cudnnGetVersion.restype = ctypes.c_size_t
        if self.lib.cudnnGetVersion() != 92000:
            raise RuntimeError("Torch-mapped cuDNN library disagrees with the supported9.20 profile")
        self.lib.cudnnGetErrorString.restype = ctypes.c_char_p
        self.lib.cudnnGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudnnCreate.argtypes = [ctypes.POINTER(pointer)]
        self.lib.cudnnDestroy.argtypes = [pointer]
        self.lib.cudnnSetStream.argtypes = [pointer, pointer]
        self.lib.cudnnBackendCreateDescriptor.argtypes = [ctypes.c_int, ctypes.POINTER(pointer)]
        self.lib.cudnnBackendSetAttribute.argtypes = [pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int64, pointer]
        self.lib.cudnnBackendGetAttribute.argtypes = [
            pointer,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64),
            pointer,
        ]
        self.lib.cudnnBackendFinalize.argtypes = [pointer]
        self.lib.cudnnBackendDestroyDescriptor.argtypes = [pointer]
        self.lib.cudnnBackendExecute.argtypes = [pointer, pointer, pointer]

    def _check(self, status, operation):
        if status:
            message = self.lib.cudnnGetErrorString(status).decode()
            raise RuntimeError(f"Lychee cuDNN {operation}: {message} ({status})")

    def _create(self, descriptor_type):
        descriptor = ctypes.c_void_p()
        self._check(
            self.lib.cudnnBackendCreateDescriptor(descriptor_type, ctypes.byref(descriptor)),
            f"create {descriptor_type.name}",
        )
        self._descriptors.append(descriptor)
        return descriptor

    def _set(self, descriptor, attribute, value_type, values):
        array = (_CTYPE[value_type] * len(values))(*values)
        self._check(
            self.lib.cudnnBackendSetAttribute(descriptor, attribute, value_type, len(values), array),
            f"set {attribute.name}",
        )

    def _finalize(self, descriptor):
        self._check(self.lib.cudnnBackendFinalize(descriptor), "finalize descriptor")

    def _tensor(self, uid, dimensions, strides):
        descriptor = self._create(_Descriptor.TENSOR)
        # CUDNN_DATA_BFLOAT16=9; the convolution accumulation is FLOAT=0.
        for attribute, value_type, values in [
            (_Attribute.TENSOR_ALIGNMENT, _Type.INT64, [32]),
            (_Attribute.TENSOR_DATA_TYPE, _Type.DATA_TYPE, [9]),
            (_Attribute.TENSOR_DIMENSIONS, _Type.INT64, dimensions),
            (_Attribute.TENSOR_STRIDES, _Type.INT64, strides),
            (_Attribute.TENSOR_UID, _Type.INT64, [uid]),
            (_Attribute.TENSOR_VIRTUAL, _Type.BOOLEAN, [False]),
        ]:
            self._set(descriptor, attribute, value_type, values)
        self._finalize(descriptor)
        return descriptor

    def _build(self):
        x = self._tensor(120, [1, 1280, 1, 10], [12800, 10, 10, 1])
        w = self._tensor(119, [1280, 1280, 1, 3], [3840, 3, 3, 1])
        y = self._tensor(121, [1, 1280, 1, 5], [6400, 5, 5, 1])
        conv = self._create(_Descriptor.CONVOLUTION)
        for attribute, value_type, values in [
            (_Attribute.CONV_COMP_TYPE, _Type.DATA_TYPE, [0]),
            (_Attribute.CONV_MODE, _Type.CONVOLUTION_MODE, [1]),
            (_Attribute.CONV_SPATIAL_DIMS, _Type.INT64, [2]),
            (_Attribute.CONV_DILATIONS, _Type.INT64, [1, 1]),
            (_Attribute.CONV_STRIDES, _Type.INT64, [1, 2]),
            (_Attribute.CONV_POST_PADDING, _Type.INT64, [0, 1]),
            (_Attribute.CONV_PRE_PADDING, _Type.INT64, [0, 1]),
        ]:
            self._set(conv, attribute, value_type, values)
        self._finalize(conv)
        operation = self._create(_Descriptor.CONVOLUTION_FORWARD)
        for attribute, descriptor in [
            (_Attribute.FORWARD_CONV, conv),
            (_Attribute.FORWARD_WEIGHT, w),
            (_Attribute.FORWARD_INPUT, x),
            (_Attribute.FORWARD_OUTPUT, y),
        ]:
            self._set(operation, attribute, _Type.BACKEND_DESCRIPTOR, [descriptor.value])
        self._set(operation, _Attribute.FORWARD_ALPHA, _Type.FLOAT, [1.0])
        self._set(operation, _Attribute.FORWARD_BETA, _Type.FLOAT, [0.0])
        self._finalize(operation)
        graph = self._create(_Descriptor.OPERATION_GRAPH)
        self._set(graph, _Attribute.GRAPH_HANDLE, _Type.HANDLE, [self._handle.value])
        self._set(graph, _Attribute.GRAPH_OPERATIONS, _Type.BACKEND_DESCRIPTOR, [operation.value])
        self._finalize(graph)
        engine = self._create(_Descriptor.ENGINE)
        self._set(engine, _Attribute.ENGINE_GRAPH, _Type.BACKEND_DESCRIPTOR, [graph.value])
        self._set(engine, _Attribute.ENGINE_INDEX, _Type.INT64, [23])
        self._finalize(engine)
        choices = []
        for knob, value in _RELEASED_KNOBS.items():
            choice = self._create(_Descriptor.KNOB_CHOICE)
            self._set(choice, _Attribute.KNOB_TYPE, _Type.KNOB_TYPE, [knob])
            self._set(choice, _Attribute.KNOB_VALUE, _Type.INT64, [value])
            self._finalize(choice)
            choices.append(choice.value)
        engine_config = self._create(_Descriptor.ENGINE_CONFIG)
        self._set(engine_config, _Attribute.ENGINE_CONFIG_ENGINE, _Type.BACKEND_DESCRIPTOR, [engine.value])
        self._set(engine_config, _Attribute.ENGINE_CONFIG_KNOBS, _Type.BACKEND_DESCRIPTOR, choices)
        self._finalize(engine_config)
        self._plan = self._create(_Descriptor.EXECUTION_PLAN)
        self._set(self._plan, _Attribute.PLAN_HANDLE, _Type.HANDLE, [self._handle.value])
        self._set(self._plan, _Attribute.PLAN_ENGINE_CONFIG, _Type.BACKEND_DESCRIPTOR, [engine_config.value])
        self._finalize(self._plan)
        count, size = ctypes.c_int64(), ctypes.c_int64()
        self._check(
            self.lib.cudnnBackendGetAttribute(
                self._plan, _Attribute.PLAN_WORKSPACE_SIZE, _Type.INT64, 1, ctypes.byref(count), ctypes.byref(size)
            ),
            "get workspace size",
        )
        if count.value != 1 or not 0 <= size.value <= self.MAX_WORKSPACE_BYTES:
            raise RuntimeError(f"Lychee adaptor workspace exceeds bound: count={count.value}, bytes={size.value}")
        self.workspace_bytes = size.value

    def execute(self, inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None):
        if tuple(inputs.shape) != (1, 1280, 10) or tuple(weight.shape) != (1280, 1280, 3):
            raise ValueError("Lychee released adaptor plan requires B1,C1280,L10,K3")
        tensors = [inputs, weight] + ([] if bias is None else [bias])
        if any(tensor.device != self.device or tensor.dtype != torch.bfloat16 for tensor in tensors):
            raise ValueError("Lychee adaptor plan tensors must be BF16 on the plan's CUDA device")
        if bias is not None and tuple(bias.shape) != (1280,):
            raise ValueError("Lychee adaptor plan bias must have1280 channels")
        with self._lock, torch.accelerator.device_index(self.device.index):
            if self._closed:
                raise RuntimeError("Lychee adaptor plan is closed")
            stream = torch.cuda.current_stream(self.device)
            if self._completion is not None:
                stream.wait_event(self._completion)
            x, w = inputs.contiguous(), weight.contiguous()
            y = torch.empty((1, 1280, 5), device=self.device, dtype=torch.bfloat16)
            for tensor in [x, w, y, self.workspace] + ([] if bias is None else [bias]):
                tensor.record_stream(stream)
            pack = self._create(_Descriptor.VARIANT_PACK)
            try:
                self._check(self.lib.cudnnSetStream(self._handle, stream.cuda_stream), "set stream")
                self._set(pack, _Attribute.PACK_UIDS, _Type.INT64, [120, 119, 121])
                self._set(pack, _Attribute.PACK_POINTERS, _Type.VOID_PTR, [x.data_ptr(), w.data_ptr(), y.data_ptr()])
                self._set(
                    pack,
                    _Attribute.PACK_WORKSPACE,
                    _Type.VOID_PTR,
                    [self.workspace.data_ptr() if self.workspace_bytes else 0],
                )
                self._finalize(pack)
                self._check(self.lib.cudnnBackendExecute(self._handle, self._plan, pack), "execute adaptor plan")
                result = y if bias is None else y + bias.reshape(1, -1, 1)
                # An event plus CPU lock serializes reuse of the same workspace
                # and handle across streams without a host synchronization.
                self._completion = torch.cuda.Event()
                self._completion.record(stream)
                return result
            except BaseException:
                try:
                    stream.synchronize()
                finally:
                    self.close(suppress_errors=True)
                raise
            finally:
                if pack in self._descriptors:
                    self.lib.cudnnBackendDestroyDescriptor(pack)
                    self._descriptors.remove(pack)

    def close(self, *, suppress_errors=False):
        with self._lock, torch.accelerator.device_index(self.device.index):
            if self._closed:
                return
            self._closed = True
            errors = []
            if self._completion is not None:
                try:
                    self._completion.synchronize()
                except Exception as error:
                    errors.append(error)
                self._completion = None
            # Descriptors depend on each other; release them in reverse order.
            for descriptor in reversed(self._descriptors):
                try:
                    self._check(self.lib.cudnnBackendDestroyDescriptor(descriptor), "destroy descriptor")
                except Exception as error:
                    errors.append(error)
            self._descriptors.clear()
            if self._handle.value:
                try:
                    self._check(self.lib.cudnnDestroy(self._handle), "destroy handle")
                except Exception as error:
                    errors.append(error)
                self._handle = ctypes.c_void_p()
            self.workspace = None
            if errors and not suppress_errors:
                raise errors[0]

    def __del__(self):
        try:
            self.close(suppress_errors=True)
        except Exception:
            pass


class LycheeAudioConv1d(nn.Conv1d):
    """Checkpoint-compatible Conv1d with the released streaming CUDA profile."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._released_plan = None
        self._released_lock = threading.RLock()
        self._released_closed = False

    def _is_released_config(self):
        return (
            self.in_channels == self.out_channels == 1280
            and self.kernel_size == (3,)
            and self.stride == (2,)
            and self.padding == (1,)
            and self.dilation == (1,)
            and self.groups == 1
            and self.padding_mode == "zeros"
        )

    def _conv_forward(self, inputs, weight, bias):
        weight = weight.to(inputs.dtype)
        bias = None if bias is None else bias.to(inputs.dtype)
        if not (
            inputs.is_cuda
            and inputs.dtype == torch.bfloat16
            and self._is_released_config()
            and _supports_released_profile(inputs.device)
        ):
            return super()._conv_forward(inputs, weight, bias)
        if inputs.ndim != 3 or inputs.shape[1:] != (1280, 10) or inputs.shape[0] < 1:
            raise ValueError("Lychee released CUDA adaptor requires nonempty [B,1280,10] input")
        if torch.is_grad_enabled():
            raise RuntimeError("Lychee released CUDA adaptor requires inference/no_grad mode")
        with self._released_lock:
            if self._released_closed:
                raise RuntimeError("Lychee released CUDA adaptor is closed")
            if self._released_plan is None:
                self._released_plan = LycheeAudioCudnnPlan(inputs.device)
            # The released stream encoder handles each400ms window separately.
            # Independent B1 execution preserves that math when a caller batches.
            rows = [self._released_plan.execute(row.unsqueeze(0), weight, bias) for row in inputs]
            return rows[0] if len(rows) == 1 else torch.cat(rows, dim=0)

    def close(self):
        with self._released_lock:
            if self._released_closed:
                return
            self._released_closed = True
            if self._released_plan is not None:
                self._released_plan.close()
                self._released_plan = None
