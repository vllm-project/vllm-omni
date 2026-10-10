# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""F0-owned released FP32 convolution and activation math on qualified CUDA.

Plans use Torch's loaded cuDNN library and current stream. Model instances own
bounded caches; CPU, training, gradients and unsupported configurations delegate
Torch's existing model-owned implementations without changing global flags.
"""

from __future__ import annotations

import ctypes as C
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from threading import RLock

import torch
from torch import nn
from vllm.triton_utils import tl, triton
from vllm.triton_utils import tldevice as libdevice

from .convolution import Conv1d

RELEASED_KNOBS = {"TILE_SIZE": 8, "TILEK": 1, "STAGES": 4, "IDX_MODE": 0, "LDGC": 0, "SPECFILT": 0}
KNOB_TYPES = {"TILE_SIZE": 2, "TILEK": 13, "STAGES": 14, "IDX_MODE": 18, "LDGC": 22, "SPECFILT": 23}
TYPE_CTYPE = {
    0: C.c_void_p,
    1: C.c_int,
    2: C.c_bool,
    3: C.c_int64,
    4: C.c_float,
    6: C.c_void_p,
    7: C.c_int,
    9: C.c_int,
    15: C.c_void_p,
}
MAX_WORKSPACE_BYTES = 16 * 1024**2
MAX_CACHED_PLANS = 4
F0_LIBDEVICE = Path("/usr/local/cuda-12.9/nvvm/libdevice/libdevice.10.bc")
# Actual18 plus labeled cropped-real-mel module replays qualify even L2..58.
# Reachable final L60/62 and other inputs retain the existing delegate.
QUALIFIED_LENGTHS = frozenset(range(2, 59, 2))


def validate_plan_contract(shape, weight_shape, config, engine, knobs):
    shape, weight_shape = tuple(shape), tuple(weight_shape)
    if len(shape) != 3 or shape[0] != 1 or shape[1] not in (80, 512) or shape[2] not in QUALIFIED_LENGTHS:
        raise ValueError("Unqualified F0 convolution shape")
    if weight_shape != (512, shape[1], 3):
        raise ValueError("Unqualified F0 weight shape")
    if any(config[key] != [value] for key, value in (("stride", 1), ("padding", 1), ("dilation", 1))):
        raise ValueError("Unqualified F0 convolution configuration")
    if config["groups"] != 1 or engine != 38 or knobs != RELEASED_KNOBS:
        raise ValueError("Unqualified F0 engine or knobs")
    return shape, weight_shape


def _qualified_runtime(inputs):
    return (
        inputs.is_cuda
        and inputs.dtype == torch.float32
        and not torch.is_autocast_enabled("cuda")
        and str(torch.__version__) == "2.13.0+cu132"
        and torch.version.cuda == "13.2"
        and torch._C._cudnn.getVersionInt() == 92000
        and torch.cuda.get_device_capability(inputs.device) == (8, 0)
        and torch.backends.cudnn.enabled
        and torch.backends.cudnn.allow_tf32
    )


def _needs_grad(*values):
    return torch.is_grad_enabled() and any(value is not None and value.requires_grad for value in values)


class _F0Plan:
    padding = 1

    @staticmethod
    def validate_shapes(shape, weight_shape):
        config = {"stride": [1], "padding": [1], "dilation": [1], "groups": 1}
        return validate_plan_contract(shape, weight_shape, config, 38, RELEASED_KNOBS)

    def __init__(self, shape, weight_shape, device):
        engine, knobs = 38, RELEASED_KNOBS
        self.device = torch.device(device)
        self.last_event = None
        self.last_stream = None
        self.shape, self.weight_shape = self.validate_shapes(shape, weight_shape)
        torch._C._cudnn.getVersionInt()
        mapped = sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if line.split() and line.split()[-1].endswith("/libcudnn.so.9")
            }
        )
        if len(mapped) != 1:
            raise RuntimeError("Expected exactly Torch's one loaded cuDNN main library")
        self.library_path = mapped[0]
        self.lib = C.CDLL(self.library_path)
        self.lib.cudnnGetVersion.restype = C.c_size_t
        self.lib.cudnnGetErrorString.restype = C.c_char_p
        self.lib.cudnnGetErrorString.argtypes = [C.c_int]
        self.version = self.lib.cudnnGetVersion()
        if self.version != 92000:
            raise RuntimeError("Installed-header FP32 plan requires Torch-loaded cuDNN92000")
        ptr = C.c_void_p
        self.lib.cudnnBackendCreateDescriptor.argtypes = [C.c_int, C.POINTER(ptr)]
        self.lib.cudnnBackendSetAttribute.argtypes = [ptr, C.c_int, C.c_int, C.c_int64, ptr]
        self.lib.cudnnBackendGetAttribute.argtypes = [ptr, C.c_int, C.c_int, C.c_int64, C.POINTER(C.c_int64), ptr]
        self.lib.cudnnBackendFinalize.argtypes = [ptr]
        self.lib.cudnnBackendDestroyDescriptor.argtypes = [ptr]
        self.lib.cudnnCreate.argtypes = [C.POINTER(ptr)]
        self.lib.cudnnDestroy.argtypes = [ptr]
        self.lib.cudnnSetStream.argtypes = [ptr, ptr]
        self.lib.cudnnBackendExecute.argtypes = [ptr, ptr, ptr]
        self.descriptors, self.handle, self.workspace = [], ptr(), None
        self.knobs = knobs
        try:
            self.check(self.lib.cudnnCreate(C.byref(self.handle)), "create handle")
            length, channels = self.shape[-1], self.shape[1]
            x = self.fp32_tensor(120, [1, channels, 1, length], [channels * length, length, length, 1])
            kernel = self.weight_shape[-1]
            w = self.fp32_tensor(119, [512, channels, 1, kernel], [channels * kernel, kernel, kernel, 1])
            y = self.fp32_tensor(121, [1, 512, 1, length], [512 * length, length, length, 1])
            conv = self.create(1)
            for attr, typ, values in [
                (100, 1, [0]),
                (101, 7, [1]),
                (106, 3, [2]),
                (102, 3, [1, 1]),
                (103, 3, [1, 1]),
                (104, 3, [0, self.padding]),
                (105, 3, [0, self.padding]),
            ]:
                self.set(conv, attr, typ, values)
            self.finalize(conv)
            operation = self.create(10)
            for attr, desc in [(702, conv), (703, w), (704, x), (705, y)]:
                self.set(operation, attr, 15, [desc.value])
            self.set(operation, 700, 4, [1.0])
            self.set(operation, 701, 4, [0.0])
            self.finalize(operation)
            graph = self.create(15)
            self.set(graph, 800, 0, [self.handle.value])
            self.set(graph, 801, 15, [operation.value])
            self.finalize(graph)
            selected = self.create(2)
            self.set(selected, 1300, 15, [graph.value])
            self.set(selected, 1301, 3, [engine])
            self.finalize(selected)
            choices = []
            for knob, value in knobs.items():
                choice = self.create(7)
                self.set(choice, 600, 9, [KNOB_TYPES[knob]])
                self.set(choice, 601, 3, [value])
                self.finalize(choice)
                choices.append(choice.value)
            engine_config = self.create(3)
            self.set(engine_config, 300, 15, [selected.value])
            self.set(engine_config, 302, 15, choices)
            self.finalize(engine_config)
            self.plan = self.create(5)
            self.set(self.plan, 400, 0, [self.handle.value])
            self.set(self.plan, 401, 15, [engine_config.value])
            self.finalize(self.plan)
            self.workspace_bytes = self.get_int64(self.plan, 402)
            if not 0 <= self.workspace_bytes <= MAX_WORKSPACE_BYTES:
                raise RuntimeError("Released F0 plan workspace exceeds16MiB bound")
            self.workspace = None
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:
                error._lychee_failed_plan = self
                error._lychee_cleanup_errors = [cleanup_error]
            raise

    def fp32_tensor(self, uid, dimensions, strides):
        descriptor = self.create(17)
        for attr, typ, values in [
            (900, 3, [32]),
            (901, 1, [0]),
            (902, 3, dimensions),
            (903, 3, strides),
            (906, 3, [uid]),
            (907, 2, [False]),
        ]:
            self.set(descriptor, attr, typ, values)
        self.finalize(descriptor)
        return descriptor

    def check(self, status, operation):
        if status:
            raise RuntimeError(
                f"Lychee F0 cuDNN {operation}: {self.lib.cudnnGetErrorString(status).decode()} ({status})"
            )

    def create(self, kind):
        result = C.c_void_p()
        self.check(self.lib.cudnnBackendCreateDescriptor(kind, C.byref(result)), f"create {kind}")
        self.descriptors.append(result)
        return result

    def set(self, descriptor, attribute, value_type, values):
        array = (TYPE_CTYPE[value_type] * len(values))(*values)
        self.check(
            self.lib.cudnnBackendSetAttribute(descriptor, attribute, value_type, len(values), array), f"set {attribute}"
        )

    def finalize(self, descriptor):
        self.check(self.lib.cudnnBackendFinalize(descriptor), "finalize")

    def get_int64(self, descriptor, attribute):
        count, result = C.c_int64(), C.c_int64()
        self.check(
            self.lib.cudnnBackendGetAttribute(descriptor, attribute, 3, 1, C.byref(count), C.byref(result)),
            f"get {attribute}",
        )
        if count.value != 1:
            raise RuntimeError("F0 plan workspace attribute count differs")
        return result.value

    def allocate_workspace(self):
        self.workspace = torch.empty(self.workspace_bytes, device=self.device, dtype=torch.uint8)

    def execute(self, inputs, weight, bias):
        if (
            tuple(inputs.shape) != self.shape
            or tuple(weight.shape) != self.weight_shape
            or bias.shape != (512,)
            or self.workspace is None
        ):
            raise ValueError("F0 plan input/weight/bias/workspace differs")
        tensors = (inputs, weight, bias)
        if any(value.dtype != torch.float32 or not value.is_cuda or value.device != self.device for value in tensors):
            raise ValueError("F0 plan requires CUDA FP32 tensors on its owning device")
        with torch.accelerator.device_index(self.device.index):
            stream = torch.cuda.current_stream(self.device)
            if self.last_event is not None:
                stream.wait_event(self.last_event)
            x, w = inputs.contiguous(), weight.contiguous()
            y = torch.empty((1, 512, self.shape[-1]), device=self.device, dtype=torch.float32)
            for value in (x, w, bias, y, self.workspace):
                value.record_stream(stream)
            self.check(self.lib.cudnnSetStream(self.handle, stream.cuda_stream), "set current stream")
            pack = self.create(16)
            primary = None
            try:
                self.set(pack, 1000, 3, [120, 119, 121])
                self.set(pack, 1001, 6, [x.data_ptr(), w.data_ptr(), y.data_ptr()])
                self.set(pack, 1003, 6, [self.workspace.data_ptr() if self.workspace_bytes else 0])
                self.finalize(pack)
                self.last_stream = stream
                self.check(self.lib.cudnnBackendExecute(self.handle, self.plan, pack), "execute released plan")
                return y + bias.reshape(1, -1, 1)
            except BaseException as error:
                primary = error
                raise
            finally:
                # Even an execution error can have queued device work. Order only
                # this plan's stream before its workspace can be reused or freed.
                failures = []
                try:
                    event = torch.cuda.Event()
                    event.record(stream)
                    self.last_event = event
                    self.last_stream = None
                except Exception as error:
                    failures.append(error)
                try:
                    self.check(self.lib.cudnnBackendDestroyDescriptor(pack), "destroy variant pack")
                    self.descriptors.remove(pack)
                except Exception as error:
                    # Keep failed descriptors owned so close can retry after
                    # the completion fence, even when execution already failed.
                    failures.append(error)
                if failures:
                    if primary is not None:
                        primary._lychee_cleanup_errors = getattr(primary, "_lychee_cleanup_errors", []) + failures
                    else:
                        failures[0]._lychee_cleanup_errors = failures[1:]
                        raise failures[0]

    def close(self):
        # A failed completion fence proves nothing about queued work. Retain
        # every owned resource for an explicit close retry on the same owner.
        if self.last_stream is not None:
            self.last_stream.synchronize()
        elif self.last_event is not None:
            self.last_event.synchronize()
        self.last_event = None
        self.last_stream = None
        failure = None
        for descriptor in list(reversed(self.descriptors)):
            try:
                self.check(self.lib.cudnnBackendDestroyDescriptor(descriptor), "destroy owned descriptor")
            except Exception as error:
                failure = failure or error
            else:
                self.descriptors.remove(descriptor)
        if not self.descriptors and self.handle:
            try:
                self.check(self.lib.cudnnDestroy(self.handle), "destroy owned handle")
            except Exception as error:
                failure = failure or error
            else:
                self.handle = C.c_void_p()
        if not self.descriptors and not self.handle:
            self.workspace = None
        if failure is not None:
            raise failure


class _ReleasedConv1d(Conv1d):
    """Preserve Conv1d/weight_norm state with a bounded released-math cache."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._f0_plans = OrderedDict()
        self._f0_failed_plans = []
        self._f0_lock = RLock()

    def _make_plan(self, shape, weight_shape, device):
        return _F0Plan(shape, weight_shape, device)

    def _retain_failed_plan(self, plan):
        if all(owner is not plan for owner in self._f0_failed_plans):
            self._f0_failed_plans.append(plan)

    def _close_after_error(self, plan, error):
        try:
            plan.close()
        except Exception as cleanup_error:
            self._retain_failed_plan(plan)
            error._lychee_cleanup_errors = getattr(error, "_lychee_cleanup_errors", []) + [cleanup_error]
        else:
            self._f0_failed_plans = [owner for owner in self._f0_failed_plans if owner is not plan]
            for key in [key for key, owner in self._f0_plans.items() if owner is plan]:
                del self._f0_plans[key]

    def _conv_forward(self, inputs, weight, bias):
        if not self._eligible(inputs, weight, bias):
            return super()._conv_forward(inputs, weight, bias)
        key = (inputs.device, tuple(inputs.shape), tuple(weight.shape))
        with self._f0_lock, torch.accelerator.device_index(inputs.device.index):
            if self._f0_failed_plans:
                raise RuntimeError("Released convolution has failed owners; clear plans before reuse")
            plan = self._f0_plans.get(key)
            if plan is None:
                try:
                    plan = self._make_plan(inputs.shape, weight.shape, inputs.device)
                except BaseException as error:
                    failed = getattr(error, "_lychee_failed_plan", None)
                    if failed is not None:
                        self._retain_failed_plan(failed)
                    raise
                try:
                    while self._f0_plans and (
                        len(self._f0_plans) >= MAX_CACHED_PLANS
                        or sum(value.workspace_bytes for value in self._f0_plans.values()) + plan.workspace_bytes
                        > MAX_WORKSPACE_BYTES
                    ):
                        victim_key, victim = next(iter(self._f0_plans.items()))
                        try:
                            victim.close()
                        except Exception:
                            self._retain_failed_plan(victim)
                            raise
                        del self._f0_plans[victim_key]
                    plan.allocate_workspace()
                except BaseException as error:
                    self._close_after_error(plan, error)
                    raise
                self._f0_plans[key] = plan
            self._f0_plans.move_to_end(key)
            try:
                return plan.execute(inputs, weight, bias)
            except BaseException as error:
                self._close_after_error(plan, error)
                raise

    def clear_released_plans(self):
        failure = None
        with self._f0_lock:
            owners = list(self._f0_plans.values()) + self._f0_failed_plans
            seen = set()
            for plan in owners:
                if id(plan) in seen:
                    continue
                seen.add(id(plan))
                try:
                    with torch.accelerator.device_index(plan.device.index):
                        plan.close()
                except Exception as error:
                    self._retain_failed_plan(plan)
                    failure = failure or error
                else:
                    for key in [key for key, owner in self._f0_plans.items() if owner is plan]:
                        del self._f0_plans[key]
                    self._f0_failed_plans = [owner for owner in self._f0_failed_plans if owner is not plan]
        if failure is not None:
            raise failure

    def _apply(self, fn, recurse=True):
        self.clear_released_plans()
        return super()._apply(fn, recurse=recurse)

    def __deepcopy__(self, memo):
        # Parametrized modules otherwise copy __dict__ directly, including live
        # locks and device resources. Preserve their parameterized class/state.
        replica = self.__new__(type(self))
        memo[id(self)] = replica
        replica.__dict__ = deepcopy(
            {
                key: value
                for key, value in self.__dict__.items()
                if key not in ("_f0_plans", "_f0_failed_plans", "_f0_lock")
            },
            memo,
        )
        replica._f0_plans = OrderedDict()
        replica._f0_failed_plans = []
        replica._f0_lock = RLock()
        return replica

    def __getstate__(self):
        state = super().__getstate__().copy()
        state.pop("_f0_plans", None)
        state.pop("_f0_failed_plans", None)
        state.pop("_f0_lock", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._f0_plans = OrderedDict()
        self._f0_failed_plans = []
        self._f0_lock = RLock()

    def __del__(self):
        if hasattr(self, "_f0_plans"):
            try:
                self.clear_released_plans()
            except Exception:
                pass


class F0Conv1d(_ReleasedConv1d):
    """F0 alone admits its qualified K3/p1 FP32 shapes."""

    def _eligible(self, inputs, weight, bias):
        return (
            not self.training
            and not _needs_grad(inputs, weight, bias)
            and _qualified_runtime(inputs)
            and inputs.ndim == 3
            and inputs.shape[0] == 1
            and inputs.shape[1] in (80, 512)
            and inputs.shape[2] in QUALIFIED_LENGTHS
            and tuple(weight.shape) == (512, inputs.shape[1], 3)
            and weight.dtype == inputs.dtype
            and weight.device == inputs.device
            and bias is not None
            and bias.shape == (512,)
            and bias.dtype == inputs.dtype
            and bias.device == inputs.device
            and self.groups == 1
            and self.stride == (1,)
            and self.padding == (1,)
            and self.dilation == (1,)
            and self.padding_mode == "zeros"
        )

    def _conv_forward(self, inputs, weight, bias):
        return super()._conv_forward(inputs, weight, bias)


def _released_f0_libdevice_path():
    # F0 qualification is for this fixed toolchain, independently of CUDA_HOME
    # or the AR activation override. Never silently select unqualified math.
    if not F0_LIBDEVICE.is_file():
        raise RuntimeError(f"Lychee released F0 math requires fixed CUDA12.9 libdevice: {F0_LIBDEVICE}")
    return str(F0_LIBDEVICE)


@triton.jit
def _elu_kernel(x_ptr, y_ptr, n: tl.constexpr, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    x = tl.load(x_ptr + index, index < n, 0).to(tl.float32)
    tl.store(y_ptr + index, tl.where(x > 0, x, libdevice.expm1(x)), index < n)


class F0ELU(nn.ELU):
    def forward(self, inputs):
        if (
            self.training
            or _needs_grad(inputs)
            or not _qualified_runtime(inputs)
            or self.alpha != 1
            or self.inplace
            or inputs.ndim != 3
            or inputs.shape[:2] != (1, 512)
            or inputs.shape[2] not in QUALIFIED_LENGTHS
            or not F0_LIBDEVICE.is_file()
        ):
            return super().forward(inputs)
        x = inputs.contiguous()
        result = torch.empty_like(x)
        if x.numel():
            _elu_kernel[(triton.cdiv(x.numel(), 1024),)](
                x,
                result,
                x.numel(),
                block=1024,
                enable_fp_fusion=False,
                extern_libs={"libdevice": _released_f0_libdevice_path()},
            )
        return result
