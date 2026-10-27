# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Weak, unique-storage liveness accounting for controller tensor operations."""

import weakref
from contextlib import contextmanager

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode, _pop_mode_temporarily
from torch.utils._pytree import tree_flatten


class TensorLedger(TorchDispatchMode):
    """Observe controller tensor allocations without retaining them.

    Existing state is seeded before each pair; native model callbacks are
    suspended because their outputs belong to the pipeline.
    """

    def __init__(self, state=None):
        super().__init__()
        self.refs = {}
        self.peak = 0
        self.suspended = False
        self.seed(state)

    def seed(self, obj):
        if isinstance(obj, torch.Tensor):
            self.observe(obj)
        elif hasattr(obj, "__dataclass_fields__"):
            for name in obj.__dataclass_fields__:
                self.seed(getattr(obj, name))
        elif isinstance(obj, (tuple, list)):
            for value in obj:
                self.seed(value)
        elif isinstance(obj, dict):
            for value in obj.values():
                self.seed(value)

    @property
    def active_bytes(self):
        storages = {}
        expired = []
        for key, (reference, storage_key, size) in self.refs.items():
            if reference() is None:
                expired.append(key)
            else:
                storages[storage_key] = size
        for key in expired:
            del self.refs[key]
        return sum(storages.values())

    def observe(self, tensor):
        storage = tensor.untyped_storage()
        self.refs[id(tensor)] = (weakref.ref(tensor), (str(tensor.device), storage.data_ptr()), storage.nbytes())
        self.peak = max(self.peak, self.active_bytes)

    @contextmanager
    def native(self):
        """Suspend accounting while preserving other modes and nested suspension."""
        suspended = self.suspended
        self.suspended = True
        try:
            if _get_current_dispatch_mode() is self:
                # Remove only our topmost mode to avoid dispatch overhead in model forwards.
                with _pop_mode_temporarily():
                    yield
            else:
                # Another mode owns the stack top; keep its ordering and only skip accounting.
                yield
        finally:
            self.suspended = suspended

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        if not self.suspended:
            for value in tree_flatten(output)[0]:
                if isinstance(value, torch.Tensor):
                    self.observe(value)
        return output
