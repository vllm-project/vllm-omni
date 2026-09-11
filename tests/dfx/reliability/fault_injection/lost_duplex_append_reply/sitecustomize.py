# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Test-only fault injector for a committed duplex append's lost reply.

This directory is prepended to ``PYTHONPATH`` only for the matching reliability
server.  Python imports ``sitecustomize`` before the server entry point, so the
import hook can patch the correlated transport without adding a production
environment-variable failpoint.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import threading
from types import ModuleType

_TARGET = "vllm_omni.engine.rpc_result_router"


def _patch_correlated_client(module: ModuleType) -> None:
    client_cls = module.CorrelatedRpcClient
    if getattr(client_cls, "_duplex_lost_reply_injected", False):
        return

    original_execute = client_cls.execute
    injection_lock = threading.Lock()
    successful_appends = 0

    def execute(self, key, message, **kwargs):
        nonlocal successful_appends
        result = original_execute(self, key, message, **kwargs)
        should_inject = (
            isinstance(key, tuple)
            and key[0] == "duplex"
            and type(message).__name__ == "AppendDuplexInputMessage"
            and bool(getattr(message, "operation_id", None))
        )
        if should_inject:
            with injection_lock:
                successful_appends += 1
                if successful_appends == 2:
                    print(
                        "[reliability][lost-reply] dropping committed scheduler append reply",
                        flush=True,
                    )
                    raise TimeoutError("injected committed duplex append reply loss")
        return result

    client_cls.execute = execute
    client_cls._duplex_lost_reply_injected = True


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped) -> None:
        self._wrapped = wrapped

    def create_module(self, spec):
        create_module = getattr(self._wrapped, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module) -> None:
        self._wrapped.exec_module(module)
        _patch_correlated_client(module)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        del target
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        sys.meta_path.remove(self)
        spec.loader = _PatchLoader(spec.loader)
        return spec


if _TARGET in sys.modules:
    _patch_correlated_client(sys.modules[_TARGET])
else:
    sys.meta_path.insert(0, _PatchFinder())
