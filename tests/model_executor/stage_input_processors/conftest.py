# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Test-support import fallback for ``vllm_omni`` pure-logic tests.

These unit tests exercise ``vllm_omni`` helper logic without a running vLLM
runtime.  When the real ``vllm`` package is not installed, importing
``vllm_omni`` submodules would fail at ``import vllm``; this conftest then
installs a *test-only* fallback that:

1. Bypasses ``vllm_omni/__init__.py`` (so its ``patch`` / ``transformers``
   machinery is never executed) by pre-inserting a synthetic ``vllm_omni``
   package into ``sys.modules`` whose ``__path__`` points at the real
   package directory.
2. Stubs the entire ``vllm`` package tree with permissive ``_Dummy`` classes
   so module-level ``from vllm... import ...`` statements resolve.

The fallback is active **only** when real ``vllm`` is not importable.  When
vllm is available (e.g. on CI) it is a no-op and the real package is imported,
which guards against the fallback masking genuine import problems.

Scope: only pure-logic / unit-level tests (no model execution). Any test that
needs a real vLLM runtime must be marked to run on GPU/NPU CI instead.
"""

import importlib.abc
import importlib.machinery
import os
import sys
import types

# ---------------------------------------------------------------------------
# Locate the repo root (this file lives at:
#   <repo>/tests/model_executor/stage_input_processors/conftest.py)
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_VLLM_OMNI_DIR = os.path.join(_REPO_ROOT, "vllm_omni")


class _Meta(type):
    """Metaclass so that *class-level* attribute access on a _Dummy also
    yields _Dummy (e.g. ``SomeStubClass.foo`` used in module-level code)."""

    def __getattr__(cls, name):
        return _Dummy


class _Dummy(metaclass=_Meta):
    """Permissive stand-in for any vllm symbol.

    Supports subclassing (``__mro_entries__``), ``total=False`` TypedDict
    style (``__init_subclass__``), ``X | None`` unions, and arbitrary
    attribute / call / item access — enough to get module *import* past the
    vllm dependency while keeping pure helper functions callable.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Dummy()

    def __getattr__(self, name):
        return _Dummy

    def __class_getitem__(cls, item):
        return cls

    def __mro_entries__(self, bases):
        return (_Dummy,)

    def __init_subclass__(cls, **kwargs):
        pass

    def __or__(self, other):
        return _Dummy

    def __ror__(self, other):
        return _Dummy


class _StubLoader(importlib.abc.Loader):
    """Creates an empty module with a wildcard __getattr__ for any vllm.*."""

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []
        module.__getattr__ = lambda attr: _Dummy  # type: ignore[method-assign]
        return module

    def exec_module(self, module):
        pass


class _StubFinder(importlib.abc.MetaPathFinder):
    """Auto-creates a stub package for ``vllm`` and any ``vllm.*`` submodule."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vllm" or fullname.startswith("vllm."):
            return importlib.machinery.ModuleSpec(fullname, _StubLoader(), is_package=True)
        return None


def _install_shim():
    """Bypass the real vllm_omni/__init__ and stub the vllm package tree."""
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    # Synthetic vllm_omni parent package (skips the real __init__).
    pkg = types.ModuleType("vllm_omni")
    pkg.__path__ = [_VLLM_OMNI_DIR]
    pkg.__package__ = "vllm_omni"
    pkg._SHIM_ACTIVE = True  # type: ignore[attr-defined]
    sys.modules.setdefault("vllm_omni", pkg)
    # Wildcard vllm stub finder (meta path -> takes precedence for vllm.*).
    sys.meta_path.insert(0, _StubFinder())


def _activate_if_needed():
    """Install the import fallback only when real vllm is unavailable."""
    try:
        import vllm  # noqa: F401
    except ModuleNotFoundError:
        _install_shim()


_activate_if_needed()
