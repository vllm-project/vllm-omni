# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Test-only import fallback: lets pure-logic tests run without real ``vllm``.

Bypasses ``vllm_omni/__init__.py`` via a synthetic package in ``sys.modules``
and stubs the ``vllm`` tree with permissive ``_Dummy`` classes.  Active only
when real ``vllm`` is unavailable (CI no-op), so it cannot mask genuine
import problems.  Scope: pure-logic unit tests only.
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
    """Class-level attribute access on a ``_Dummy`` also yields ``_Dummy``."""

    def __getattr__(cls, name):
        return _Dummy


class _Dummy(metaclass=_Meta):
    """Permissive stand-in for any vllm symbol (import-grade compatibility)."""

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
