# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test fixtures for RFC #4590 disaggregated diffusion.

The torch-free foundation modules (``stage_roles``, ``stage_payload``, and the
generic ``diffusion`` transition processor) are loaded directly by file path so
these unit tests run without importing the full ``vllm_omni`` package or torch —
useful on machines without the GPU runtime. Tests that require torch/vllm are
marked ``needs_runtime`` and skipped when those deps are absent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# This file lives at <repo>/tests/diffusion/disaggregated/conftest.py, so the
# vllm_omni package root is <repo>/vllm_omni.
VLLM_OMNI_ROOT = Path(__file__).resolve().parents[3] / "vllm_omni"


def _load_module_by_path(mod_name: str, rel_path: str):
    """Load a single module file by path, bypassing package __init__ imports."""
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = VLLM_OMNI_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def stage_roles():
    return _load_module_by_path("_rfc4590_stage_roles", "diffusion/stage_roles.py")


@pytest.fixture(scope="session")
def stage_payload():
    # Register under the REAL dotted name so interface.py's lazy ``_transport()``
    # (which checks ``sys.modules["vllm_omni.diffusion.stage_payload"]``) resolves
    # the torch-free leaf without importing the ``vllm_omni`` package (torch).
    return _load_module_by_path("vllm_omni.diffusion.stage_payload", "diffusion/stage_payload.py")


@pytest.fixture(scope="session")
def diffusion_processor():
    return _load_module_by_path(
        "_rfc4590_diffusion_processor",
        "model_executor/stage_input_processors/diffusion.py",
    )


@pytest.fixture(scope="session")
def interface_mod():
    # interface.py imports torch only under TYPE_CHECKING, so it loads torch-free.
    # Pre-register the stage_payload leaf under its real dotted name so interface's
    # lazy ``_transport()`` resolves it without importing the vllm_omni package.
    _load_module_by_path("vllm_omni.diffusion.stage_payload", "diffusion/stage_payload.py")
    return _load_module_by_path("_rfc4590_interface", "diffusion/models/interface.py")


# A fabricated base whose (module, name) == ("torch", "Tensor") so the payload
# module's duck-typed ``_is_tensor`` check accepts instances without importing
# torch. Defined at class-creation time (reassigning __bases__ later is fragile).
_TorchTensorBase = type("Tensor", (), {})
_TorchTensorBase.__module__ = "torch"


class FakeTensor(_TorchTensorBase):
    """Minimal torch.Tensor stand-in for torch-free payload tests.

    Records detach/cpu/contiguous/clone calls so sanitize behavior is
    observable, and reports a ``data_ptr`` so the alias-avoiding clone path in
    :func:`sanitize_transport_tensor` can be exercised.
    """

    def __init__(self, shape=(2, 3), dtype="float32", data_ptr=1000):
        self.shape = shape
        self.dtype = dtype
        self._data_ptr = data_ptr
        self.calls: list[str] = []

    def detach(self):
        self.calls.append("detach")
        return self

    def cpu(self):
        self.calls.append("cpu")
        return self

    def contiguous(self):
        self.calls.append("contiguous")
        return self

    def clone(self):
        self.calls.append("clone")
        return FakeTensor(self.shape, self.dtype, data_ptr=self._data_ptr + 1)

    def data_ptr(self):
        return self._data_ptr


@pytest.fixture
def fake_tensor_cls():
    return FakeTensor
