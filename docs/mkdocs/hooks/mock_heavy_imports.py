# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Hook to mock heavy packages before any module import during docs build.

ReadTheDocs build environment has limited RAM (~7 GB). When mkdocstrings/griffe
falls back to dynamic inspection (import), loading torch + vllm + CUDA libraries
exhausts memory and the build is OOM-killed.

This hook installs sys.modules stubs and a meta-path finder to intercept imports
of heavy packages, replacing them with lightweight MagicMock objects.
"""

from __future__ import annotations

import logging
import sys
from importlib.machinery import ModuleSpec
from unittest.mock import MagicMock

logger = logging.getLogger("mkdocs")

# Packages that pull in native extensions / large ML libraries.
# Any sub-module of these is also intercepted.
HEAVY_PACKAGES = [
    "torch",
    "torchaudio",
    "torchvision",
    "torchcre",
    "vllm",
    "vllm_ascend",
    "vllm_platform",
    "transformers",
    "diffusers",
    "accelerate",
    "einops",
    "triton",
    "tritonserver",
    "xformers",
    "flash_attn",
    "flashinfer",
    "cuda",
    "cuda_toolkit",
    "nvidia",
    "numba",
    "onnx",
    "onnxruntime",
    "megablocks",
    "stability",
    "sentencepiece",
    "tokenizers",
    "tiktoken",
    "safetensors",
    "soundfile",
    "librosa",
    "scipy",
    "sklearn",
    "cv2",
    "decord",
    "av",
    "PIL",
    "pillow",
    "pynvml",
    "pycuda",
    "cutlass",
    "mamba_ssm",
    "causal_conv1d",
    "timm",
    "open_clip",
    "zmq",
    "janus",
    "msgspec",
    "numpy",
    "aenum",
]


def _make_mock_module(fullname: str, loader: _HeavyImportsLoader | None = None) -> MagicMock:
    """Create a MagicMock that looks enough like a module to satisfy import machinery."""
    mock = MagicMock()
    mock.__name__ = fullname
    mock.__file__ = f"<mocked {fullname}>"
    mock.__loader__ = loader
    mock.__package__ = fullname.rsplit(".", 1)[0] if "." in fullname else None
    mock.__path__ = [f"<mocked {fullname}>"]
    # MagicMock raises AttributeError for dunder names like __spec__ in Py3.12+,
    # so we must explicitly set it.  Without a valid __spec__, the import system
    # cannot traverse sub-modules (it needs __spec__.submodule_search_locations).
    mock.__spec__ = ModuleSpec(fullname, loader, is_package=True)
    mock.__spec__.submodule_search_locations = [f"<mocked {fullname}>"]
    return mock


def _install_mock(name: str) -> MagicMock:
    mock = _make_mock_module(name)
    sys.modules[name] = mock
    return mock


class _HeavyImportsLoader:
    """Modern loader that replaces the module with a MagicMock during exec_module."""

    def __init__(self, fullname: str) -> None:
        self.fullname = fullname

    def create_module(self, spec):
        return None  # Let importlib create a default module

    def exec_module(self, module) -> None:
        mock = _make_mock_module(self.fullname, loader=self)
        sys.modules[self.fullname] = mock


class _HeavyImportsFinder:
    """Meta-path finder that intercepts heavy packages using find_spec protocol."""

    def find_spec(self, fullname: str, path=None, target=None):
        # Already a MagicMock in sys.modules — nothing to do
        existing = sys.modules.get(fullname)
        if existing is not None:
            if isinstance(existing, MagicMock):
                return None
            # If a real (pre-allocated) module exists but not mocked yet,
            # we still intercept to replace it with a mock.
            for pkg in HEAVY_PACKAGES:
                if fullname == pkg or fullname.startswith(pkg + "."):
                    loader = _HeavyImportsLoader(fullname)
                    return ModuleSpec(fullname, loader, is_package=True)
            return None

        for pkg in HEAVY_PACKAGES:
            if fullname == pkg or fullname.startswith(pkg + "."):
                loader = _HeavyImportsLoader(fullname)
                return ModuleSpec(fullname, loader, is_package=True)
        return None


def on_startup(*args, **kwargs) -> None:
    logger.info("Mocking heavy packages for docs build (torch, vllm, transformers, …)")

    # Pre-populate top-level packages so "import torch" hits sys.modules first.
    for pkg in HEAVY_PACKAGES:
        if pkg not in sys.modules:
            _install_mock(pkg)

    # Install a meta-path finder to catch sub-modules (e.g. torch.nn, vllm.engine)
    # that haven't been pre-populated.
    sys.meta_path.insert(0, _HeavyImportsFinder())

    logger.info("Heavy package mocking installed")
