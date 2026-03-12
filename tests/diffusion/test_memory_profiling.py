# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from collections.abc import Generator
from pathlib import Path
from types import ModuleType

import pytest


# Find repo root by looking for pyproject.toml marker
def _find_repo_root(start: Path) -> Path:
    """Walk up from start to find repo root (contains pyproject.toml)."""
    current = start.resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise FileNotFoundError(f"Could not find repo root from {start}")


_REPO_ROOT = _find_repo_root(Path(__file__).resolve())
_MEMORY_PROFILING_PATH = _REPO_ROOT / "vllm_omni" / "diffusion" / "memory_profiling.py"


def _load_memory_profiling() -> ModuleType:
    """Load memory_profiling module without polluting sys.modules for other tests."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("vllm_omni.diffusion.memory_profiling", _MEMORY_PROFILING_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def memory_profiling_module() -> Generator[ModuleType, None, None]:
    """Load memory_profiling module for testing, with cleanup."""
    # Save any existing module state
    original_modules = {}
    test_module_name = "vllm_omni.diffusion.memory_profiling"
    if test_module_name in sys.modules:
        original_modules[test_module_name] = sys.modules.pop(test_module_name)

    # Also clean up parent packages
    for name in list(sys.modules.keys()):
        if name.startswith("vllm_omni.") and name != test_module_name:
            original_modules[name] = sys.modules.pop(name)

    # Load and yield
    module = _load_memory_profiling()
    yield module

    # Cleanup: restore original state
    sys.modules.pop(test_module_name, None)
    for name, mod in original_modules.items():
        sys.modules[name] = mod


def test_memory_log_env_var_name_is_stable(memory_profiling_module):
    assert memory_profiling_module.get_memory_log_env_var() == "VLLM_OMNI_DIFFUSION_LOG_MEMORY"


def test_memory_profiling_disabled_by_default(monkeypatch, memory_profiling_module):
    monkeypatch.delenv("VLLM_OMNI_DIFFUSION_LOG_MEMORY", raising=False)
    assert memory_profiling_module.is_memory_profiling_enabled() is False


def test_memory_profiling_truthy_values(monkeypatch, memory_profiling_module):
    monkeypatch.setenv("VLLM_OMNI_DIFFUSION_LOG_MEMORY", "true")
    assert memory_profiling_module.is_memory_profiling_enabled() is True


def test_format_cuda_memory_snapshot_none(memory_profiling_module):
    assert memory_profiling_module.format_cuda_memory_snapshot(None) == "cuda=unavailable"
