# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import sys
from collections.abc import Generator
from types import ModuleType

import pytest


def _is_vllm_related(name: str) -> bool:
    return name == "vllm" or name.startswith("vllm.") or name == "vllm_omni" or name.startswith("vllm_omni.")


@pytest.fixture
def memory_profiling_module() -> Generator[ModuleType, None, None]:
    """Load memory_profiling module for testing, with cleanup.

    Saves all vllm/vllm_omni modules (including bare parents) before loading,
    then restores them afterward so this test does not affect others.
    """
    original_modules: dict[str, ModuleType | None] = {}
    for name in list(sys.modules.keys()):
        if _is_vllm_related(name):
            original_modules[name] = sys.modules.pop(name)

    # Use importlib.import_module so Python resolves the module normally.
    # Fall back to skip if vllm_omni is not installed (e.g., in environments
    # where the full package is not importable).
    import importlib

    try:
        module = importlib.import_module("vllm_omni.diffusion.memory_profiling")
    except ModuleNotFoundError:
        pytest.skip("vllm_omni.diffusion.memory_profiling is not available")

    yield module

    # Cleanup: remove any vllm/vllm_omni modules created during the test.
    for name in list(sys.modules.keys()):
        if _is_vllm_related(name):
            sys.modules.pop(name, None)
    # Restore original state.
    for name, mod in original_modules.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


def test_memory_log_env_var_name_is_stable(memory_profiling_module):
    assert memory_profiling_module.get_memory_log_env_var() == "VLLM_OMNI_DIFFUSION_LOG_MEMORY"


def test_memory_profiling_disabled_by_default(monkeypatch, memory_profiling_module):
    monkeypatch.delenv("VLLM_OMNI_DIFFUSION_LOG_MEMORY", raising=False)
    logger = logging.getLogger("vllm_omni.core.sched.omni_generation_scheduler")
    original_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        assert memory_profiling_module.is_memory_profiling_enabled() is False
    finally:
        logger.setLevel(original_level)


def test_memory_profiling_truthy_values(monkeypatch, memory_profiling_module):
    monkeypatch.setenv("VLLM_OMNI_DIFFUSION_LOG_MEMORY", "true")
    assert memory_profiling_module.is_memory_profiling_enabled() is True


def test_memory_profiling_enabled_by_debug_log_level(monkeypatch, memory_profiling_module):
    monkeypatch.delenv("VLLM_OMNI_DIFFUSION_LOG_MEMORY", raising=False)
    logger = logging.getLogger("vllm_omni.core.sched.omni_generation_scheduler")
    original_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        assert memory_profiling_module.is_memory_profiling_enabled() is True
    finally:
        logger.setLevel(original_level)


def test_format_cuda_memory_snapshot_none(memory_profiling_module):
    assert memory_profiling_module.format_cuda_memory_snapshot(None) == "cuda=unavailable"
