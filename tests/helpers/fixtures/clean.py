# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Cleanup fixtures for process-global test state.

Torch dtype/device restoration is autouse. GPU memory, the pipeline registry,
and the speaker cache stay opt-in.
"""

from __future__ import annotations

import pytest

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES


@pytest.fixture(autouse=True)
def restore_torch_default_dtype_and_device():
    """Restore process-global dtype and device after every test.

    ``get_default_device()`` reports ``cpu`` both when no default device was
    set and when ``cpu`` was set explicitly. Writing that value back enables
    default-device dispatch, so snapshot the active context instead.
    """
    import torch

    default_dtype = torch.get_default_dtype()
    device_context = getattr(torch._GLOBAL_DEVICE_CONTEXT, "device_context", None)
    default_device = None if device_context is None else torch.get_default_device()
    try:
        yield
    finally:
        torch.set_default_dtype(default_dtype)
        torch.set_default_device(default_device)


@pytest.fixture
def clean_gpu_memory_between_tests():
    """Opt-in GPU pre/post hooks for a test (no environment-variable gate).

    Use as a test parameter or ``@pytest.mark.usefixtures("clean_gpu_memory_between_tests")``.
    """
    from tests.helpers.clean import cleanup_test_environment

    print("\n=== PRE-TEST DEVICE CLEANUP ===")
    cleanup_test_environment()
    yield
    cleanup_test_environment()


@pytest.fixture
def clean_pipeline_registry():
    """Ensure the OMNI_PIPELINES are in a clean state for a test that mutates it."""
    snapshot = dict(OMNI_PIPELINES)
    yield
    OMNI_PIPELINES.clear()
    OMNI_PIPELINES.update(snapshot)


@pytest.fixture
def clean_speaker_cache():
    """Reset the process-wide speaker cache singleton before and after the test."""
    import vllm_omni.utils.speaker_cache as _sc

    def _reset():
        with _sc._SINGLETON_LOCK:
            if _sc._SINGLETON is not None:
                _sc._SINGLETON.clear()
            _sc._SINGLETON = None

    _reset()
    yield
    _reset()
