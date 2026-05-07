import os

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def default_env():
    # Keep behavior but avoid import-time side effects (RFC #2299).
    keys = ("VLLM_WORKER_MULTIPROC_METHOD", "VLLM_TARGET_DEVICE")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = previous["VLLM_WORKER_MULTIPROC_METHOD"] or "spawn"
    os.environ["VLLM_TARGET_DEVICE"] = previous["VLLM_TARGET_DEVICE"] or (
        "cuda" if torch.cuda.is_available() and torch.accelerator.device_count() > 0 else "cpu"
    )
    yield
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(scope="session")
def model_prefix() -> str:
    prefix = os.environ.get("MODEL_PREFIX", "")
    return f"{prefix.rstrip('/')}/" if prefix else ""


@pytest.fixture(autouse=True)
def clean_gpu_memory_between_tests():
    # Import here so ``tests.helpers.env`` (and vLLM platform modules) load only
    # after session autouse fixtures like ``default_env`` have run (RFC #2299).
    from tests.helpers.env import run_post_test_cleanup, run_pre_test_cleanup

    print("\n=== PRE-TEST GPU CLEANUP ===")
    run_pre_test_cleanup()
    yield
    run_post_test_cleanup()


@pytest.fixture(scope="session", autouse=True)
def default_vllm_config():
    """Set a default VllmConfig for the whole test session.

    Session scope ensures module-scoped fixtures (e.g. ``omni_runner``) and
    deferred imports of ``tests.helpers.runtime`` both see the same context.
    Function-scoped autouse ran too late for ``OmniRunner`` setup and could
    desynchronize vLLM init vs request preprocessing (e.g. renderer state).
    """
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    # Use CPU device if no GPU is available (e.g., in CI environments)
    has_gpu = torch.cuda.is_available() and torch.accelerator.device_count() > 0
    device = "cuda" if has_gpu else "cpu"
    device_config = DeviceConfig(device=device)

    with set_current_vllm_config(VllmConfig(device_config=device_config)):
        yield
