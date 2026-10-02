# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Session-scoped config and environment fixtures."""

from __future__ import annotations

import os

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

from vllm_omni.config import config_factory
from vllm_omni.quantization import factory


@pytest.fixture(scope="session", autouse=True)
def default_env():
    # Keep behavior but avoid import-time side effects (RFC #2299).
    keys = ("VLLM_WORKER_MULTIPROC_METHOD", "VLLM_TARGET_DEVICE")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = previous["VLLM_WORKER_MULTIPROC_METHOD"] or "spawn"
    if previous["VLLM_TARGET_DEVICE"]:
        pass  # already set, keep it
    elif torch.cuda.is_available() and torch.accelerator.device_count() > 0:
        os.environ["VLLM_TARGET_DEVICE"] = "cuda"
    elif hasattr(torch, "npu") and torch.npu.is_available():
        os.environ["VLLM_TARGET_DEVICE"] = "npu"
    else:
        os.environ["VLLM_TARGET_DEVICE"] = "cpu"
    yield
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(scope="session", autouse=True)
def default_vllm_config():
    """Set a default VllmConfig for the whole test session.

    Session scope ensures module-scoped fixtures (e.g. ``omni_runner``) and
    deferred imports of ``tests.helpers.runtime`` both see the same context.
    Function-scoped autouse ran too late for ``OmniRunner`` setup and could
    desynchronize vLLM init vs request preprocessing (e.g. renderer state).
    """
    # Use CPU device if no GPU is available (e.g., in CI environments)
    if torch.cuda.is_available() and torch.accelerator.device_count() > 0:
        device = "cuda"
    elif hasattr(torch, "npu") and torch.npu.is_available():
        device = "npu"
    else:
        device = "cpu"
    device_config = DeviceConfig(device=device)

    with set_current_vllm_config(VllmConfig(device_config=device_config)):
        yield


@pytest.fixture
def local_model_configs_only(monkeypatch):
    """Only read HF and checkpoint quantization configs from local model directories.
    If the model references a fake path, e.g., `test-model`, it resolves with no config
    or checkpoint quantization without touching HF Hub.
    """
    read = factory.read_checkpoint_quantization_config
    monkeypatch.setattr(
        factory,
        "read_checkpoint_quantization_config",
        lambda model, revision: read(model, revision) if os.path.isdir(model) else None,
    )
    get_config = config_factory.get_config

    def _get_config(model, *args, **kwargs):
        if not os.path.isdir(model):
            raise OSError(f"{model!r} is not a local model directory")
        return get_config(model, *args, **kwargs)

    monkeypatch.setattr(config_factory, "get_config", _get_config)
