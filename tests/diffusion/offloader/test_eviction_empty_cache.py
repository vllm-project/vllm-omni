# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The per-eviction ``empty_cache`` must not run on XPU.

Returning the freed segments to the driver is not what makes the HBM reusable
-- the caching allocator already recycles those blocks. On XPU it does have a
cost: it churns the addresses of the collective receive buffers allocated
after it, and the XPU collective backend keeps a non-reclaimable driver
registration per distinct receive-buffer address, so device memory outside the
PyTorch pool was observed growing across the measured requests.

These tests pin the platform split: XPU skips the call, every other platform
keeps the previous behaviour byte for byte.
"""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.offloader.module_residency import (
    PinnedModuleStager,
    _StorageGroup,
    _TensorBinding,
)
from vllm_omni.diffusion.offloader.sequential_backend import SequentialOffloadHook
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


@pytest.fixture
def platform_probe(monkeypatch):
    """Drive ``is_xpu`` and count ``empty_cache`` without touching a device."""
    calls: list[str] = []

    def _install(*, is_xpu: bool):
        monkeypatch.setattr(current_omni_platform, "is_xpu", lambda: is_xpu, raising=False)
        monkeypatch.setattr(current_omni_platform, "empty_cache", lambda: calls.append("empty_cache"), raising=False)
        monkeypatch.setattr(current_omni_platform, "synchronize", lambda: calls.append("synchronize"), raising=False)
        return calls

    return _install


def _hook_on_cpu_target() -> tuple[SequentialOffloadHook, nn.Module]:
    module = nn.Linear(4, 4)
    hook = SequentialOffloadHook([module], torch.device("cpu"), pin_memory=False)
    return hook, module


def _staged_module() -> PinnedModuleStager:
    """A stager in the ``loaded`` state without requiring an accelerator."""
    stager = object.__new__(PinnedModuleStager)
    target = torch.zeros(4)
    master = torch.zeros(4).view(torch.uint8)
    binding = _TensorBinding(target=target, dtype=target.dtype, shape=(4,), stride=(1,), storage_offset=0)
    stager.loaded = True
    stager.cache_retention = None
    stager._groups = [_StorageGroup(master=master, bindings=[binding])]
    stager._device_storages = [torch.zeros(16, dtype=torch.uint8)]
    return stager


@pytest.mark.parametrize("is_xpu, expected", [(True, 0), (False, 1)])
def test_stager_offload_skips_empty_cache_only_on_xpu(platform_probe, is_xpu, expected) -> None:
    calls = platform_probe(is_xpu=is_xpu)
    stager = _staged_module()

    stager.offload()

    assert stager.loaded is False
    assert calls.count("synchronize") == 1, "the stage-boundary sync must run on every platform"
    assert calls.count("empty_cache") == expected


@pytest.mark.parametrize("is_xpu, expected", [(True, 0), (False, 1)])
def test_sequential_to_cpu_skips_empty_cache_only_on_xpu(platform_probe, monkeypatch, is_xpu, expected) -> None:
    calls = platform_probe(is_xpu=is_xpu)
    hook, module = _hook_on_cpu_target()
    # ``_to_cpu`` returns early when the module already lives on CPU, which
    # would make the assertion pass for the wrong reason. ``meta`` gives a
    # non-CPU device without needing an accelerator; the transfer itself is
    # stubbed out because meta storage cannot be copied.
    module.to(torch.device("meta"))
    monkeypatch.setattr(
        SequentialOffloadHook,
        "_move_params",
        # ``_move_params`` reports whether anything was evicted; the eviction
        # ``empty_cache`` is only reached when it did, so the stub must say so.
        staticmethod(lambda *args, **kwargs: calls.append("move_params") or True),
    )

    hook._to_cpu(module)

    assert calls.count("move_params") == 1, "the eviction itself must still happen"
    assert calls.count("empty_cache") == expected


def test_stager_forced_release_still_flushes_on_xpu(platform_probe) -> None:
    """Failure paths keep their explicit flush; only the eviction path is gated."""
    calls = platform_probe(is_xpu=True)
    stager = _staged_module()

    stager._release_cache(force=True)

    assert calls.count("empty_cache") == 1
