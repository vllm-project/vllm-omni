# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-device 310P wiring tests.

``test_310p_patches.py`` covers the same contract with fake dependency modules,
which pins the import path but cannot qualify device behavior. These tests run
on a real 310P card and assert the chain end to end: the runtime reports 310P,
``is_310p()`` agrees, and the module-level branches it gates actually select the
310P worker, model runner and patches.

The branches are evaluated at import time, so a detection regression silently
leaves a 310P host on the generic worker/runner with no 310P patch applied.
"""

from __future__ import annotations

import pytest

pytest.importorskip("vllm_ascend")

import torch_npu
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType, device_type_from_runtime_soc

from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.core_model, pytest.mark.omni, *hardware_marks(res={"npu": "310P"}, num_cards=1)]


@pytest.fixture(scope="module", autouse=True)
def require_real_310p() -> None:
    """Skip rather than fail if the runner is not actually a 310P card."""
    runtime = device_type_from_runtime_soc(torch_npu.npu.get_soc_version())
    if runtime is not AscendDeviceType._310P:
        pytest.skip(f"not a 310P host (runtime device type: {runtime})")


def test_runtime_and_package_agree_on_310p() -> None:
    runtime = device_type_from_runtime_soc(torch_npu.npu.get_soc_version())
    assert runtime is AscendDeviceType._310P
    # vllm-ascend resolves the device from build info; a mismatch means the
    # installed package was built for other hardware.
    assert get_ascend_device_type() is AscendDeviceType._310P


def test_omni_detects_310p_and_not_a5() -> None:
    from vllm_omni.platforms.npu import is_a5
    from vllm_omni.platforms.npu._310p import is_310p

    assert is_310p() is True
    assert is_a5() is False


def test_worker_base_selects_310p_worker() -> None:
    from vllm_ascend._310p.worker_310p import NPUWorker310

    from vllm_omni.platforms.npu.worker.base import OmniNPUWorkerBase

    assert NPUWorker310 in OmniNPUWorkerBase.__mro__


def test_model_runner_selects_310p_runner() -> None:
    from vllm_ascend._310p.model_runner_310p import NPUModelRunner310

    from vllm_omni.platforms.npu.worker.npu_model_runner import OmniNPUModelRunner

    assert NPUModelRunner310 in OmniNPUModelRunner.__mro__


def test_worker_patch_is_applied() -> None:
    from vllm_omni.platforms.npu._310p import apply_patches
    from vllm_omni.platforms.npu._310p import patch as patch_registry

    apply_patches()
    assert patch_registry._WORKER_PATCHED is True
