# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CUDA check for the standalone flash_attn package."""

import sys
from types import ModuleType

import pytest
from vllm.platforms import current_platform

# Importing CudaOmniPlatform imports CUDA-only vLLM extensions.
if not current_platform.is_cuda():
    pytest.skip("CUDA-only platform tests", allow_module_level=True)

from vllm_omni.platforms.cuda.platform import CudaOmniPlatform  # noqa: E402

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "gpu_name,installed,expected",
    [
        ("NVIDIA A100-SXM4-80GB", True, True),
        ("Tesla T4", True, False),
        ("NVIDIA A100-SXM4-80GB", False, False),
    ],
    ids=["ampere", "turing", "not-installed"],
)
def test_cuda_uses_flash_attn_gpu_check(
    monkeypatch: pytest.MonkeyPatch, gpu_name: str, installed: bool, expected: bool
) -> None:
    # None in sys.modules makes `import flash_attn` raise ImportError.
    monkeypatch.setitem(sys.modules, "flash_attn", ModuleType("flash_attn") if installed else None)
    monkeypatch.setattr(CudaOmniPlatform, "get_device_name", classmethod(lambda cls, device_id=0: gpu_name))

    assert CudaOmniPlatform.supports_standalone_flash_attn() is expected
