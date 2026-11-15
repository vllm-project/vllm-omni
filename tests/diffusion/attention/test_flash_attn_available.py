# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
from types import ModuleType

import pytest

from vllm_omni.diffusion.attention.backends.utils import fa

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize(
    "installed, is_rocm, cdna, fa_triton, expected",
    [
        (False, False, False, False, False),  # not installed
        (True, False, False, False, True),  # CUDA
        (True, True, True, False, True),  # ROCm CDNA: CK kernels
        (True, True, False, False, False),  # ROCm RDNA: no CK kernels
        (True, True, False, True, True),  # ROCm RDNA: Triton backend enabled
    ],
)
def test_is_flash_attn_available(monkeypatch, installed, is_rocm, cdna, fa_triton, expected):
    # None in sys.modules makes `import flash_attn` raise ImportError.
    monkeypatch.setitem(sys.modules, "flash_attn", ModuleType("flash_attn") if installed else None)
    monkeypatch.setattr(fa.current_omni_platform, "is_rocm", lambda: is_rocm)
    rocm = ModuleType("vllm.platforms.rocm")
    setattr(rocm, "on_cdna", lambda: cdna)
    setattr(rocm, "flash_attn_triton_available", lambda: fa_triton)
    monkeypatch.setitem(sys.modules, "vllm.platforms.rocm", rocm)

    assert fa.is_flash_attn_available() is expected
