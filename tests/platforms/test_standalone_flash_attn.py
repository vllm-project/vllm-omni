# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Platform check for the standalone flash_attn package."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_omni.platforms.interface import OmniPlatform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _set_flash_attn(monkeypatch: pytest.MonkeyPatch, installed: bool) -> None:
    # None in sys.modules makes `import flash_attn` raise ImportError.
    monkeypatch.setitem(sys.modules, "flash_attn", ModuleType("flash_attn") if installed else None)


@pytest.mark.parametrize("installed", [True, False])
def test_default_checks_import(monkeypatch: pytest.MonkeyPatch, installed: bool) -> None:
    _set_flash_attn(monkeypatch, installed)

    assert OmniPlatform.supports_standalone_flash_attn() is installed


@pytest.mark.parametrize(
    "installed,cdna,fa_triton,expected",
    [
        (True, True, False, True),
        (True, False, False, False),
        (True, False, True, True),
        (False, True, False, False),
    ],
    ids=["cdna", "rdna", "rdna-triton", "not-installed"],
)
def test_rocm_follows_vit_rule(
    monkeypatch: pytest.MonkeyPatch,
    installed: bool,
    cdna: bool,
    fa_triton: bool,
    expected: bool,
) -> None:
    # ROCm import reads the architecture; these policy tests need no device.
    with patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(gcnArchName="gfx942")):
        import vllm.platforms.rocm as rocm

        from vllm_omni.platforms.rocm.platform import RocmOmniPlatform

    _set_flash_attn(monkeypatch, installed)
    monkeypatch.setattr(rocm, "on_cdna", lambda: cdna)
    monkeypatch.setattr(rocm, "flash_attn_triton_available", lambda: fa_triton)

    assert RocmOmniPlatform.supports_standalone_flash_attn() is expected
