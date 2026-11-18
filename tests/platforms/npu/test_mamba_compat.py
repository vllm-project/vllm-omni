# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import builtins
from typing import Any

import pytest

from vllm_omni.platforms.npu.mamba_compat import _install_prefill_end_shim

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("error_type", [ImportError, RuntimeError])
def test_mamba_import_only_ignores_missing_dependencies(monkeypatch, error_type):
    from vllm_omni.platforms.npu import mamba_compat

    original_import = builtins.__import__
    error = error_type("Ascend allocator import failed")

    def import_module(name, *args, **kwargs):
        if name == "vllm_ascend.patch.platform.patch_mamba_manager":
            raise error
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", import_module)
        if error_type is ImportError:
            mamba_compat._install_ascend_mamba_prefill_end_compat()
        else:
            with pytest.raises(RuntimeError) as caught:
                mamba_compat._install_ascend_mamba_prefill_end_compat()
            assert caught.value is error


def test_prefill_end_shim_wraps_only_the_old_signature():
    class _Old:
        def get_num_blocks_to_allocate(self, num_tokens: int) -> int:
            return num_tokens

    class _Named:
        def get_num_blocks_to_allocate(self, num_tokens: int, prefill_end: int = 0) -> int:
            return num_tokens + prefill_end

    class _VarKeyword:
        def get_num_blocks_to_allocate(self, num_tokens: int, **kwargs: Any) -> int:
            return num_tokens + int(kwargs["prefill_end"])

    assert _install_prefill_end_shim(_Old) is True
    # The wrapper accepts the keyword the original method rejects.
    assert getattr(_Old(), "get_num_blocks_to_allocate")(4, prefill_end=7) == 4
    # A second install must not stack another wrapper around the one that already drops the argument.
    assert _install_prefill_end_shim(_Old) is False
    assert getattr(_Old(), "get_num_blocks_to_allocate")(4, prefill_end=7) == 4

    assert _install_prefill_end_shim(_Named) is False
    assert _Named().get_num_blocks_to_allocate(4, prefill_end=7) == 11

    assert _install_prefill_end_shim(_VarKeyword) is False
    assert _VarKeyword().get_num_blocks_to_allocate(4, prefill_end=7) == 11


def test_only_npu_platform_installs_mamba_compat(monkeypatch):
    from vllm_omni.platforms.interface import OmniPlatform
    from vllm_omni.platforms.npu import mamba_compat

    pytest.importorskip("vllm_ascend")
    from vllm_omni.platforms.npu.platform import NPUOmniPlatform

    calls = []
    monkeypatch.setattr(mamba_compat, "_install_ascend_mamba_prefill_end_compat", lambda: calls.append("npu"))
    OmniPlatform.prepare_stage_engine()
    assert calls == []
    NPUOmniPlatform.prepare_stage_engine()
    assert calls == ["npu"]
