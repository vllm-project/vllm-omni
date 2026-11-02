# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Import contracts for optional FlashAttention providers, not numeric tests.

Load the complete production module with a per-module import hook. Platform
selection and optional providers are controlled without modifying sys.modules,
global builtins, or another test's already-imported attention modules.
"""

import builtins
import importlib.util
import logging
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_FA_PATH = Path(__file__).resolve().parents[3] / "vllm_omni/diffusion/attention/backends/utils/fa.py"
_CUDA_PROVIDERS = (
    "vllm.vllm_flash_attn",
    "flash_attn_interface",
    "flash_attn",
    "flash_attn.flash_attn_interface",
)


class _Platform:
    def __init__(self, name):
        self.name = name

    def is_rocm(self):
        return self.name == "rocm"

    def is_xpu(self):
        return self.name == "xpu"

    def is_musa(self):
        return self.name == "musa"

    def is_cuda(self):
        return self.name == "cuda"

    def get_device_capability(self):
        return None


def _load_fa(platform, available_provider=None):
    platforms = ModuleType("vllm_omni.platforms")
    setattr(platforms, "current_omni_platform", _Platform(platform))
    logger = ModuleType("vllm.logger")
    setattr(logger, "init_logger", logging.getLogger)
    attempts = []
    provider = ModuleType("test_flash_attention_provider")

    def kernel(*args, **kwargs):
        raise AssertionError("An import contract must not execute a kernel")

    setattr(provider, "flash_attn_varlen_func", kernel)
    setattr(provider, "flash_attn_func", kernel)

    def isolated_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm_omni.platforms":
            return platforms
        if name == "vllm.logger":
            return logger
        if name in (*_CUDA_PROVIDERS, "vllm._aiter_ops", "vllm._xpu_ops"):
            attempts.append(name)
            if name == available_provider:
                return provider
            raise ImportError(f"Optional test provider {name} is absent")
        return builtins.__import__(name, globals, locals, fromlist, level)

    spec = importlib.util.spec_from_file_location("_isolated_fa_import_contract", _FA_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__builtins__"] = {**vars(builtins), "__import__": isolated_import}
    spec.loader.exec_module(module)
    return module, attempts, kernel


@pytest.mark.parametrize("platform", ["cpu", "npu", "unspecified"])
def test_non_cuda_platforms_do_not_probe_cuda_providers(platform):
    module, attempts, _ = _load_fa(platform, available_provider="vllm.vllm_flash_attn")
    assert attempts == []
    assert module.flash_attn_func is None
    assert module.flash_attn_varlen_func is None
    assert not module.HAS_FLASH_ATTN


def test_cuda_keeps_vllm_dispatcher_preference():
    module, attempts, kernel = _load_fa("cuda", available_provider="vllm.vllm_flash_attn")
    assert attempts == ["vllm.vllm_flash_attn"]
    assert module.flash_attn_varlen_func is kernel
    assert module.HAS_FLASH_ATTN
    assert module.FLASH_ATTN_SUPPORTS_FP8


def test_cuda_keeps_optional_provider_discovery_order():
    module, attempts, _ = _load_fa("cuda")
    assert attempts == list(_CUDA_PROVIDERS)
    assert not module.HAS_FLASH_ATTN


@pytest.mark.parametrize(
    "platform,provider",
    [("rocm", "vllm._aiter_ops"), ("xpu", "vllm._xpu_ops"), ("musa", "flash_attn_interface")],
)
def test_other_accelerators_keep_their_discovery_branch(platform, provider):
    module, attempts, _ = _load_fa(platform)
    assert attempts == [provider]
    assert not module.HAS_FLASH_ATTN
