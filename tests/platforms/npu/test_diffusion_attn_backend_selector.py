# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the NPU diffusion attention backend selector.

Regression coverage for the eager ``mindiesd`` import in
``NPUOmniPlatform.get_diffusion_attn_backend_cls``: the import must run only
for backends that reach mindiesd kernels (FLASH_ATTN, and RAINFUSION_ATTN via
its dense FlashAttention fallback), so that a broken optional mindiesd
install cannot block backends that never use it (e.g. TORCH_SDPA).

The platform module is loaded from source with fake vllm/vllm-ascend
dependencies, so the tests run on CPU without NPU or mindiesd installed.
"""

from __future__ import annotations

import builtins
import importlib.util
import sys
import types
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_BACKEND_NAMES = (
    "FLASH_ATTN",
    "FLASH_ATTN_HUB",
    "FLASH_ATTN_3_HUB",
    "TORCH_SDPA",
    "RAINFUSION_ATTN",
)


def _repo_root() -> Path:
    marker = Path("vllm_omni") / "platforms" / "npu" / "platform.py"
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).is_file():
            return parent
    raise FileNotFoundError(f"could not locate repo root containing {marker}")


def _install_fake_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _make_fake_backend_enum():
    members = {
        name: SimpleNamespace(get_path=lambda n=name: f"fake.backends.{n.lower()}.Backend") for name in _BACKEND_NAMES
    }

    class _FakeEnumMeta(type):
        def __getitem__(cls, name):
            return members[name]

        def __getattr__(cls, name):
            try:
                return members[name]
            except KeyError:
                raise AttributeError(name) from None

    class FakeDiffusionAttentionBackendEnum(metaclass=_FakeEnumMeta):
        pass

    return FakeDiffusionAttentionBackendEnum


class _FakeOmniPlatformEnum(Enum):
    CUDA = "cuda"
    NPU = "npu"


class _FakeNPUPlatform:
    pass


class _FakeOmniPlatform:
    _omni_enum = _FakeOmniPlatformEnum.NPU

    @classmethod
    def validate_diffusion_attn_backend(cls, selected_backend: str) -> None:
        pass


def _load_platform_module(monkeypatch: pytest.MonkeyPatch):
    """Load vllm_omni/platforms/npu/platform.py with fake dependencies."""
    _install_fake_module(monkeypatch, "vllm")
    _install_fake_module(
        monkeypatch,
        "vllm.logger",
        init_logger=lambda name: SimpleNamespace(
            debug=lambda *a, **k: None,
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
        ),
    )
    _install_fake_module(monkeypatch, "vllm_ascend")
    _install_fake_module(monkeypatch, "vllm_ascend.platform", NPUPlatform=_FakeNPUPlatform)
    for pkg in (
        "vllm_omni",
        "vllm_omni.diffusion",
        "vllm_omni.diffusion.attention",
        "vllm_omni.diffusion.attention.backends",
        "vllm_omni.platforms",
    ):
        _install_fake_module(monkeypatch, pkg)
    _install_fake_module(
        monkeypatch,
        "vllm_omni.diffusion.attention.backends.registry",
        DiffusionAttentionBackendEnum=_make_fake_backend_enum(),
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.interface",
        OmniPlatform=_FakeOmniPlatform,
        OmniPlatformEnum=_FakeOmniPlatformEnum,
    )

    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "platform.py"
    spec = importlib.util.spec_from_file_location("npu_platform_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NPUOmniPlatform


@pytest.fixture
def mindiesd_tracker(monkeypatch: pytest.MonkeyPatch):
    """Make mindiesd importable and record eager import attempts."""
    monkeypatch.delitem(sys.modules, "mindiesd", raising=False)
    attempts = []
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mindiesd":
            attempts.append(name)
            return types.ModuleType("mindiesd")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name, *a, **k: object() if name == "mindiesd" else None,
    )
    return attempts


@pytest.mark.parametrize(
    ("selected_backend", "expect_mindiesd_import"),
    [
        ("FLASH_ATTN", True),
        # HF-hub variants fall back to local FLASH_ATTN on NPU.
        ("FLASH_ATTN_HUB", True),
        ("FLASH_ATTN_3_HUB", True),
        ("RAINFUSION_ATTN", True),
        # Backends that never reach mindiesd must not trigger the import.
        ("TORCH_SDPA", False),
    ],
)
def test_eager_mindiesd_import_scoped_to_mindiesd_backends(
    monkeypatch: pytest.MonkeyPatch,
    mindiesd_tracker: list[str],
    selected_backend: str,
    expect_mindiesd_import: bool,
):
    platform_cls = _load_platform_module(monkeypatch)

    path = platform_cls.get_diffusion_attn_backend_cls(selected_backend, head_size=64)

    assert (len(mindiesd_tracker) > 0) is expect_mindiesd_import
    expected = "FLASH_ATTN" if selected_backend in ("FLASH_ATTN_HUB", "FLASH_ATTN_3_HUB") else selected_backend
    assert path == f"fake.backends.{expected.lower()}.Backend"


def test_explicit_flash_attn_without_mindiesd_still_resolves(
    monkeypatch: pytest.MonkeyPatch,
):
    platform_cls = _load_platform_module(monkeypatch)
    monkeypatch.delitem(sys.modules, "mindiesd", raising=False)
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a, **k: None)

    path = platform_cls.get_diffusion_attn_backend_cls("FLASH_ATTN", head_size=64)

    assert path == "fake.backends.flash_attn.Backend"


def test_default_backend_prefers_flash_attn_when_mindiesd_available(
    monkeypatch: pytest.MonkeyPatch,
    mindiesd_tracker: list[str],
):
    platform_cls = _load_platform_module(monkeypatch)

    path = platform_cls.get_diffusion_attn_backend_cls(None, head_size=64)

    assert path == "fake.backends.flash_attn.Backend"
    assert len(mindiesd_tracker) > 0


def test_default_backend_falls_back_to_sdpa_without_mindiesd(
    monkeypatch: pytest.MonkeyPatch,
):
    platform_cls = _load_platform_module(monkeypatch)
    monkeypatch.delitem(sys.modules, "mindiesd", raising=False)
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a, **k: None)

    path = platform_cls.get_diffusion_attn_backend_cls(None, head_size=64)

    assert path == "fake.backends.torch_sdpa.Backend"
