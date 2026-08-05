# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for diffusion attention backend selection."""

from types import SimpleNamespace

import pytest

import vllm_omni.diffusion.attention.selector as selector
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _ConfiguredBackend:
    @classmethod
    def get_name(cls) -> str:
        return "CONFIGURED"


class _PlatformBackend:
    @classmethod
    def get_name(cls) -> str:
        return "PLATFORM"


@pytest.fixture(autouse=True)
def clear_selector_caches():
    selector._cached_get_backend_cls.cache_clear()
    selector._log_backend_resolution.cache_clear()
    yield
    selector._cached_get_backend_cls.cache_clear()
    selector._log_backend_resolution.cache_clear()


@pytest.mark.parametrize(
    ("role", "role_category", "expected_backend"),
    [
        ("ltx2.audio_to_video", "cross", "EXACT"),
        ("ltx2.video_to_audio", "cross", "CATEGORY"),
        ("joint", None, "DEFAULT"),
    ],
)
def test_configured_selection_precedence(
    monkeypatch,
    role: str,
    role_category: str | None,
    expected_backend: str,
):
    config = AttentionConfig(
        default=AttentionSpec(backend="DEFAULT"),
        per_role={
            "cross": AttentionSpec(backend="CATEGORY"),
            "ltx2.audio_to_video": AttentionSpec(backend="EXACT"),
        },
    )
    calls = []

    def fake_get_backend(backend_name, head_size, allow_trtllm_default=True):
        calls.append((backend_name, head_size, allow_trtllm_default))
        return _ConfiguredBackend

    monkeypatch.setattr(selector, "_cached_get_backend_cls", fake_get_backend)

    backend, spec = selector.get_attn_backend_for_role(
        role=role,
        role_category=role_category,
        head_size=128,
        attention_config=config,
    )

    assert backend is _ConfiguredBackend
    assert spec is not None
    assert spec.backend == expected_backend
    assert calls == [(expected_backend, 128, True)]


@pytest.mark.parametrize("attention_config", [None, AttentionConfig()])
def test_platform_default_used_without_resolved_spec(monkeypatch, attention_config):
    calls = []

    def fake_get_backend(backend_name, head_size, allow_trtllm_default=True):
        calls.append((backend_name, head_size, allow_trtllm_default))
        return _PlatformBackend

    monkeypatch.setattr(selector, "_cached_get_backend_cls", fake_get_backend)

    backend, spec = selector.get_attn_backend_for_role(
        role="self",
        head_size=64,
        attention_config=attention_config,
        allow_trtllm_default=False,
    )

    assert backend is _PlatformBackend
    assert spec is None
    assert calls == [(None, 64, False)]


def test_backend_class_resolution_is_cached(monkeypatch):
    platform_calls = []
    load_calls = []
    fake_platform = SimpleNamespace(
        get_diffusion_attn_backend_cls=lambda **kwargs: (platform_calls.append(kwargs) or "fake.module.Backend")
    )

    monkeypatch.setattr("vllm_omni.platforms.current_omni_platform", fake_platform)

    def fake_load_backend(path):
        load_calls.append(path)
        return _ConfiguredBackend

    monkeypatch.setattr(selector, "_load_backend_cls", fake_load_backend)

    first = selector._cached_get_backend_cls("FLASH_ATTN", 128, False)
    second = selector._cached_get_backend_cls("FLASH_ATTN", 128, False)

    assert first is second is _ConfiguredBackend
    assert platform_calls == [
        {
            "selected_backend": "FLASH_ATTN",
            "head_size": 128,
            "allow_trtllm_default": False,
        }
    ]
    assert load_calls == ["fake.module.Backend"]


def test_load_backend_cls_reports_missing_module():
    with pytest.raises(ImportError, match="Failed to import module missing_attention_backend"):
        selector._load_backend_cls("missing_attention_backend.Backend")


def test_load_backend_cls_reports_missing_class(monkeypatch):
    monkeypatch.setattr(selector.importlib, "import_module", lambda _: SimpleNamespace())

    with pytest.raises(AttributeError, match="Class MissingBackend not found in module"):
        selector._load_backend_cls("fake.module.MissingBackend")
