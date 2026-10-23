# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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

    def fake_get_cls(**kwargs):
        platform_calls.append(kwargs)
        return "fake.module.Backend"

    monkeypatch.setattr("vllm_omni.platforms.current_omni_platform.resolve_diffusion_attn_backend", fake_get_cls)

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


def test_capability_query_loads_explicit_cudnn_without_platform(monkeypatch):
    platform_calls = []

    def fake_get_cls(**kwargs):
        platform_calls.append(kwargs)
        return "fake.module.MustNotBeCalled"

    monkeypatch.setattr("vllm_omni.platforms.current_omni_platform.resolve_diffusion_attn_backend", fake_get_cls)

    config = AttentionConfig(default=AttentionSpec(backend="CUDNN_ATTN"))
    backend = selector.get_attn_backend_for_capability(role="self", attention_config=config)

    assert backend.get_name() == "CUDNN_ATTN"
    assert backend.supports_attention_mask()
    assert platform_calls == []


def test_capability_query_uses_unknown_head_size_for_platform_default(monkeypatch):
    calls = []

    def fake_get_backend(backend_name, head_size, allow_trtllm_default=True):
        calls.append((backend_name, head_size, allow_trtllm_default))
        return _PlatformBackend

    monkeypatch.setattr(selector, "_cached_get_backend_cls", fake_get_backend)

    backend = selector.get_attn_backend_for_capability(role="self", attention_config=AttentionConfig())

    assert backend is _PlatformBackend
    assert calls == [(None, selector.HEAD_SIZE_UNKNOWN, True)]


def test_load_backend_cls_reports_missing_module():
    with pytest.raises(ImportError, match="Failed to import module missing_attention_backend"):
        selector._load_backend_cls("missing_attention_backend.Backend")


def test_load_backend_cls_reports_missing_class(monkeypatch):
    monkeypatch.setattr(selector.importlib, "import_module", lambda _: SimpleNamespace())

    with pytest.raises(AttributeError, match="Class MissingBackend not found in module"):
        selector._load_backend_cls("fake.module.MissingBackend")


@pytest.fixture
def sparse_platform(monkeypatch):
    from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
    from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

    class Platform(OmniPlatform):
        _omni_enum = OmniPlatformEnum.CUDA

        @classmethod
        def get_diffusion_attn_backend_cls(cls, *args, **kwargs):
            pytest.fail("Sparse resolution must not run dense provider checks")

    calls = []

    class Adapter:
        @staticmethod
        def validate_selection(implementation, head_size):
            calls.append((implementation, head_size))

    class Backend(_ConfiguredBackend):
        @classmethod
        def get_block_sparse_adapter(cls):
            return Adapter

    monkeypatch.setattr(DiffusionAttentionBackendEnum, "get_class", lambda self: Backend)
    monkeypatch.setattr(DiffusionAttentionBackendEnum, "get_path", lambda self: "fake.module.Backend")
    monkeypatch.setattr("vllm_omni.platforms.current_omni_platform", Platform)
    return Platform, Backend, Adapter, calls


@pytest.mark.parametrize("capability_query", [False, True])
def test_sparse_resolution_uses_platform_and_preserves_method(monkeypatch, sparse_platform, capability_query):
    from vllm_omni.diffusion.attention.block_sparse import BlockSparseBackend

    platform, backend, _, calls = sparse_platform
    loaded = []

    def load(path):
        loaded.append(path)
        return backend

    monkeypatch.setattr(selector, "_load_backend_cls", load)
    config = AttentionConfig(
        default={
            "name": "block_sparse",
            "config": {"backend": {"require": "FLASH_ATTN", "implementation": "future-provider-id"}},
        }
    )
    if capability_query:
        result = selector.get_attn_backend_for_capability("self", config)
        assert result is BlockSparseBackend
    else:
        result, spec = selector.get_attn_backend_for_role("self", 192, config)
        assert result is backend
        assert spec is config.default
    assert calls == [("future-provider-id", -1 if capability_query else 192)]
    assert loaded == ["fake.module.Backend"]

    # A platform-specific rejection must be honored on both selection paths.
    def reject(cls, *args, **kwargs):
        raise ValueError("platform policy rejected sparse execution")

    monkeypatch.setattr(platform, "resolve_diffusion_attn_backend", classmethod(reject))
    with pytest.raises(ValueError, match="platform policy rejected"):
        if capability_query:
            selector.get_attn_backend_for_capability("self", config)
        else:
            selector.get_attn_backend_for_role("self", 192, config)


@pytest.mark.parametrize("platform_name", ["ROCM", "NPU", "XPU", "MUSA", "UNSPECIFIED"])
def test_sparse_platform_rejects_before_loading_provider(monkeypatch, sparse_platform, platform_name):
    from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
    from vllm_omni.platforms.interface import OmniPlatformEnum

    platform, _, _, calls = sparse_platform
    monkeypatch.setattr(platform, "_omni_enum", OmniPlatformEnum[platform_name])

    def unexpected_load(self):
        pytest.fail("Unsupported method/platform must fail before importing a provider")

    monkeypatch.setattr(DiffusionAttentionBackendEnum, "get_class", unexpected_load)
    with pytest.raises(ValueError, match="block_sparse requires CUDA"):
        platform.resolve_diffusion_attn_backend("FLASH_ATTN", 128, method="block_sparse")
    assert calls == []


def test_sparse_platform_requires_adapter(monkeypatch, sparse_platform):
    platform, backend, _, calls = sparse_platform
    monkeypatch.setattr(backend, "get_block_sparse_adapter", classmethod(lambda cls: None))
    with pytest.raises(ValueError, match="FLASH_ATTN: no adapter"):
        platform.resolve_diffusion_attn_backend("FLASH_ATTN", 128, method="block_sparse")
    assert calls == []


@pytest.mark.parametrize("error_type", [ImportError, ValueError, RuntimeError])
def test_sparse_platform_preserves_adapter_errors(monkeypatch, sparse_platform, error_type):
    platform, _, adapter, _ = sparse_platform
    error = error_type("provider failure")

    def fail(*args):
        raise error

    monkeypatch.setattr(adapter, "validate_selection", staticmethod(fail))
    with pytest.raises(error_type) as caught:
        platform.resolve_diffusion_attn_backend("FLASH_ATTN", 128, method="block_sparse")
    assert caught.value is error


@pytest.mark.parametrize("backend", [None, "FLASH_ATTN"])
@pytest.mark.parametrize("allow_default", [False, True])
def test_method_resolution_preserves_dense_platform_policy(backend, allow_default):
    from vllm_omni.platforms.interface import OmniPlatform

    calls = []

    class Platform(OmniPlatform):
        @classmethod
        def get_diffusion_attn_backend_cls(cls, selected_backend, head_size, allow_trtllm_default):
            calls.append((selected_backend, head_size, allow_trtllm_default))
            return "platform.override.Backend"

    assert Platform.resolve_diffusion_attn_backend(backend, 128, allow_default) == "platform.override.Backend"
    assert calls == [(backend, 128, allow_default)]


@pytest.mark.parametrize(
    ("backend", "method", "message"),
    [(None, "block_sparse", "requires an explicit backend"), ("FLASH_ATTN", "unknown", "Unknown diffusion")],
)
def test_method_resolution_rejects_invalid_requests(sparse_platform, backend, method, message):
    platform, _, _, calls = sparse_platform
    with pytest.raises(ValueError, match=message):
        platform.resolve_diffusion_attn_backend(backend, 128, method=method)
    assert calls == []
