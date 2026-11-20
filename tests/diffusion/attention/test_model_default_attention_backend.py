# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A model's per-GPU default attention backend sits between configured backends and the platform default."""

import pytest
from vllm.platforms.interface import DeviceCapability

import vllm_omni.diffusion.attention.selector as selector_mod
import vllm_omni.diffusion.config as config_mod
import vllm_omni.platforms as platforms_mod
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _ModelDefault:
    @staticmethod
    def get_name() -> str:
        return "MODEL_DEFAULT"


class _PlatformDefault:
    @staticmethod
    def get_name() -> str:
        return "PLATFORM_DEFAULT"


class _Configured:
    @staticmethod
    def get_name() -> str:
        return "CONFIGURED"


class _Platform:
    """The two platform queries the model default reads."""

    def __init__(self, *, cuda: bool, major: int) -> None:
        self._cuda = cuda
        self._major = major

    def is_cuda(self) -> bool:
        return self._cuda

    def get_device_capability(self) -> DeviceCapability:
        return DeviceCapability(major=self._major, minor=0)


@pytest.mark.parametrize("model", ["MiniMaxH3Pipeline", "MiniMaxH3ModularPipeline"])
def test_minimax_h3_defaults_to_flashinfer_on_sm10x_only(model: str) -> None:
    metadata = get_diffusion_model_metadata(model)
    assert metadata.default_attention_backend(10) == "FLASHINFER_ATTN"
    for major in (None, 8, 9, 12):
        assert metadata.default_attention_backend(major) is None


def test_models_without_a_default_keep_the_platform_default() -> None:
    assert get_diffusion_model_metadata("MiniMaxH3ImagePipeline").default_attention_backend(10) is None
    assert get_diffusion_model_metadata(None).default_attention_backend(10) is None


@pytest.mark.parametrize(
    ("model", "cuda", "major", "expected"),
    [
        ("MiniMaxH3Pipeline", True, 10, "FLASHINFER_ATTN"),
        ("MiniMaxH3Pipeline", True, 9, None),
        ("MiniMaxH3Pipeline", False, 10, None),
        ("MiniMaxH3ImagePipeline", True, 10, None),
        (None, True, 10, None),
    ],
)
def test_model_default_follows_the_configured_model_and_device(
    monkeypatch: pytest.MonkeyPatch, model: str | None, cuda: bool, major: int, expected: str | None
) -> None:
    monkeypatch.setattr(platforms_mod, "current_omni_platform", _Platform(cuda=cuda, major=major))
    config = OmniDiffusionConfig(model_class_name=model) if model is not None else None
    monkeypatch.setattr(config_mod, "get_current_diffusion_config_or_none", lambda: config)
    assert selector_mod._model_default_backend() == expected


def _fake_resolver(*, model_default_available: bool):
    calls = []

    def resolve(backend_name, head_size, allow_trtllm_default=True):
        calls.append(backend_name)
        if backend_name is None:
            return _PlatformDefault
        if backend_name == "FLASHINFER_ATTN":
            if not model_default_available:
                raise ValueError("FlashInfer is unavailable")
            return _ModelDefault
        return _Configured

    return resolve, calls


@pytest.fixture
def h3_on_sm10x(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(selector_mod, "_model_default_backend", lambda: "FLASHINFER_ATTN")


def test_model_default_replaces_the_platform_default(monkeypatch: pytest.MonkeyPatch, h3_on_sm10x: None) -> None:
    resolve, calls = _fake_resolver(model_default_available=True)
    monkeypatch.setattr(selector_mod, "_cached_get_backend_cls", resolve)
    backend, spec = selector_mod.get_attn_backend_for_role("self", 128)
    assert backend is _ModelDefault and spec is None
    assert calls == ["FLASHINFER_ATTN"]


def test_configured_backend_wins_over_the_model_default(monkeypatch: pytest.MonkeyPatch, h3_on_sm10x: None) -> None:
    resolve, calls = _fake_resolver(model_default_available=True)
    monkeypatch.setattr(selector_mod, "_cached_get_backend_cls", resolve)
    attention_config = OmniDiffusionConfig(
        diffusion_attention_config={"default": "TORCH_SDPA"}
    ).diffusion_attention_config
    backend, spec = selector_mod.get_attn_backend_for_role("self", 128, attention_config=attention_config)
    assert backend is _Configured and spec is not None and spec.backend == "TORCH_SDPA"
    assert calls == ["TORCH_SDPA"]


def test_unavailable_model_default_falls_back_to_the_platform_default(
    monkeypatch: pytest.MonkeyPatch, h3_on_sm10x: None
) -> None:
    resolve, calls = _fake_resolver(model_default_available=False)
    monkeypatch.setattr(selector_mod, "_cached_get_backend_cls", resolve)
    backend, spec = selector_mod.get_attn_backend_for_role("self", 128)
    assert backend is _PlatformDefault and spec is None
    assert calls == ["FLASHINFER_ATTN", None]


def test_without_a_model_default_the_platform_default_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(selector_mod, "_model_default_backend", lambda: None)
    resolve, calls = _fake_resolver(model_default_available=True)
    monkeypatch.setattr(selector_mod, "_cached_get_backend_cls", resolve)
    backend, spec = selector_mod.get_attn_backend_for_role("self", 128)
    assert backend is _PlatformDefault and spec is None
    assert calls == [None]
