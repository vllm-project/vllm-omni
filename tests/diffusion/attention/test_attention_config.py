# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for per-role attention backend configuration (RFC: per-role-attention-backend).

Tests cover:
- AttentionSpec and AttentionConfig normalization
- Role-aware backend resolution with category fallback
- OmniDiffusionConfig attention shorthand handling
- AttentionMetadata.extra field
"""

from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.attention.layer as layer_mod
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.config import (
    get_current_diffusion_config,
    get_current_diffusion_config_or_none,
    set_current_diffusion_config,
)
from vllm_omni.diffusion.data import (
    AttentionConfig,
    AttentionSpec,
    OmniDiffusionConfig,
    build_attention_config,
    parse_attention_config,
)


class TestAttentionSpec:
    def test_construct_no_extra(self):
        spec = AttentionSpec(backend="FLASH_ATTN")
        assert spec.extra == {}

    def test_mapping_extra_normalized(self):
        spec = AttentionSpec(backend="SAGE_ATTN", extra={"quant": "int8"})
        assert spec.backend == "SAGE_ATTN"
        assert spec.extra == {"quant": "int8"}

    def test_invalid_backend_type(self):
        with pytest.raises(TypeError):
            AttentionSpec(backend=123)  # type: ignore[arg-type]


class TestAttentionConfig:
    def test_empty_config(self):
        config = AttentionConfig()
        assert config.default is None
        assert config.per_role == {}

    def test_constructor_normalizes_mappings(self):
        config = AttentionConfig(
            default={"backend": "FLASH_ATTN"},
            per_role={
                "self": {"backend": "SPARSE_BLOCK", "extra": {"block_size": 128}},
                "cross": "SAGE_ATTN",
            },
        )
        assert config.default.backend == "FLASH_ATTN"
        assert config.per_role["self"].backend == "SPARSE_BLOCK"
        assert config.per_role["self"].extra == {"block_size": 128}
        assert config.per_role["cross"].backend == "SAGE_ATTN"

    def test_constructor_flattens_nested_per_role_tree(self):
        config = AttentionConfig(
            per_role={
                "ltx2": {
                    "audio_self": {"backend": "FLASH_ATTN"},
                    "audio_to_video": {"backend": "SAGE_ATTN"},
                }
            }
        )
        assert config.per_role["ltx2.audio_self"].backend == "FLASH_ATTN"
        assert config.per_role["ltx2.audio_to_video"].backend == "SAGE_ATTN"

    def test_constructor_normalizes_auto_to_unset(self):
        config = AttentionConfig(
            default={"backend": "auto"},
            per_role={
                "self": "auto",
                "cross": {"backend": "SAGE_ATTN"},
            },
        )
        assert config.default is None
        assert "self" not in config.per_role
        assert config.per_role["cross"].backend == "SAGE_ATTN"

    def test_resolve_exact_match(self):
        config = AttentionConfig(
            default=AttentionSpec(backend="FLASH_ATTN"),
            per_role={
                "self": AttentionSpec(backend="SPARSE_BLOCK"),
                "cross": AttentionSpec(backend="SAGE_ATTN"),
            },
        )
        spec, _ = config.resolve_with_source(role="self")
        assert spec.backend == "SPARSE_BLOCK"

        spec, _ = config.resolve_with_source(role="cross")
        assert spec.backend == "SAGE_ATTN"

    def test_resolve_with_source_reports_match_origin(self):
        config = AttentionConfig(
            default=AttentionSpec(backend="FLASH_ATTN"),
            per_role={
                "cross": AttentionSpec(backend="SAGE_ATTN"),
                "ltx2.audio_to_video": AttentionSpec(backend="SPARSE_BLOCK"),
            },
        )

        spec, source = config.resolve_with_source(role="ltx2.audio_to_video", role_category="cross")
        assert spec is not None
        assert spec.backend == "SPARSE_BLOCK"
        assert source == "attention_config.per_role['ltx2.audio_to_video']"

        spec, source = config.resolve_with_source(role="ltx2.video_to_audio", role_category="cross")
        assert spec is not None
        assert spec.backend == "SAGE_ATTN"
        assert source == "attention_config.per_role['cross'] (role_category fallback)"

        spec, source = config.resolve_with_source(role="self")
        assert spec is not None
        assert spec.backend == "FLASH_ATTN"
        assert source == "attention_config.default"

    def test_resolve_category_fallback(self):
        config = AttentionConfig(
            default=AttentionSpec(backend="FLASH_ATTN"),
            per_role={
                "cross": AttentionSpec(backend="SAGE_ATTN"),
            },
        )
        # "ltx2.audio_to_video" falls back to category "cross"
        spec, _ = config.resolve_with_source(role="ltx2.audio_to_video", role_category="cross")
        assert spec.backend == "SAGE_ATTN"

    def test_resolve_exact_overrides_category(self):
        config = AttentionConfig(
            per_role={
                "cross": AttentionSpec(backend="SAGE_ATTN"),
                "ltx2.audio_to_video": AttentionSpec(backend="FLASH_ATTN"),
            },
        )
        # Exact match wins over category
        spec, _ = config.resolve_with_source(role="ltx2.audio_to_video", role_category="cross")
        assert spec.backend == "FLASH_ATTN"

    def test_resolve_default_fallback(self):
        config = AttentionConfig(
            default=AttentionSpec(backend="FLASH_ATTN"),
        )
        spec, _ = config.resolve_with_source(role="self")
        assert spec.backend == "FLASH_ATTN"

        spec, _ = config.resolve_with_source(role="joint")
        assert spec.backend == "FLASH_ATTN"

    def test_resolve_returns_none_when_empty(self):
        config = AttentionConfig()
        spec, _ = config.resolve_with_source(role="self")
        assert spec is None

    def test_resolve_no_category_no_default(self):
        config = AttentionConfig(
            per_role={"self": AttentionSpec(backend="SPARSE_BLOCK")},
        )
        # Unknown role with no category and no default
        spec, _ = config.resolve_with_source(role="joint")
        assert spec is None

    def test_full_ltx2_scenario(self):
        """Test the LTX2 6-role stress test from the RFC."""
        config = AttentionConfig(
            default=AttentionSpec(backend="FLASH_ATTN"),
            per_role={
                "self": AttentionSpec(backend="SPARSE_BLOCK", extra={"block_size": 128}),
                "cross": AttentionSpec(backend="SAGE_ATTN"),
                "ltx2.audio_self": AttentionSpec(backend="FLASH_ATTN"),
                "ltx2.audio_to_video": AttentionSpec(backend="FLASH_ATTN", extra={"causal_window": 64}),
            },
        )

        # video self → exact match "self"
        assert config.resolve_with_source("self")[0].backend == "SPARSE_BLOCK"

        # audio self → exact match "ltx2.audio_self"
        assert config.resolve_with_source("ltx2.audio_self", "self")[0].backend == "FLASH_ATTN"

        # video-text cross → exact match "cross"
        assert config.resolve_with_source("cross")[0].backend == "SAGE_ATTN"

        # audio-text cross → category fallback to "cross"
        assert config.resolve_with_source("ltx2.audio_text_cross", "cross")[0].backend == "SAGE_ATTN"

        # audio-to-video → exact match
        spec, _ = config.resolve_with_source("ltx2.audio_to_video", "cross")
        assert spec.backend == "FLASH_ATTN"
        assert spec.extra == {"causal_window": 64}

        # video-to-audio → category fallback to "cross"
        assert config.resolve_with_source("ltx2.video_to_audio", "cross")[0].backend == "SAGE_ATTN"


class TestAttentionMetadataExtra:
    def test_default_extra_is_empty(self):
        meta = AttentionMetadata()
        assert meta.extra == {}

    def test_extra_passthrough(self):
        block_mask = torch.ones(4, 4)
        meta = AttentionMetadata(extra={"block_mask": block_mask, "kv_indices": [0, 1, 2]})
        assert torch.equal(meta.extra["block_mask"], block_mask)
        assert meta.extra["kv_indices"] == [0, 1, 2]

    def test_extra_does_not_affect_existing_fields(self):
        mask = torch.ones(2, 8)
        meta = AttentionMetadata(attn_mask=mask, extra={"foo": "bar"})
        assert meta.attn_mask is mask
        assert meta.extra == {"foo": "bar"}


class TestBuildAttentionConfig:
    def test_env_sets_default_when_no_higher_priority_input(self, monkeypatch):
        monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")

        config = build_attention_config()

        assert config.default is not None
        assert config.default.backend == "TORCH_SDPA"

    def test_attention_backend_overrides_env(self, monkeypatch):
        monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")

        config = parse_attention_config(attention_backend="SAGE_ATTN")

        assert config.default is not None
        assert config.default.backend == "SAGE_ATTN"

    def test_parse_attention_config_does_not_read_env(self, monkeypatch):
        monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")

        config = parse_attention_config()

        assert config.default is None

    def test_attention_backend_auto_disables_env_fallback(self, monkeypatch):
        monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")

        config = parse_attention_config(attention_backend="auto")

        assert config.default is None

    def test_explicit_default_ignores_env(self, monkeypatch):
        monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "self=FLASH_ATTN,cross=TORCH_SDPA")

        config = build_attention_config(
            AttentionConfig(default=AttentionSpec(backend="FLASH_ATTN")),
        )

        assert config.default is not None
        assert config.default.backend == "FLASH_ATTN"

    def test_env_auto_does_not_set_default(self, monkeypatch):
        monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "auto")

        config = build_attention_config()

        assert config.default is None

    def test_attention_backend_conflicts_with_explicit_default(self):
        with pytest.raises(ValueError):
            parse_attention_config(
                AttentionConfig(default=AttentionSpec(backend="FLASH_ATTN")),
                attention_backend="SAGE_ATTN",
            )


class TestOmniDiffusionConfigAttentionParsing:
    """Test OmniDiffusionConfig attention shorthand and structured config."""

    def test_diffusion_attention_backend_sets_default(self):
        config = OmniDiffusionConfig.from_kwargs(diffusion_attention_backend="SAGE_ATTN")
        assert isinstance(config.diffusion_attention_config, AttentionConfig)
        assert config.diffusion_attention_config.default is not None
        assert config.diffusion_attention_config.default.backend == "SAGE_ATTN"

    def test_diffusion_attention_backend_auto_means_platform_default(self):
        config = OmniDiffusionConfig.from_kwargs(diffusion_attention_backend="auto")
        assert isinstance(config.diffusion_attention_config, AttentionConfig)
        assert config.diffusion_attention_config.default is None

    def test_diffusion_attention_backend_and_default_are_mutually_exclusive(self):
        with pytest.raises(ValueError):
            OmniDiffusionConfig.from_kwargs(
                diffusion_attention_backend="SAGE_ATTN",
                diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="FLASH_ATTN")),
            )

    def test_dict_diffusion_attention_config(self):
        config = OmniDiffusionConfig(
            diffusion_attention_config={
                "default": {"backend": "FLASH_ATTN"},
                "per_role": {"self": "SPARSE_BLOCK"},
            }
        )
        assert config.diffusion_attention_config.default.backend == "FLASH_ATTN"
        assert config.diffusion_attention_config.per_role["self"].backend == "SPARSE_BLOCK"

    def test_no_diffusion_attention_config_defaults_to_empty(self):
        config = OmniDiffusionConfig()
        assert isinstance(config.diffusion_attention_config, AttentionConfig)
        assert config.diffusion_attention_config.default is None
        assert config.diffusion_attention_config.per_role == {}


class TestCurrentDiffusionConfig:
    def test_get_current_diffusion_config_or_none_defaults_to_none(self):
        assert get_current_diffusion_config_or_none() is None

    def test_get_current_diffusion_config_raises_when_unset(self):
        with pytest.raises(AssertionError, match="Diffusion config is not set"):
            get_current_diffusion_config()

    def test_set_current_diffusion_config_restores_previous_value(self):
        outer = SimpleNamespace(name="outer")
        inner = SimpleNamespace(name="inner")

        with set_current_diffusion_config(outer):
            assert get_current_diffusion_config() is outer
            with set_current_diffusion_config(inner):
                assert get_current_diffusion_config() is inner
            assert get_current_diffusion_config() is outer

        assert get_current_diffusion_config_or_none() is None


class TestAttentionInitUsesCurrentDiffusionConfig:
    def test_attention_init_uses_current_diffusion_config_without_forward_context(self, monkeypatch):
        class _FakeAttentionImpl:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def forward(self, query, key, value, attn_metadata=None):
                return query

        class _FakeBackend:
            @staticmethod
            def get_name() -> str:
                return "FAKE_BACKEND"

            @staticmethod
            def get_impl_cls():
                return _FakeAttentionImpl

        captured = {}

        def _fake_get_attn_backend_for_role(
            role,
            head_size,
            attention_config=None,
            role_category=None,
        ):
            captured["role"] = role
            captured["head_size"] = head_size
            captured["role_category"] = role_category
            captured["attention_config"] = attention_config
            return _FakeBackend, AttentionSpec(backend="TORCH_SDPA", extra={"block_size": 128})

        class _FakeRingParallelAttention:
            def __init__(self, sp_group, attn_backend_pref=None):
                self.sp_group = sp_group
                self.attn_backend_pref = attn_backend_pref

        monkeypatch.setattr(layer_mod, "get_attn_backend_for_role", _fake_get_attn_backend_for_role)
        monkeypatch.setattr(layer_mod.SDPABackend, "get_impl_cls", staticmethod(lambda: _FakeAttentionImpl))
        monkeypatch.setattr(layer_mod, "build_parallel_attention_strategy", lambda **kwargs: object())
        monkeypatch.setattr(layer_mod, "get_sp_group", lambda: SimpleNamespace(ring_group="ring-group"))
        monkeypatch.setattr(layer_mod, "RingParallelAttention", _FakeRingParallelAttention)
        monkeypatch.setattr(layer_mod, "is_forward_context_available", lambda: False)
        monkeypatch.setattr(
            layer_mod,
            "get_forward_context",
            lambda: (_ for _ in ()).throw(AssertionError("Attention init should not read ForwardContext")),
        )

        od_config = SimpleNamespace(
            diffusion_attention_config=AttentionConfig(
                default=AttentionSpec(backend="FLASH_ATTN"),
                per_role={"cross": AttentionSpec(backend="TORCH_SDPA", extra={"block_size": 128})},
            ),
            parallel_config=SimpleNamespace(ring_degree=2),
        )

        with set_current_diffusion_config(od_config):
            attn = Attention(
                num_heads=4,
                head_size=64,
                causal=False,
                softmax_scale=1.0,
                role="cross",
                role_category="cross",
                qkv_layout="BSND",
            )

        assert captured["role"] == "cross"
        assert captured["role_category"] == "cross"
        assert captured["head_size"] == 64
        assert captured["attention_config"] is od_config.diffusion_attention_config
        assert attn.backend_pref == "TORCH_SDPA"
        assert attn.attention.kwargs["backend_kwargs"] == {"block_size": 128}
        assert attn.attention.kwargs["qkv_layout"] == "BSND"
        assert attn.use_ring is True
        assert attn.ring_runner is not None
        assert attn.ring_runner.attn_backend_pref == "TORCH_SDPA"
