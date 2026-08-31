# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for per-role attention backend configuration (RFC: per-role-attention-backend).

Tests cover:
- AttentionSpec and AttentionConfig normalization
- Role-aware backend resolution with category fallback
- OmniDiffusionConfig attention shorthand handling
- AttentionMetadata.extra field
"""

from types import SimpleNamespace
from unittest.mock import Mock

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

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class TestAttentionSpec:
    def test_construct_no_skip_softmax(self):
        spec = AttentionSpec(backend="FLASH_ATTN")
        assert spec.skip_softmax is None

    def test_skip_softmax_mapping_coerced(self):
        spec = AttentionSpec(backend="TRTLLM_ATTN", skip_softmax={"target_sparsity": 0.5})
        assert spec.backend == "TRTLLM_ATTN"
        assert spec.skip_softmax.target_sparsity == 0.5

    def test_invalid_backend_type(self):
        with pytest.raises(TypeError):
            AttentionSpec(backend=123)  # type: ignore[arg-type]

    def test_skip_softmax_rejected_on_non_trtllm(self):
        with pytest.raises(ValueError, match="only supported by the TRTLLM_ATTN"):
            AttentionSpec(backend="TORCH_SDPA", skip_softmax={"target_sparsity": 0.5})

    def test_quant_serialized_with_defaults_and_overrides(self):
        assert AttentionSpec(backend="TRTLLM_ATTN", quant={"dtype_qk": "fp8_e4m3"}).backend_kwargs()["quant"] == {
            "dtype_qk": "fp8_e4m3",
            "q_block_size": 1,
            "k_block_size": 16,
        }
        assert AttentionSpec(
            backend="TRTLLM_ATTN", quant={"dtype_qk": "int8", "q_block_size": 4, "k_block_size": 16}
        ).backend_kwargs()["quant"] == {"dtype_qk": "int8", "q_block_size": 4, "k_block_size": 16}

    def test_quant_superset_fields_passed_through(self):
        # FLASHINFER_ATTN-style config (dtype_vo / flashinfer_backend) round-trips on the shared spec.
        bk = AttentionSpec(
            backend="FLASHINFER_ATTN",
            quant={"dtype_qk": "bfloat16", "dtype_vo": "fp8_e4m3", "flashinfer_backend": "trtllm-gen"},
        ).backend_kwargs()["quant"]
        assert bk["dtype_qk"] == "bfloat16"
        assert bk["dtype_vo"] == "fp8_e4m3"
        assert bk["flashinfer_backend"] == "trtllm-gen"

    def test_quant_and_skip_softmax_coexist(self):
        bk = AttentionSpec(
            backend="TRTLLM_ATTN", quant={"dtype_qk": "int8"}, skip_softmax={"target_sparsity": 0.5}
        ).backend_kwargs()
        assert bk["target_sparsity"] == 0.5 and bk["quant"]["dtype_qk"] == "int8"

    @pytest.mark.parametrize(
        "spec, match",
        [
            ({"backend": "TORCH_SDPA", "quant": {"dtype_qk": "int8"}}, "only supported by the TRTLLM_ATTN"),
            ({"backend": "TRTLLM_ATTN", "quant": {"dtype_qk": "int4"}}, "quant.dtype_qk"),
            ({"backend": "TRTLLM_ATTN", "quant": {"dtype_qk": "int8", "k_block_size": 8}}, "quant.k_block_size"),
        ],
    )
    def test_quant_validation_rejects(self, spec, match):
        with pytest.raises(ValueError, match=match):
            AttentionSpec(**spec)

    def test_fastvideo_vsa_topk_serialized(self):
        spec = AttentionSpec(backend="FASTVIDEO_VSA", fastvideo_vsa_topk=96)
        assert spec.backend_kwargs() == {"topk": 96}

    def test_fastvideo_vsa_topk_rejected_for_other_backend(self):
        with pytest.raises(ValueError, match="only supported by the FASTVIDEO_VSA"):
            AttentionSpec(backend="TORCH_SDPA", fastvideo_vsa_topk=96)

    def test_block_sparse_defaults_applied_when_backend_selected(self):
        spec = AttentionSpec(backend="RAINFUSION_ATTN")
        assert spec.block_sparse.sparsity == 0.8
        assert spec.backend_kwargs() == {
            "sparsity": 0.8,
            "start_step": 0,
            "end_step": 0,
            "precision": "bf16",
        }

    def test_block_sparse_skip_layers_selector_expanded(self):
        spec = AttentionSpec(
            backend="RAINFUSION_ATTN",
            block_sparse={"sparsity": 0.9, "start_step": 12, "skip_layers": "0-2,38"},
        )
        assert spec.block_sparse.skip_layer_indices == {0, 1, 2, 38}
        assert spec.backend_kwargs() == {
            "sparsity": 0.9,
            "start_step": 12,
            "end_step": 0,
            "precision": "bf16",
            "skip_layers": [0, 1, 2, 38],
        }

    def test_block_sparse_rejected_on_dense_backend(self):
        with pytest.raises(ValueError, match="block_sparse is only supported by"):
            AttentionSpec(backend="FLASH_ATTN", block_sparse={"sparsity": 0.8})

    @pytest.mark.parametrize(
        "block_sparse",
        [{"sparsity": 1.5}, {"start_step": -1}, {"end_step": -1}, {"precision": "bogus"}],
    )
    def test_block_sparse_invalid_values(self, block_sparse):
        with pytest.raises(ValueError):
            AttentionSpec(backend="RAINFUSION_ATTN", block_sparse=block_sparse)

    def test_block_sparse_end_step_precision_forwarded(self):
        spec = AttentionSpec(
            backend="RAINFUSION_ATTN",
            block_sparse={"sparsity": 0.9, "start_step": 5, "end_step": 3, "precision": "fp8"},
        )
        assert spec.backend_kwargs() == {
            "sparsity": 0.9,
            "start_step": 5,
            "end_step": 3,
            "precision": "fp8",
        }

    def test_block_sparse_precision_enum_input_normalized(self):
        """RainFusionPrecision enum members must normalize to their string value.

        Regression: the str,Enum members passed the value check but str() later
        produced "RainFusionPrecision.MIX" instead of "mix" (StrEnum fixes it).
        """
        from vllm_omni.diffusion.attention.backends.rainfusion_attn import RainFusionConfig
        from vllm_omni.diffusion.data import RainFusionPrecision

        spec = AttentionSpec(
            backend="RAINFUSION_ATTN",
            block_sparse={"sparsity": 0.8, "precision": RainFusionPrecision.MIX},
        )
        kwargs = spec.backend_kwargs()
        assert kwargs["precision"] == "mix"
        # The config path is where str(value) used to mangle enum members.
        cfg = RainFusionConfig.from_backend_kwargs(kwargs)
        assert cfg.precision == "mix"
        assert type(cfg.precision) is str


class TestAttentionConfig:
    def test_empty_config(self):
        config = AttentionConfig()
        assert config.default is None
        assert config.per_role == {}

    def test_constructor_normalizes_mappings(self):
        config = AttentionConfig(
            default={"backend": "FLASH_ATTN"},
            per_role={
                "self": {"backend": "TRTLLM_ATTN", "skip_softmax": {"target_sparsity": 0.5}},
                "cross": "SAGE_ATTN",
            },
        )
        assert config.default.backend == "FLASH_ATTN"
        assert config.per_role["self"].backend == "TRTLLM_ATTN"
        assert config.per_role["self"].skip_softmax.target_sparsity == 0.5
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
                "self": AttentionSpec(backend="SPARSE_BLOCK"),
                "cross": AttentionSpec(backend="SAGE_ATTN"),
                "ltx2.audio_self": AttentionSpec(backend="FLASH_ATTN"),
                "ltx2.audio_to_video": AttentionSpec(backend="FLASH_ATTN"),
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

    @pytest.fixture(autouse=True)
    def _clear_diffusion_attention_backend_env(self, monkeypatch):
        # OmniDiffusionConfig.__post_init__ applies DIFFUSION_ATTENTION_BACKEND via
        # build_attention_config(); clear it so these tests assert config defaults,
        # not whatever the process inherited from CI / sibling tests.
        monkeypatch.delenv("DIFFUSION_ATTENTION_BACKEND", raising=False)

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
    @pytest.mark.parametrize("mindiesd_available", [False, True])
    def test_paged_npu_memory_profile_uses_sdpa_only_without_mindiesd(self, monkeypatch, mindiesd_available):
        attention = Attention.__new__(Attention)
        attention._scheduler_paged_kv = True
        attention.paged_kv_cache_role = "primary"
        attention.backend_pref = "FLASH_ATTN"
        attention.attn_backend = SimpleNamespace(supports_piecewise_spans=True, get_name=lambda: "FLASH_ATTN")
        flash_output = torch.ones(1)
        sdpa_output = torch.zeros(1)
        dense_flash = Mock(return_value=flash_output)
        sdpa = Mock(return_value=sdpa_output)
        attention.attention = SimpleNamespace(forward=dense_flash)
        attention.sdpa_fallback = SimpleNamespace(forward=sdpa)
        monkeypatch.setattr(layer_mod, "is_forward_context_available", lambda: True)
        monkeypatch.setattr(
            layer_mod,
            "get_forward_context",
            lambda: SimpleNamespace(in_diffusion_kv_memory_profile=True),
        )
        monkeypatch.setattr(layer_mod.current_omni_platform, "is_npu", lambda: True)
        monkeypatch.setattr(
            layer_mod.current_omni_platform,
            "supports_diffusion_dense_flash_attention",
            lambda: mindiesd_available,
        )

        result = attention._run_local_attention(
            torch.zeros(1, dtype=torch.bfloat16),
            torch.zeros(1, dtype=torch.bfloat16),
            torch.zeros(1, dtype=torch.bfloat16),
            AttentionMetadata(),
        )

        if mindiesd_available:
            assert result is flash_output
            dense_flash.assert_called_once()
            sdpa.assert_not_called()
        else:
            assert result is sdpa_output
            dense_flash.assert_not_called()
            sdpa.assert_called_once()

    def test_dense_flash_import_error_is_not_reclassified(self, monkeypatch):
        attention = Attention.__new__(Attention)
        attention._scheduler_paged_kv = False
        attention.paged_kv_cache_role = "primary"
        attention.backend_pref = "FLASH_ATTN"
        attention.attn_backend = SimpleNamespace(supports_piecewise_spans=True, get_name=lambda: "FLASH_ATTN")
        attention.attention = SimpleNamespace(
            forward=Mock(side_effect=ModuleNotFoundError("No module named 'mindiesd'", name="mindiesd"))
        )
        attention.sdpa_fallback = SimpleNamespace(forward=Mock(side_effect=AssertionError("must not fall back")))
        monkeypatch.setattr(layer_mod, "is_forward_context_available", lambda: False)

        with pytest.raises(ModuleNotFoundError, match="mindiesd"):
            attention._run_local_attention(
                torch.zeros(1, dtype=torch.bfloat16),
                torch.zeros(1, dtype=torch.bfloat16),
                torch.zeros(1, dtype=torch.bfloat16),
                AttentionMetadata(),
            )

    def test_paged_marker_does_not_change_dense_backend(self, monkeypatch):
        class _FakeAttentionImpl:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def forward(self, query, key, value, attn_metadata=None):
                return query

        class _DenseBackend:
            supports_paged_kv = False

            @staticmethod
            def get_name() -> str:
                return "TORCH_SDPA"

            @staticmethod
            def get_impl_cls():
                return _FakeAttentionImpl

        calls = []

        def _fake_get_attn_backend_for_role(*args, **kwargs):
            calls.append((args, kwargs))
            return _DenseBackend, None

        monkeypatch.setattr(layer_mod, "get_attn_backend_for_role", _fake_get_attn_backend_for_role)
        monkeypatch.setattr(layer_mod.SDPABackend, "get_impl_cls", staticmethod(lambda: _FakeAttentionImpl))
        monkeypatch.setattr(layer_mod, "build_parallel_attention_strategy", lambda **kwargs: object())
        monkeypatch.setattr(layer_mod, "is_forward_context_available", lambda: False)

        od_config = SimpleNamespace(
            diffusion_attention_config=AttentionConfig(),
            diffusion_kv_mode=layer_mod.DiffusionKVCacheMode.DENSE_LEGACY,
            parallel_config=SimpleNamespace(ring_degree=1),
            diffusion_kv_cache_dtype=None,
            diffusion_kv_cache_skip_step_indices=None,
            diffusion_kv_cache_skip_layer_indices=None,
        )

        with set_current_diffusion_config(od_config):
            attention = Attention(
                num_heads=4,
                head_size=64,
                causal=False,
                softmax_scale=1.0,
                paged_kv_cache_role="primary",
            )

        assert len(calls) == 1
        assert attention.attn_backend is _DenseBackend
        assert attention.backend_pref is None

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
            allow_trtllm_default=False,
        ):
            captured["role"] = role
            captured["head_size"] = head_size
            captured["role_category"] = role_category
            captured["attention_config"] = attention_config
            return _FakeBackend, AttentionSpec(backend="TRTLLM_ATTN", skip_softmax={"target_sparsity": 0.5})

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
                per_role={"cross": AttentionSpec(backend="TRTLLM_ATTN", skip_softmax={"target_sparsity": 0.5})},
            ),
            parallel_config=SimpleNamespace(ring_degree=2),
            diffusion_kv_cache_dtype=None,
            diffusion_kv_cache_skip_step_indices=None,
            diffusion_kv_cache_skip_layer_indices=None,
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
        assert attn.backend_pref == "TRTLLM_ATTN"
        assert attn.attention.kwargs["backend_kwargs"] == {"target_sparsity": 0.5}
        assert attn.attention.kwargs["qkv_layout"] == "BSND"
        assert attn.use_ring is True
        assert attn.ring_runner is not None
        assert attn.ring_runner.attn_backend_pref == "TRTLLM_ATTN"


class TestDiffusionKvCacheQuantization:
    @staticmethod
    def _install_attention_init_stubs(monkeypatch):
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

            @staticmethod
            def supports_kv_cache_dtype(kv_cache_dtype, platform_key) -> bool:
                return True

        class _FakeRingParallelAttention:
            def __init__(self, sp_group, attn_backend_pref=None):
                self.sp_group = sp_group
                self.attn_backend_pref = attn_backend_pref

        monkeypatch.setattr(
            layer_mod,
            "get_attn_backend_for_role",
            lambda role, head_size, attention_config=None, role_category=None, allow_trtllm_default=False: (
                _FakeBackend,
                None,
            ),
        )
        monkeypatch.setattr(layer_mod.SDPABackend, "get_impl_cls", staticmethod(lambda: _FakeAttentionImpl))
        monkeypatch.setattr(layer_mod, "build_parallel_attention_strategy", lambda **kwargs: object())
        monkeypatch.setattr(layer_mod, "get_sp_group", lambda: SimpleNamespace(ring_group="ring-group"))
        monkeypatch.setattr(layer_mod, "RingParallelAttention", _FakeRingParallelAttention)
        monkeypatch.setattr(layer_mod, "is_forward_context_available", lambda: False)

    def test_diffusion_kv_cache_dtype_none_does_not_trigger_ring_quantization_error(self, monkeypatch):
        self._install_attention_init_stubs(monkeypatch)
        od_config = SimpleNamespace(
            diffusion_attention_config=AttentionConfig(),
            parallel_config=SimpleNamespace(ring_degree=2),
            diffusion_kv_cache_dtype=None,
            diffusion_kv_cache_skip_step_indices=None,
            diffusion_kv_cache_skip_layer_indices=None,
        )

        with set_current_diffusion_config(od_config):
            attn = Attention(
                num_heads=4,
                head_size=64,
                causal=False,
                softmax_scale=1.0,
            )

        assert attn._kv_cache_dtype is None

    def test_diffusion_kv_cache_dtype_auto_does_not_trigger_ring_quantization_error(self, monkeypatch):
        self._install_attention_init_stubs(monkeypatch)
        od_config = SimpleNamespace(
            diffusion_attention_config=AttentionConfig(),
            parallel_config=SimpleNamespace(ring_degree=2),
            diffusion_kv_cache_dtype="auto",
            diffusion_kv_cache_skip_step_indices=None,
            diffusion_kv_cache_skip_layer_indices=None,
        )

        with set_current_diffusion_config(od_config):
            attn = Attention(
                num_heads=4,
                head_size=64,
                causal=False,
                softmax_scale=1.0,
            )

        assert attn._kv_cache_dtype is None

    def test_diffusion_kv_cache_dtype_fp8_raises_with_ring_attention(self, monkeypatch):
        self._install_attention_init_stubs(monkeypatch)
        od_config = SimpleNamespace(
            diffusion_attention_config=AttentionConfig(),
            parallel_config=SimpleNamespace(ring_degree=2),
            diffusion_kv_cache_dtype="fp8",
            diffusion_kv_cache_skip_step_indices=None,
            diffusion_kv_cache_skip_layer_indices=None,
        )

        with set_current_diffusion_config(od_config):
            with pytest.raises(ValueError, match="KV quantization is not compatible with ring attention"):
                Attention(
                    num_heads=4,
                    head_size=64,
                    causal=False,
                    softmax_scale=1.0,
                )
