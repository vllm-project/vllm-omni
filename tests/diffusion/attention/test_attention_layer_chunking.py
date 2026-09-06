# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Layer-level injection of KV quantization + attention chunking options.

The Attention layer parses the diffusion config once (``_init_kv_cache_quantization``)
and injects/removes ``kv_cache_dtype`` and ``attn_chunking`` in
``attn_metadata.extra`` together (``_with_kv_cache_dtype``): backends never
see chunking without a dtype, and every skip gate disables both at once.

These tests build the layer via ``__new__`` (no heavy ``__init__``) and set
exactly the attributes the two methods read.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import vllm_omni.diffusion.attention.layer as layer_mod
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.chunking import AttnChunkingOptions
from vllm_omni.diffusion.attention.layer import Attention

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _bare_layer(
    kv_dtype: str | None = "fp8",
    chunking: AttnChunkingOptions | None = None,
    disable: bool = False,
    skip_steps: set[int] | None = None,
    skip_layers: set[int] | None = None,
) -> Attention:
    layer = Attention.__new__(Attention)
    layer._kv_cache_dtype = kv_dtype
    layer._attn_chunking = chunking
    layer._disable_kv_quant = disable
    layer._kv_cache_skip_steps = skip_steps
    layer._kv_cache_skip_layers = skip_layers
    layer.layer_idx = None
    return layer


class TestWithKvCacheDtypeInjection:
    def test_active_quant_injects_dtype_and_chunking(self) -> None:
        options = AttnChunkingOptions(q_chunk=8, head_chunk=2)
        layer = _bare_layer(chunking=options)

        metadata = layer._with_kv_cache_dtype(AttentionMetadata(extra={"other": 1}))

        assert metadata.extra["kv_cache_dtype"] == "fp8"
        assert metadata.extra["attn_chunking"] is options
        assert metadata.extra["other"] == 1  # untouched

    def test_active_quant_without_chunking_only_dtype(self) -> None:
        layer = _bare_layer(chunking=None)

        metadata = layer._with_kv_cache_dtype(AttentionMetadata(extra={}))

        assert metadata.extra["kv_cache_dtype"] == "fp8"
        assert "attn_chunking" not in metadata.extra

    def test_none_metadata_gets_both_keys(self) -> None:
        options = AttnChunkingOptions(q_chunk=8)
        layer = _bare_layer(chunking=options)

        metadata = layer._with_kv_cache_dtype(None)

        assert metadata is not None
        assert metadata.extra == {"kv_cache_dtype": "fp8", "attn_chunking": options}

    def test_quant_disabled_pops_both_keys(self) -> None:
        options = AttnChunkingOptions(q_chunk=8)
        layer = _bare_layer(kv_dtype=None)
        # A stale chunking key (e.g. shared metadata object) must not survive.
        metadata = AttentionMetadata(extra={"kv_cache_dtype": "fp8", "attn_chunking": options})

        cleaned = layer._with_kv_cache_dtype(metadata)

        assert "kv_cache_dtype" not in cleaned.extra
        assert "attn_chunking" not in cleaned.extra

    def test_layer_opt_out_pops_both_keys(self) -> None:
        layer = _bare_layer(disable=True)
        metadata = AttentionMetadata(extra={"kv_cache_dtype": "fp8", "attn_chunking": AttnChunkingOptions(q_chunk=8)})

        cleaned = layer._with_kv_cache_dtype(metadata)

        assert "kv_cache_dtype" not in cleaned.extra
        assert "attn_chunking" not in cleaned.extra

    def test_skip_step_hit_pops_both_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(layer_mod, "is_forward_context_available", lambda: True)
        monkeypatch.setattr(layer_mod, "get_forward_context", lambda: SimpleNamespace(denoise_step_idx=3))
        layer = _bare_layer(skip_steps={3})

        cleaned = layer._with_kv_cache_dtype(AttentionMetadata(extra={"kv_cache_dtype": "fp8"}))

        assert "kv_cache_dtype" not in cleaned.extra


class TestInitKvCacheQuantizationParsing:
    @staticmethod
    def _config(**overrides) -> SimpleNamespace:
        config = SimpleNamespace(
            diffusion_kv_cache_dtype="fp8",
            parallel_config=SimpleNamespace(ring_degree=1),
            diffusion_kv_cache_skip_step_indices=None,
            diffusion_kv_cache_skip_layer_indices=None,
            diffusion_attn_q_chunk=1,
            diffusion_attn_head_chunk=0,
            diffusion_attn_head_chunk_min_kv=50000,
        )
        config.__dict__.update(overrides)
        return config

    @pytest.fixture
    def npu_platform(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(layer_mod, "current_omni_platform", SimpleNamespace(device_name="npu"))

    def _bare_init_layer(self) -> Attention:
        layer = Attention.__new__(Attention)
        layer.attention = SimpleNamespace(supports_kv_cache_dtype=lambda dtype, platform: True)
        layer.attn_backend = SimpleNamespace(get_name=lambda: "FLASH_ATTN")
        layer._kv_cache_dtype = None
        layer._attn_chunking = None
        layer._kv_cache_skip_steps = None
        layer._kv_cache_skip_layers = None
        return layer

    def test_parses_chunk_fields_into_options(self, npu_platform) -> None:
        layer = self._bare_init_layer()
        config = self._config(diffusion_attn_q_chunk=8, diffusion_attn_head_chunk=2)

        layer._init_kv_cache_quantization(config)

        assert layer._attn_chunking == AttnChunkingOptions(q_chunk=8, head_chunk=2, head_chunk_min_kv=50000)

    def test_inert_defaults_stay_none(self, npu_platform) -> None:
        layer = self._bare_init_layer()

        layer._init_kv_cache_quantization(self._config())

        assert layer._kv_cache_dtype == "fp8"
        assert layer._attn_chunking is None

    def test_no_dtype_means_no_chunking(self, npu_platform) -> None:
        # Config validation rejects chunk flags without fp8 upstream; the
        # layer stays defensive and drops chunking with the dtype.
        layer = self._bare_init_layer()
        config = self._config(diffusion_kv_cache_dtype=None, diffusion_attn_q_chunk=8)

        layer._init_kv_cache_quantization(config)

        assert layer._kv_cache_dtype is None
        assert layer._attn_chunking is None
