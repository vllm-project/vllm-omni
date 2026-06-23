# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Model specific tests for CacheDiT enablement.
"""

import sys
from unittest.mock import Mock, patch

import pytest
import torch
from cache_dit.caching.cache_blocks.pattern_0_1_2 import CachedBlocks_Pattern_0_1_2

import vllm_omni.diffusion.cache.cache_dit_backend as cd_backend
from vllm_omni.diffusion.cache.cache_dit_backend import CacheDiTAdapterConfig, CacheDiTBackend
from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer
from vllm_omni.diffusion.models.helios.helios_transformer import HeliosTransformer3DModel
from vllm_omni.diffusion.models.longcat_image.longcat_image_transformer import LongCatImageTransformer2DModel
from vllm_omni.diffusion.models.ltx2.ltx2_transformer import LTX2VideoTransformer3DModel
from vllm_omni.platforms import current_omni_platform

# NOTE: We patch DreamID Omni's modules here with mocks so that we can import and inspect
# the class even though the dependency may not be set up correctly; this is ok for these
# tests because we just inspect it and never initialize the model.
for mod in ("dreamid_omni", "dreamid_omni.modules", "dreamid_omni.modules.model"):
    sys.modules.setdefault(mod, Mock())
# isort: split
from vllm_omni.diffusion.models.dreamid_omni.fusion import FusionModel as DreamIdOmniModel  # noqa: E402

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SEPARATE_CFG_TRANSFORMERS = [
    DreamIdOmniModel,
    LTX2VideoTransformer3DModel,
    HeliosTransformer3DModel,
    LongCatImageTransformer2DModel,
    Cosmos3VFMTransformer,
]

SAMPLE_CACHE_CONFIG = DiffusionCacheConfig()


@pytest.mark.parametrize("transformer_model", SEPARATE_CFG_TRANSFORMERS)
def test_cache_dit_configs_have_separate_cfg(transformer_model):
    """Check models with separate CFG set has_separate_cfg=True in their configs."""
    assert hasattr(transformer_model, "_cache_dit_adapter_config")
    assert isinstance(transformer_model._cache_dit_adapter_config, CacheDiTAdapterConfig)
    assert transformer_model._cache_dit_adapter_config.has_separate_cfg is True


@patch("vllm_omni.diffusion.cache.cache_dit_backend.BlockAdapter")
@patch("vllm_omni.diffusion.cache.cache_dit_backend.cache_dit")
def test_separate_wan22_custom_enabler_has_separate_cfg(mock_cache_dit, mock_block_adapter):
    """Ensure that Wan22, which has a custom enabler, setts custom CFG correctly."""
    mock_pipeline = Mock()
    cd_backend.enable_cache_for_wan22(mock_pipeline, SAMPLE_CACHE_CONFIG)

    mock_cache_dit.enable_cache.assert_called_once()
    adapter_kwargs = mock_block_adapter.call_args.kwargs
    assert adapter_kwargs["has_separate_cfg"] is True


@patch("vllm_omni.diffusion.cache.cache_dit_backend.BlockAdapter")
@patch("vllm_omni.diffusion.cache.cache_dit_backend.cache_dit")
def test_cosmos3_cache_dit_wraps_gen_layers(mock_cache_dit, mock_block_adapter):
    """Cosmos3 should cache only the repeated GEN pathway blocks."""
    mock_pipeline = Mock()
    gen_layers = object()
    mock_pipeline.transformer.gen_layers = gen_layers
    mock_pipeline.transformer._cache_dit_adapter_config = Cosmos3VFMTransformer._cache_dit_adapter_config

    cd_backend.enable_cache_for_cosmos3(mock_pipeline, SAMPLE_CACHE_CONFIG)

    mock_cache_dit.enable_cache.assert_called_once()
    adapter_kwargs = mock_block_adapter.call_args.kwargs
    assert adapter_kwargs["transformer"] is mock_pipeline.transformer
    assert adapter_kwargs["blocks"] == [gen_layers]
    assert adapter_kwargs["has_separate_cfg"] is True
    assert adapter_kwargs["check_forward_pattern"] is False


# This test is skipped on ROCm since rocm_unquantized_gemm doesn't support CPU backend
@pytest.mark.skipif(
    current_omni_platform.is_rocm(),
    reason="vLLM ROCm custom ops lack CPU fallback",
)
def test_ltx2_cache_dit_receives_audio_as_encoder(init_fake_tp_group):
    """CacheDiT Pattern_0 treats the second positional arg as encoder_hidden_states,
    which is a collision for one of the kwargs in LTX2 since we treat the audio
    hidden states as encoder_hidden_states.

    This test ensures that a tiny LTX2 transformer can be initialized and run
    through the cache DiT backend without hitting a collision on the kwargs.
    """
    seq_len = 4
    video_in = torch.full((1, seq_len, 16), 1.0)
    audio_in = torch.full((1, seq_len, 16), 2.0)
    text_in = torch.full((1, seq_len, 16), 3.0)
    audio_text_in = torch.full((1, seq_len, 16), 4.0)

    model = LTX2VideoTransformer3DModel(
        in_channels=16,
        out_channels=16,
        patch_size=1,
        patch_size_t=1,
        num_attention_heads=2,
        attention_head_dim=8,
        cross_attention_dim=16,
        audio_in_channels=16,
        audio_out_channels=16,
        audio_num_attention_heads=2,
        audio_attention_head_dim=8,
        audio_cross_attention_dim=16,
        num_layers=2,
        caption_channels=16,
    )

    # NOTE: This is currently using the LTX2 custom enabler, but the custom
    # enablers will be consolidated after
    # https://github.com/vllm-project/vllm-omni/pull/2527 lands.
    LTX2Pipeline = type("LTX2Pipeline", (), {})
    pipeline = LTX2Pipeline()
    pipeline.transformer = model
    backend = CacheDiTBackend(DiffusionCacheConfig())
    backend.enable(pipeline)
    backend.refresh(pipeline, num_inference_steps=5)

    # Wrap call_Fn_blocks in CacheDiT so that we can verify the
    # hidden/encoder states are what we expect them to be
    captured = {}
    original = CachedBlocks_Pattern_0_1_2.call_Fn_blocks

    def call_Fn_blocks_and_capture(self, hidden_states, encoder_hidden_states, *a, **kw):
        captured["hidden_states"] = hidden_states
        captured["encoder_hidden_states"] = encoder_hidden_states
        return original(self, hidden_states, encoder_hidden_states, *a, **kw)

    # Also, map projections to identity so that we can just check
    # the captured tensors directly instead of having to reproject
    identity = torch.nn.Identity()
    with (
        patch.object(model, "proj_in", identity),
        patch.object(model, "audio_proj_in", identity),
        patch.object(CachedBlocks_Pattern_0_1_2, "call_Fn_blocks", call_Fn_blocks_and_capture),
        torch.no_grad(),
    ):
        model(
            hidden_states=video_in,
            audio_hidden_states=audio_in,
            encoder_hidden_states=text_in,
            audio_encoder_hidden_states=audio_text_in,
            timestep=torch.tensor([[1000.0] * seq_len]),
            num_frames=1,
            height=2,
            width=2,
            audio_num_frames=seq_len,
        )

    # Pattern_0 maps (hidden_states, encoder_hidden_states) to (video, audio)
    assert torch.equal(captured["hidden_states"], video_in)
    assert torch.equal(captured["encoder_hidden_states"], audio_in)
