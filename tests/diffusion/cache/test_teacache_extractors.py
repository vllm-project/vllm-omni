# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for TeaCache extractor functions.

This module provides a generic testing framework for model-specific extractor functions
used by TeaCache. Each model's extractor can be tested by:
1. Creating a fixture that returns model module
2. Creating a fixture that returns sample inputs for that model
3. Creating a test class that inherits from BaseExtractorTest
4. Implementing any model-specific test methods

Currently implemented:
- TestFlux2KleinExtractor: Flux2Klein model extractor
- TestFlux2Extractor: Flux2 model extractor
- TestFluxExtractor: Flux model extractor
- TestMiniMaxH3Extractor: MiniMaxH3DiTModel extractor
"""

from abc import ABC, abstractmethod
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.cache.teacache.extractors import (
    extract_flux2_context,
    extract_flux2_klein_context,
    extract_flux_context,
    extract_minimax_h3_context,
)
from vllm_omni.diffusion.models.flux.flux_transformer import FluxTransformer2DModel
from vllm_omni.diffusion.models.flux2_klein.flux2_klein_transformer import (
    Flux2Transformer2DModel,
)

pytestmark = [pytest.mark.core_model]


@pytest.fixture(scope="function", autouse=True)
def setup_tp_group():
    """Set up TP group for each test function"""
    with patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", return_value=1):
        with patch("vllm.distributed.parallel_state.get_tp_group") as mock_get_tp_group:
            mock_tp_group = MagicMock()
            mock_tp_group.world_size = 1
            mock_get_tp_group.return_value = mock_tp_group
            yield


class BaseExtractorTest(ABC):
    """Base class for testing TeaCache extractors.

    Subclasses should implement:
    - get_extractor(): Return extractor function
    - get_module(): Return model module
    - get_sample_inputs(): Return sample inputs for model
    """

    @abstractmethod
    def get_extractor(self):
        """Return extractor function to test."""
        pass

    @abstractmethod
    def get_module(self):
        """Return model module instance."""
        pass

    @abstractmethod
    def get_sample_inputs(self):
        """Return sample inputs for model."""
        pass


class TestFlux2KleinExtractor(BaseExtractorTest):
    """Test extract_flux2_klein_context function."""

    def get_extractor(self):
        return extract_flux2_klein_context

    @pytest.fixture
    def flux2_klein_module(self):
        """Create a minimal Flux2Transformer2DModel for testing."""
        model = Flux2Transformer2DModel(
            num_layers=2,
            num_single_layers=2,
            num_attention_heads=48,
            attention_head_dim=128,
            joint_attention_dim=15360,
        )
        return model

    def get_module(self, flux2_klein_module):
        return flux2_klein_module

    @pytest.fixture
    def sample_inputs(self):
        """Create sample input tensors for Flux2Klein.

        Note: hidden_states uses in_channels=128 (default for Flux2Klein),
        not inner_dim=6144. The x_embedder projects from 128 -> 6144.
        encoder_hidden_states uses joint_attention_dim=15360 (model default),
        which then gets projected to inner_dim=6144 by context_embedder.
        """
        batch_size = 1
        img_seq_len = 1024
        txt_seq_len = 512
        in_channels = 128  # Model default in_channels
        txt_dim = 15360  # Model default joint_attention_dim

        return {
            "hidden_states": torch.randn(batch_size, img_seq_len, in_channels),
            "encoder_hidden_states": torch.randn(batch_size, txt_seq_len, txt_dim),
            "timestep": torch.tensor([500]),
            "img_ids": torch.randint(0, 64, (batch_size, img_seq_len, 4)),
            "txt_ids": torch.randint(0, 64, (batch_size, txt_seq_len, 4)),
            "guidance": torch.tensor([3.5]),
        }

    def get_sample_inputs(self, sample_inputs):
        return sample_inputs

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_modulated_input_shape(self, flux2_klein_module, sample_inputs):
        """Test that modulated_input has correct shape matching the model's inner_dim.

        Note: After x_embedder projection, hidden_states are projected from
        in_channels (128) to inner_dim (6144), so modulated_input should match
        the projected shape, not the input shape.
        """
        context = extract_flux2_klein_context(flux2_klein_module, **sample_inputs)

        batch_size, img_seq_len, _ = sample_inputs["hidden_states"].shape
        inner_dim = flux2_klein_module.inner_dim
        assert context.modulated_input.shape == (batch_size, img_seq_len, inner_dim)

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_run_transformer_blocks_callable(self, flux2_klein_module, sample_inputs):
        """Test that run_transformer_blocks is callable."""
        context = extract_flux2_klein_context(flux2_klein_module, **sample_inputs)
        assert callable(context.run_transformer_blocks)

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_postprocess_callable(self, flux2_klein_module, sample_inputs):
        """Test that postprocess is callable."""
        context = extract_flux2_klein_context(flux2_klein_module, **sample_inputs)
        assert callable(context.postprocess)

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_extra_states_contains_full_transformer(self, flux2_klein_module, sample_inputs):
        """Test that extra_states contains run_flux2_full_transformer_with_single."""
        context = extract_flux2_klein_context(flux2_klein_module, **sample_inputs)

        assert context.extra_states is not None
        assert "run_flux2_full_transformer_with_single" in context.extra_states
        assert callable(context.extra_states["run_flux2_full_transformer_with_single"])

    def test_without_guidance(self, flux2_klein_module, sample_inputs):
        """Test context extraction works without guidance (no CFG)."""
        inputs = sample_inputs.copy()
        inputs["guidance"] = None

        context = extract_flux2_klein_context(flux2_klein_module, **inputs)

        assert context is not None
        assert context.temb is not None

    @pytest.mark.cpu
    def test_invalid_module_raises_error(self):
        """Test that invalid module without transformer_blocks raises ValueError."""
        invalid_module = Mock()
        invalid_module.transformer_blocks = []

        with pytest.raises(ValueError, match="Module must have transformer_blocks"):
            extract_flux2_klein_context(
                invalid_module,
                hidden_states=torch.randn(1, 1024, 6144),
                encoder_hidden_states=torch.randn(1, 512, 15360),
                timestep=torch.tensor([500]),
                img_ids=torch.randint(0, 64, (1, 1024, 4)),
                txt_ids=torch.randint(0, 64, (1, 512, 4)),
            )


class TestFlux2Extractor(BaseExtractorTest):
    """Test extract_flux2_context function."""

    def get_extractor(self):
        return extract_flux2_context

    @pytest.fixture
    def flux2_module(self):
        """Create a minimal Flux2Transformer2DModel for testing."""
        from vllm_omni.diffusion.models.flux2.flux2_transformer import Flux2Transformer2DModel

        model = Flux2Transformer2DModel(
            num_layers=2,
            num_single_layers=2,
            num_attention_heads=48,
            attention_head_dim=128,
            joint_attention_dim=15360,
        )
        return model

    def get_module(self, flux2_module):
        return flux2_module

    @pytest.fixture
    def sample_inputs(self):
        """Create sample input tensors for Flux2.

        Note: hidden_states uses in_channels=128 (default for Flux2),
        not inner_dim=6144. The x_embedder projects from 128 -> 6144.
        encoder_hidden_states uses joint_attention_dim=15360 (model default),
        which then gets projected to inner_dim=6144 by context_embedder.
        """
        batch_size = 1
        img_seq_len = 1024
        txt_seq_len = 512
        in_channels = 128  # Model default in_channels
        txt_dim = 15360  # Model default joint_attention_dim

        return {
            "hidden_states": torch.randn(batch_size, img_seq_len, in_channels),
            "encoder_hidden_states": torch.randn(batch_size, txt_seq_len, txt_dim),
            "timestep": torch.tensor([500]),
            "img_ids": torch.randint(0, 64, (batch_size, img_seq_len, 4)),
            "txt_ids": torch.randint(0, 64, (batch_size, txt_seq_len, 4)),
            "guidance": torch.tensor([3.5]),
        }

    def get_sample_inputs(self, sample_inputs):
        return sample_inputs

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_modulated_input_shape(self, flux2_module, sample_inputs):
        """Test that modulated_input has correct shape matching the model's inner_dim.

        Note: After x_embedder projection, hidden_states are projected from
        in_channels (128) to inner_dim (6144), so modulated_input should match
        the projected shape, not the input shape.
        """
        context = extract_flux2_context(flux2_module, **sample_inputs)

        batch_size, img_seq_len, _ = sample_inputs["hidden_states"].shape
        inner_dim = flux2_module.inner_dim
        assert context.modulated_input.shape == (batch_size, img_seq_len, inner_dim)

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_run_transformer_blocks_callable(self, flux2_module, sample_inputs):
        """Test that run_transformer_blocks is callable."""
        context = extract_flux2_context(flux2_module, **sample_inputs)
        assert callable(context.run_transformer_blocks)

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_postprocess_callable(self, flux2_module, sample_inputs):
        """Test that postprocess is callable."""
        context = extract_flux2_context(flux2_module, **sample_inputs)
        assert callable(context.postprocess)

    def test_without_guidance(self, flux2_module, sample_inputs):
        """Test context extraction works without guidance (no CFG)."""
        inputs = sample_inputs.copy()
        inputs["guidance"] = None

        context = extract_flux2_context(flux2_module, **inputs)

        assert context is not None
        assert context.temb is not None

    @pytest.mark.cpu
    def test_invalid_module_raises_error(self):
        """Test that invalid module without transformer_blocks raises ValueError."""
        invalid_module = Mock()
        invalid_module.transformer_blocks = []

        with pytest.raises(ValueError, match="Module must have transformer_blocks"):
            extract_flux2_context(
                invalid_module,
                hidden_states=torch.randn(1, 1024, 6144),
                encoder_hidden_states=torch.randn(1, 512, 15360),
                timestep=torch.tensor([500]),
                img_ids=torch.randint(0, 64, (1, 1024, 4)),
                txt_ids=torch.randint(0, 64, (1, 512, 4)),
            )


@pytest.mark.cpu
class TestFluxExtractor(BaseExtractorTest):
    """Test extract_flux_context function."""

    @pytest.fixture(autouse=True)
    def cpu_vllm_config(self):
        """Force CPU custom-op dispatch for this test class."""
        from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

        with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
            yield

    @pytest.fixture(autouse=True)
    def mock_flux_attention_backend(self):
        """Use the SDPA backend so FLUX can be instantiated in CPU tests."""
        from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend

        with patch(
            "vllm_omni.diffusion.attention.layer.get_attn_backend_for_role",
            return_value=(SDPABackend, None),
        ):
            yield

    def get_extractor(self):
        return extract_flux_context

    @pytest.fixture
    def flux_module(self):
        """Create a minimal FluxTransformer2DModel for testing."""
        return FluxTransformer2DModel(
            num_layers=2,
            num_single_layers=2,
            num_attention_heads=2,
            attention_head_dim=16,
            joint_attention_dim=32,
            pooled_projection_dim=16,
            axes_dims_rope=(4, 4, 8),
        )

    @pytest.fixture
    def flux_module_without_guidance(self):
        """Create a minimal non-guidance-distilled FLUX transformer."""
        return FluxTransformer2DModel(
            num_layers=2,
            num_single_layers=2,
            num_attention_heads=2,
            attention_head_dim=16,
            joint_attention_dim=32,
            pooled_projection_dim=16,
            guidance_embeds=False,
            axes_dims_rope=(4, 4, 8),
        )

    def get_module(self, flux_module):
        return flux_module

    @pytest.fixture
    def sample_inputs(self):
        """Create sample input tensors for Flux."""
        batch_size = 1
        img_seq_len = 16
        txt_seq_len = 8
        in_channels = 64  # Flux default in_channels
        txt_dim = 32
        pooled_dim = 16

        return {
            "hidden_states": torch.randn(batch_size, img_seq_len, in_channels),
            "encoder_hidden_states": torch.randn(batch_size, txt_seq_len, txt_dim),
            "pooled_projections": torch.randn(batch_size, pooled_dim),
            "timestep": torch.tensor([500]),
            "img_ids": torch.randint(0, 64, (batch_size, img_seq_len, 3)),
            "txt_ids": torch.randint(0, 64, (batch_size, txt_seq_len, 3)),
            "guidance": torch.tensor([3.5]),
        }

    def get_sample_inputs(self, sample_inputs):
        return sample_inputs

    def test_modulated_input_shape(self, flux_module, sample_inputs):
        """Test that modulated_input has the projected FLUX inner dimension."""
        context = extract_flux_context(flux_module, **sample_inputs)

        batch_size, img_seq_len, _ = sample_inputs["hidden_states"].shape
        assert context.modulated_input.shape == (batch_size, img_seq_len, flux_module.inner_dim)

    def test_run_transformer_blocks_callable(self, flux_module, sample_inputs):
        """Test that run_transformer_blocks is callable."""
        context = extract_flux_context(flux_module, **sample_inputs)
        assert callable(context.run_transformer_blocks)

    def test_postprocess_callable(self, flux_module, sample_inputs):
        """Test that postprocess is callable."""
        context = extract_flux_context(flux_module, **sample_inputs)
        assert callable(context.postprocess)

    def test_postprocess_output_shape(self, flux_module, sample_inputs):
        """Test that postprocess projects back to the input channel width."""
        context = extract_flux_context(flux_module, **sample_inputs)
        output = context.postprocess(context.hidden_states)

        assert output.sample.shape == sample_inputs["hidden_states"].shape

    def test_postprocess_return_tuple_when_return_dict_false(self, flux_module, sample_inputs):
        """Test that postprocess honors return_dict=False."""
        context = extract_flux_context(flux_module, **sample_inputs, return_dict=False)
        output = context.postprocess(context.hidden_states)

        assert isinstance(output, tuple)
        assert len(output) == 1
        assert output[0].shape == sample_inputs["hidden_states"].shape

    def test_without_guidance(self, flux_module_without_guidance, sample_inputs):
        """Test context extraction works for FLUX variants without guidance embeddings."""
        inputs = sample_inputs.copy()
        inputs["guidance"] = None

        context = extract_flux_context(flux_module_without_guidance, **inputs)

        assert context is not None
        assert context.temb is not None

    def test_invalid_module_raises_error(self):
        """Test that invalid module without transformer_blocks raises ValueError."""
        invalid_module = Mock()
        invalid_module.transformer_blocks = []

        with pytest.raises(ValueError, match="Module must have transformer_blocks"):
            extract_flux_context(
                invalid_module,
                hidden_states=torch.randn(1, 16, 64),
                encoder_hidden_states=torch.randn(1, 8, 32),
                pooled_projections=torch.randn(1, 16),
                timestep=torch.tensor([500]),
                img_ids=torch.randint(0, 64, (1, 16, 3)),
                txt_ids=torch.randint(0, 64, (1, 8, 3)),
            )


class _MiniMaxH3FakeLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        *,
        bias=False,
        params_dtype=None,
        quant_config=None,
        prefix="",
        total_num_heads=None,
        total_num_kv_heads=None,
        **kwargs,
    ):
        del kwargs
        super().__init__()
        self.prefix = prefix
        self.quant_config = quant_config
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=params_dtype or torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=params_dtype or torch.float32))
        else:
            self.register_parameter("bias", None)
        self.num_heads = total_num_heads
        self.num_kv_heads = total_num_kv_heads

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias), None


class _MiniMaxH3FakeMergedLinear(_MiniMaxH3FakeLinear):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(in_features, sum(out_features), **kwargs)


class _MiniMaxH3FakeQKVLinear(_MiniMaxH3FakeLinear):
    def __init__(
        self,
        *,
        hidden_size,
        head_size,
        total_num_heads,
        total_num_kv_heads,
        **kwargs,
    ):
        qkv_features = (total_num_heads + 2 * total_num_kv_heads) * head_size
        super().__init__(
            hidden_size,
            qkv_features,
            total_num_heads=total_num_heads,
            total_num_kv_heads=total_num_kv_heads,
            **kwargs,
        )


class _MiniMaxH3FakeAttention(nn.Module):
    class FakeBackend:
        supports_prefix_kv_slicing = False

        # Mirror AttentionBackend.supports_packed_mask_free: H3's packed
        # attention path calls attn_backend.supports_packed_mask_free() to decide
        # whether to build the padding mask. Return False so the fake exercises
        # the masked path (the abstract backend default).
        @classmethod
        def supports_packed_mask_free(cls) -> bool:
            return False

    def __init__(self, **kwargs):
        del kwargs
        super().__init__()
        self.attn_backend = self.FakeBackend
        self.use_ring = False

    def forward(self, q, k, v, metadata=None):
        del k, v, metadata
        return q


def _minimax_h3_small_od_config():
    arch = {
        "num_layers": 1,
        "token_refiner_num_layers": 1,
        "hidden_size": 8,
        "num_attention_heads": 1,
        "attention_head_dim": 8,
        "ffn_hidden_size": 16,
        "latents_dim": 2,
        "audio_latents_dim": 2,
        "patch_size": (1, 2, 2),
        "text_dim": 6,
        "timestep_input_dim": 4,
        "time_embed_hidden_size": 8,
        "time_embed_dim": 4,
        "adaln_out_features": 18 * 8,
        "final_adaln_out_features": 2 * 8,
        "rope_inv_freq_len": 1,
    }
    return SimpleNamespace(
        tf_model_config=arch,
        parallel_config=SimpleNamespace(ulysses_degree=1),
    )


def _minimax_h3_sample_kwargs(*, seq_len: int = 8, device: torch.device | str = "cpu") -> dict:
    video_row_dim = 2 * 1 * 2 * 2
    audio_row_dim = 2
    text_len = 2
    img_rows = 2

    img_pos = torch.tensor([2, 3], dtype=torch.long, device=device)
    audio_pos = torch.tensor([4, 5], dtype=torch.long, device=device)
    text_pos = torch.tensor([0, 1], dtype=torch.long, device=device)
    infer_out_pos = img_pos.clone()

    token_tags = torch.full((seq_len,), -1, dtype=torch.long, device=device)
    token_tags[text_pos] = 1
    token_tags[img_pos] = 0
    token_tags[audio_pos] = 2

    inverse_indices = torch.zeros(seq_len, dtype=torch.long, device=device)

    return {
        "x": torch.randn(1, seq_len, video_row_dim, device=device),
        "audio_x": torch.randn(1, seq_len, audio_row_dim, device=device),
        "img_position_ids": torch.zeros(1, seq_len, 3, dtype=torch.float64, device=device),
        "unique_timesteps": torch.tensor([0.5], dtype=torch.float32, device=device),
        "inverse_indices": inverse_indices,
        "update_mask": torch.ones(img_rows, dtype=torch.bool, device=device),
        "token_tags": token_tags,
        "prompt_embeds": torch.randn(text_len, 6, device=device),
        "img_pos_info": {"position_ids": img_pos},
        "audio_pos_info": {"position_ids": audio_pos},
        "text_pos_info": {"position_ids": text_pos},
        "img_pos_for_infer_output_info": {"position_ids": infer_out_pos},
        "packed_seq_params": {
            "cu_seqlens_q": torch.tensor([0, seq_len], dtype=torch.int32, device=device),
            "max_seqlen_q": seq_len,
        },
        "refiner_packed_seq_params": {
            "cu_seqlens_q": torch.tensor([0, text_len], dtype=torch.int32, device=device),
            "max_seqlen_q": text_len,
        },
    }


@pytest.mark.cpu
class TestMiniMaxH3Extractor(BaseExtractorTest):
    """Test extract_minimax_h3_context function."""

    def get_extractor(self):
        return extract_minimax_h3_context

    @pytest.fixture
    def minimax_h3_module(self, monkeypatch):
        """Create a minimal MiniMaxH3DiTModel with fake parallel linears for CPU tests."""
        from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

        monkeypatch.setattr(h3, "ColumnParallelLinear", _MiniMaxH3FakeLinear)
        monkeypatch.setattr(h3, "MergedColumnParallelLinear", _MiniMaxH3FakeMergedLinear)
        monkeypatch.setattr(h3, "QKVParallelLinear", _MiniMaxH3FakeQKVLinear)
        monkeypatch.setattr(h3, "RowParallelLinear", _MiniMaxH3FakeLinear)
        monkeypatch.setattr(h3, "Attention", _MiniMaxH3FakeAttention)
        monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 1)

        model = h3.MiniMaxH3DiTModel(_minimax_h3_small_od_config(), quant_config=None)
        for submodule in model.modules():
            if isinstance(submodule, h3.MiniMaxH3Attention):
                submodule.rope._forward_method = submodule.rope.forward_native
        model.eval()
        return model

    def get_module(self, minimax_h3_module):
        return minimax_h3_module

    @pytest.fixture
    def sample_inputs(self):
        """Create sample packed forward kwargs for MiniMaxH3DiTModel."""
        return _minimax_h3_sample_kwargs(seq_len=8)

    def get_sample_inputs(self, sample_inputs):
        return sample_inputs

    def test_modulated_input_shape(self, minimax_h3_module, sample_inputs):
        """Test that modulated_input matches packed sequence shape [T, H]."""
        context = extract_minimax_h3_context(minimax_h3_module, **sample_inputs)

        seq_len = sample_inputs["x"].shape[1]
        assert context.modulated_input.shape == (seq_len, minimax_h3_module.hidden_size)

    def test_run_transformer_blocks_callable(self, minimax_h3_module, sample_inputs):
        """Test that run_transformer_blocks is callable."""
        context = extract_minimax_h3_context(minimax_h3_module, **sample_inputs)
        assert callable(context.run_transformer_blocks)

    def test_postprocess_callable(self, minimax_h3_module, sample_inputs):
        """Test that postprocess is callable."""
        context = extract_minimax_h3_context(minimax_h3_module, **sample_inputs)
        assert callable(context.postprocess)

    def test_postprocess_output_shape(self, minimax_h3_module, sample_inputs):
        """Test that postprocess returns video/audio logits for infer rows."""
        context = extract_minimax_h3_context(minimax_h3_module, **sample_inputs)
        (hidden,) = context.run_transformer_blocks()
        video_logits, audio_logits = context.postprocess(hidden)

        num_video_rows = sample_inputs["img_pos_info"]["position_ids"].shape[0]
        num_audio_rows = sample_inputs["audio_pos_info"]["position_ids"].shape[0]
        assert video_logits.shape[0] == num_video_rows
        assert audio_logits.shape[0] == num_audio_rows

    def test_extractor_matches_native_forward(self, minimax_h3_module, sample_inputs):
        """The uncached extractor path must preserve the model's forward result."""
        expected_video, expected_audio = minimax_h3_module(**sample_inputs)

        context = extract_minimax_h3_context(minimax_h3_module, **sample_inputs)
        (hidden,) = context.run_transformer_blocks()
        actual_video, actual_audio = context.postprocess(hidden)

        torch.testing.assert_close(actual_video, expected_video)
        torch.testing.assert_close(actual_audio, expected_audio)

    def test_run_transformer_blocks_forwards_packed_layout(
        self,
        minimax_h3_module,
        sample_inputs,
        monkeypatch,
    ):
        """The extractor block call must mirror the native packed contract."""
        from vllm_omni.diffusion.attention.backends.abstract import VideoTokenLayout

        video_layout = VideoTokenLayout(prefix_len=2, latent_grid=(1, 1, 2))
        inputs = {**sample_inputs, "video_token_layout": video_layout}
        block = minimax_h3_module.blocks[0]
        original_forward = block.forward
        received: list[tuple[int, VideoTokenLayout | None]] = []

        def capture_forward(*args, **kwargs):
            received.append((kwargs["packed_total"], kwargs["video_layout"]))
            return original_forward(*args, **kwargs)

        monkeypatch.setattr(block, "forward", capture_forward)
        context = extract_minimax_h3_context(minimax_h3_module, **inputs)
        context.run_transformer_blocks()

        assert received == [(sample_inputs["x"].shape[1], video_layout)]

    def test_teacache_reuses_h3_block_residual(self, minimax_h3_module, sample_inputs, monkeypatch):
        """A guaranteed cache hit must skip the H3 transformer block."""
        from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
        from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook

        block = minimax_h3_module.blocks[0]
        original_forward = block.forward
        block_calls = 0

        def counted_forward(*args, **kwargs):
            nonlocal block_calls
            block_calls += 1
            return original_forward(*args, **kwargs)

        monkeypatch.setattr(block, "forward", counted_forward)
        hook = TeaCacheHook(
            TeaCacheConfig(
                transformer_type="MiniMaxH3DiTModel",
                rel_l1_thresh=0.3,
            )
        )
        hook.initialize_hook(minimax_h3_module)

        hook.new_forward(minimax_h3_module, **sample_inputs)
        video_logits, audio_logits = hook.new_forward(minimax_h3_module, **sample_inputs)

        assert block_calls == 1
        assert video_logits.shape[0] == sample_inputs["img_pos_info"]["position_ids"].shape[0]
        assert audio_logits.shape[0] == sample_inputs["audio_pos_info"]["position_ids"].shape[0]

    def test_invalid_module_raises_error(self):
        """Test that invalid module without blocks raises ValueError."""
        invalid_module = Mock()
        invalid_module.blocks = []

        with pytest.raises(ValueError, match="Module must have blocks"):
            extract_minimax_h3_context(invalid_module, **_minimax_h3_sample_kwargs())
