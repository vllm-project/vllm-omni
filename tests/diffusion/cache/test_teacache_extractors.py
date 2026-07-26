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
- TestHiDreamExtractor: HiDream-I1-Full model extractor
"""

from abc import ABC, abstractmethod
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.cache.teacache.config import _MODEL_COEFFICIENTS, TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.extractors import (
    extract_flux2_context,
    extract_flux2_klein_context,
    extract_flux_context,
    extract_hidream_image_context,
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


@pytest.mark.cpu
@pytest.mark.cache
class TestHiDreamExtractor:
    """Test extract_hidream_image_context function."""

    TRANSFORMER_TYPE = "HiDreamImageTransformer2DModel"
    BATCH_SIZE = 1
    HIDDEN_SIZE = 8
    PATCH_TOKENS = 4
    PATCH_DIM = 16
    TEXT_LEN = 3

    def test_bootstrap_coefficients_exist(self):
        assert self.TRANSFORMER_TYPE in _MODEL_COEFFICIENTS
        config = TeaCacheConfig(transformer_type=self.TRANSFORMER_TYPE)
        assert len(config.coefficients) == 5

    @pytest.fixture
    def hidream_module(self):
        batch_size = self.BATCH_SIZE
        hidden_size = self.HIDDEN_SIZE
        patch_tokens = self.PATCH_TOKENS
        patch_dim = self.PATCH_DIM
        text_len = self.TEXT_LEN

        module = Mock()
        module.training = False
        module.llama_layers = [0, 1]
        module.caption_projection = [
            Mock(side_effect=lambda x: x),
            Mock(side_effect=lambda x: x),
            Mock(side_effect=lambda x: x),
        ]
        module.patchify = Mock(
            return_value=(
                torch.randn(batch_size, patch_tokens, patch_dim),
                torch.ones(batch_size, patch_tokens),
                torch.tensor([[2, 2]], dtype=torch.int64),
                torch.zeros(batch_size, patch_tokens, 3),
            )
        )
        module.x_embedder = Mock(side_effect=lambda x: torch.randn(batch_size, patch_tokens, hidden_size))
        module.t_embedder = Mock(return_value=torch.randn(batch_size, hidden_size))
        module.p_embedder = Mock(return_value=torch.randn(batch_size, hidden_size))
        module.pe_embedder = Mock(
            side_effect=lambda ids: torch.randn(
                ids.shape[0],
                ids.shape[1],
                2,
                2,
                hidden_size,
            ),
        )

        inner_block = Mock()
        inner_block.adaLN_modulation = Mock(
            return_value=torch.randn(batch_size, hidden_size * 12),
        )
        inner_block.norm1_i = Mock(return_value=torch.randn(batch_size, patch_tokens, hidden_size))

        double_block = Mock(
            side_effect=lambda hidden_states, hidden_states_masks, encoder_hidden_states, temb, image_rotary_emb: (
                hidden_states + 0.1,
                encoder_hidden_states[:, : text_len * 2, :],
            )
        )
        double_block.block = inner_block
        module.double_stream_blocks = [double_block]
        module.single_stream_blocks = [Mock(side_effect=lambda **kwargs: kwargs["hidden_states"] + 0.2)]
        module.final_layer = Mock(side_effect=lambda h, temb: h)
        module.unpatchify = Mock(return_value=torch.randn(batch_size, 4, 8, 8))
        return module

    @pytest.fixture
    def sample_inputs(self):
        batch_size = self.BATCH_SIZE
        hidden_size = self.HIDDEN_SIZE
        text_len = self.TEXT_LEN
        return {
            "hidden_states": torch.randn(batch_size, 4, 8, 8),
            "timesteps": torch.tensor([10]),
            "encoder_hidden_states_t5": torch.randn(batch_size, text_len, hidden_size),
            "encoder_hidden_states_llama3": [
                torch.randn(batch_size, text_len, hidden_size),
                torch.randn(batch_size, text_len, hidden_size),
            ],
            "pooled_embeds": torch.randn(batch_size, hidden_size),
            "return_dict": False,
        }

    def test_modulated_input_shape(self, hidream_module, sample_inputs):
        """Test that modulated_input matches HiDream t_embedder output shape."""
        context = extract_hidream_image_context(hidream_module, **sample_inputs)
        context.validate()

        assert context.modulated_input.shape == (self.BATCH_SIZE, self.HIDDEN_SIZE)

    def test_run_transformer_blocks_callable(self, hidream_module, sample_inputs):
        """Test that run_transformer_blocks is callable."""
        context = extract_hidream_image_context(hidream_module, **sample_inputs)
        assert callable(context.run_transformer_blocks)

    def test_postprocess_callable(self, hidream_module, sample_inputs):
        """Test that postprocess is callable."""
        context = extract_hidream_image_context(hidream_module, **sample_inputs)
        assert callable(context.postprocess)

    def test_postprocess_output_shape(self, hidream_module, sample_inputs):
        """Test that postprocess returns unpatchified latents."""
        context = extract_hidream_image_context(hidream_module, **sample_inputs)
        output = context.postprocess(context.run_transformer_blocks()[0])

        assert isinstance(output, tuple)
        assert output[0].shape == (self.BATCH_SIZE, 4, 8, 8)

    def test_postprocess_return_tuple_when_return_dict_false(self, hidream_module, sample_inputs):
        """Test that postprocess honors return_dict=False."""
        context = extract_hidream_image_context(hidream_module, **sample_inputs)
        output = context.postprocess(context.hidden_states)

        assert isinstance(output, tuple)
        assert len(output) == 1
        assert isinstance(output[0], torch.Tensor)

    def test_postprocess_return_dict_when_return_dict_true(self, hidream_module, sample_inputs):
        """Test that postprocess returns Transformer2DModelOutput when return_dict=True."""
        from diffusers.models.modeling_outputs import Transformer2DModelOutput

        inputs = sample_inputs.copy()
        inputs["return_dict"] = True
        context = extract_hidream_image_context(hidream_module, **inputs)
        output = context.postprocess(context.hidden_states)

        assert isinstance(output, Transformer2DModelOutput)
        assert isinstance(output.sample, torch.Tensor)

    def test_deprecated_encoder_hidden_states_kwarg(self, hidream_module, sample_inputs):
        """Test deprecated bundled encoder_hidden_states kwarg is unpacked."""
        inputs = sample_inputs.copy()
        inputs.pop("encoder_hidden_states_t5")
        inputs.pop("encoder_hidden_states_llama3")
        inputs["encoder_hidden_states"] = [
            sample_inputs["encoder_hidden_states_t5"],
            sample_inputs["encoder_hidden_states_llama3"],
        ]

        context = extract_hidream_image_context(hidream_module, **inputs)
        context.validate()

    def test_pre_patchified_inputs_require_img_ids(self, hidream_module, sample_inputs):
        """Test pre-patchified inputs require img_ids and img_sizes."""
        inputs = sample_inputs.copy()
        inputs["hidden_states"] = torch.randn(self.BATCH_SIZE, self.PATCH_TOKENS, self.PATCH_DIM)
        inputs["hidden_states_masks"] = torch.ones(self.BATCH_SIZE, self.PATCH_TOKENS)

        with pytest.raises(ValueError, match="img_ids.*img_sizes"):
            extract_hidream_image_context(hidream_module, **inputs)

    def test_encoder_hidden_states_is_none(self, hidream_module, sample_inputs):
        """HiDream keeps text conditioning inside block closures, not CacheContext."""
        context = extract_hidream_image_context(hidream_module, **sample_inputs)
        assert context.encoder_hidden_states is None
