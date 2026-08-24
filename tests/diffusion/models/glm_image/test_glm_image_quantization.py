# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for GLM-Image quantization support (W4A16/AutoRound).

These tests verify that the GLM-Image DiT transformer correctly accepts and uses
quantization configs for W4A16/AutoRound quantization support.
"""

import os

import pytest
import torch
from pytest_mock import MockerFixture

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.models.glm_image.glm_image_transformer import (
    ColumnParallelGELU,
    ColumnParallelSiLU,
    GlmImageAdaLayerNormContinuous,
    GlmImageAdaLayerNormZero,
    GlmImageAttention,
    GlmImageFeedForward,
    GlmImageImageProjector,
    GlmImagePrepare,
    GlmImageRotaryPosEmbed,
    GlmImageTransformer2DModel,
    GlmImageTransformerBlock,
    _positive_divisors,
    validate_glm_image_tp_constraints,
)
from vllm_omni.model_executor.models.glm_image.pipeline import GLM_IMAGE_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="function", autouse=True)
def _init_distributed():
    """Initialize the minimal distributed environment required by
    ReplicatedLinear (tensor-parallel group must exist)."""
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29502")
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="env://",
    )
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


@pytest.fixture(scope="function", autouse=True)
def _force_default_gemm(monkeypatch):
    """Force CPU-compatible GEMM dispatch for tests using CPU tensors.

    vLLM's dispatch_unquantized_gemm() selects the backend by platform
    (e.g. rocm_unquantized_gemm on AMD machines), not by tensor device.
    CPU test tensors crash with NotImplementedError on ROCm.  Monkeypatch
    the dispatcher to always return the default (torch.nn.functional.linear)
    implementation which works on any device."""
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda: default_unquantized_gemm,
    )


@pytest.fixture(scope="function", autouse=True)
def setup_mocks(mocker: MockerFixture):
    """Set up common mocks for all tests."""
    mocker.patch(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    mock_get_tp_group = mocker.patch("vllm.distributed.parallel_state.get_tp_group")
    mock_tp_group = mocker.MagicMock()
    mock_tp_group.world_size = 1
    mock_get_tp_group.return_value = mock_tp_group
    yield


class TestPositiveDivisors:
    """Test _positive_divisors helper function."""

    def test_divisors_of_1(self):
        assert _positive_divisors(1) == {1}

    def test_divisors_of_12(self):
        assert _positive_divisors(12) == {1, 2, 3, 4, 6, 12}

    def test_divisors_of_prime(self):
        assert _positive_divisors(7) == {1, 7}

    def test_divisors_of_0_returns_empty(self):
        assert _positive_divisors(0) == set()

    def test_divisors_of_negative_returns_empty(self):
        assert _positive_divisors(-5) == set()


class TestValidateGlmImageTpConstraints:
    """Test TP constraint validation for GLM-Image."""

    def test_valid_tp_size_1(self):
        """TP=1 should always be valid."""
        result = validate_glm_image_tp_constraints(
            dim=2560,
            num_heads=64,
            ffn_hidden_dim=10240,
            tensor_parallel_size=1,
        )
        assert 1 in result

    def test_valid_tp_size_2_for_divisible_dim(self):
        """TP=2 is valid when all dims are divisible by 2."""
        result = validate_glm_image_tp_constraints(
            dim=2560,
            num_heads=64,
            ffn_hidden_dim=10240,
            tensor_parallel_size=2,
        )
        assert 1 in result
        assert 2 in result

    def test_valid_tp_size_4_for_divisible_dim(self):
        """TP=4 is valid when all dims are divisible by 4."""
        result = validate_glm_image_tp_constraints(
            dim=2560,
            num_heads=64,
            ffn_hidden_dim=10240,
            tensor_parallel_size=4,
        )
        assert 1 in result
        assert 2 in result
        assert 4 in result

    def test_invalid_tp_size_3_for_divisible_dim(self):
        """TP=3 is invalid when dim is not divisible by 3."""
        with pytest.raises(ValueError, match="dim % tensor_parallel_size == 0"):
            validate_glm_image_tp_constraints(
                dim=2560,  # 2560 % 3 != 0
                num_heads=64,
                ffn_hidden_dim=10240,
                tensor_parallel_size=3,
            )

    def test_invalid_tp_size_zero(self):
        """TP=0 should raise error."""
        with pytest.raises(ValueError, match="tensor_parallel_size must be > 0"):
            validate_glm_image_tp_constraints(
                dim=2560,
                num_heads=64,
                ffn_hidden_dim=10240,
                tensor_parallel_size=0,
            )

    def test_invalid_tp_size_negative(self):
        """Negative TP size should raise error."""
        with pytest.raises(ValueError, match="tensor_parallel_size must be > 0"):
            validate_glm_image_tp_constraints(
                dim=2560,
                num_heads=64,
                ffn_hidden_dim=10240,
                tensor_parallel_size=-1,
            )


class TestGlmImageAdaLayerNormZeroQuantization:
    """Test GlmImageAdaLayerNormZero with quantization config."""

    def test_accepts_quant_config_parameter(self, mocker: MockerFixture):
        """Verify the class accepts quant_config parameter."""
        mock_quant_config = mocker.MagicMock()
        layer = GlmImageAdaLayerNormZero(
            embedding_dim=512,
            dim=2560,
            quant_config=mock_quant_config,
            prefix="test.norm1",
        )
        assert layer.linear.quant_config is mock_quant_config

    def test_accepts_none_quant_config(self):
        """Verify quant_config=None is accepted."""
        layer = GlmImageAdaLayerNormZero(
            embedding_dim=512,
            dim=2560,
            quant_config=None,
        )
        assert layer.linear.quant_config is None

    def test_forward_handles_tuple_return_from_linear(self):
        """Verify forward handles tuple returns from ReplicatedLinear."""
        layer = GlmImageAdaLayerNormZero(
            embedding_dim=512,
            dim=2560,
            quant_config=None,
        )

        batch_size = 2
        seq_len = 10
        hidden_states = torch.randn(batch_size, seq_len, 2560)
        encoder_hidden_states = torch.randn(batch_size, seq_len, 2560)
        temb = torch.randn(batch_size, 512)

        # This should work regardless of whether linear returns tuple or tensor
        result = layer(hidden_states, encoder_hidden_states, temb)
        assert len(result) == 10  # Should return 10 chunks


class TestGlmImageAdaLayerNormContinuousQuantization:
    """Test GlmImageAdaLayerNormContinuous with quantization config."""

    def test_accepts_quant_config_parameter(self, mocker: MockerFixture):
        """Verify the class accepts quant_config parameter."""
        mock_quant_config = mocker.MagicMock()
        layer = GlmImageAdaLayerNormContinuous(
            embedding_dim=2560,
            conditioning_embedding_dim=512,
            quant_config=mock_quant_config,
            prefix="test.norm_out",
        )
        assert layer.linear.quant_config is mock_quant_config

    def test_accepts_none_quant_config(self):
        """Verify quant_config=None is accepted."""
        layer = GlmImageAdaLayerNormContinuous(
            embedding_dim=2560,
            conditioning_embedding_dim=512,
            quant_config=None,
        )
        assert layer.linear.quant_config is None

    def test_forward_handles_tuple_return_from_linear(self):
        """Verify forward handles tuple returns from ReplicatedLinear."""
        layer = GlmImageAdaLayerNormContinuous(
            embedding_dim=2560,
            conditioning_embedding_dim=512,
            quant_config=None,
        )

        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 2560)
        conditioning_embedding = torch.randn(batch_size, 512)

        result = layer(x, conditioning_embedding)
        assert result.shape == (batch_size, seq_len, 2560)


class TestGlmImageAttentionQuantization:
    """Test GlmImageAttention with quantization config."""

    def test_accepts_quant_config_parameter(self, mocker: MockerFixture):
        """Verify GlmImageAttention accepts quant_config parameter."""
        mock_quant_config = mocker.MagicMock()
        attn = GlmImageAttention(
            dim=2560,
            num_heads=64,
            head_dim=40,
            quant_config=mock_quant_config,
            prefix="test.attn1",
        )
        assert attn.to_qkv.quant_config is mock_quant_config

    def test_accepts_none_quant_config(self):
        """Verify quant_config=None is accepted."""
        attn = GlmImageAttention(
            dim=2560,
            num_heads=64,
            head_dim=40,
            quant_config=None,
        )
        assert attn.to_qkv.quant_config is None


class TestColumnParallelModulesQuantization:
    """Test ColumnParallelGELU and ColumnParallelSiLU with quantization config."""

    def test_column_parallel_gelu_accepts_quant_config(self, mocker: MockerFixture):
        """Verify ColumnParallelGELU accepts quant_config."""
        mock_quant_config = mocker.MagicMock()
        layer = ColumnParallelGELU(
            dim_in=2560,
            dim_out=10240,
            quant_config=mock_quant_config,
            prefix="test.gelu",
        )
        assert layer.proj.quant_config is mock_quant_config

    def test_column_parallel_silu_accepts_quant_config(self, mocker: MockerFixture):
        """Verify ColumnParallelSiLU accepts quant_config."""
        mock_quant_config = mocker.MagicMock()
        layer = ColumnParallelSiLU(
            dim_in=2560,
            dim_out=10240,
            quant_config=mock_quant_config,
            prefix="test.silu",
        )
        assert layer.proj.quant_config is mock_quant_config


class TestGlmImageFeedForwardQuantization:
    """Test GlmImageFeedForward with quantization config."""

    def test_accepts_quant_config_parameter(self, mocker: MockerFixture):
        """Verify GlmImageFeedForward accepts quant_config parameter."""
        mock_quant_config = mocker.MagicMock()
        ff = GlmImageFeedForward(
            dim=2560,
            dim_out=2560,
            inner_dim=10240,
            activation_fn="gelu-approximate",
            quant_config=mock_quant_config,
            prefix="test.ff",
        )
        # Check that the first layer (ColumnParallelGELU) has quant_config
        gelu_layer = ff.net[0]
        assert gelu_layer.proj.quant_config is mock_quant_config

    def test_accepts_none_quant_config(self):
        """Verify quant_config=None is accepted."""
        ff = GlmImageFeedForward(
            dim=2560,
            dim_out=2560,
            inner_dim=10240,
            activation_fn="gelu-approximate",
            quant_config=None,
        )
        gelu_layer = ff.net[0]
        assert gelu_layer.proj.quant_config is None

    def test_linear_silu_activation(self):
        """Test linear-silu activation function initialization works."""
        # This test verifies the module can be instantiated with linear-silu activation.
        # Full forward testing requires proper TP group setup, so we just verify construction.
        ff = GlmImageFeedForward(
            dim=2560,
            dim_out=2560,
            inner_dim=10240,
            activation_fn="linear-silu",
            quant_config=None,
        )
        # Verify the FFN has the correct structure
        assert len(ff.net) == 3  # ColumnParallelSiLU, Identity, RowParallelLinear
        assert isinstance(ff.net[0], ColumnParallelSiLU)


class TestGlmImageTransformerBlockQuantization:
    """Test GlmImageTransformerBlock with quantization config."""

    def test_accepts_quant_config_parameter(self, mocker: MockerFixture):
        """Verify GlmImageTransformerBlock accepts quant_config parameter."""
        mock_quant_config = mocker.MagicMock()
        parallel_config = DiffusionParallelConfig(
            tensor_parallel_size=1,
            sequence_parallel_size=1,
        )
        block = GlmImageTransformerBlock(
            dim=2560,
            num_attention_heads=64,
            attention_head_dim=40,
            time_embed_dim=512,
            parallel_config=parallel_config,
            quant_config=mock_quant_config,
            prefix="test.block",
        )
        # Check that inner modules have quant_config
        assert block.norm1.linear.quant_config is mock_quant_config
        assert block.attn1.to_qkv.quant_config is mock_quant_config
        assert block.ff.net[0].proj.quant_config is mock_quant_config

    def test_accepts_none_quant_config(self):
        """Verify quant_config=None is accepted."""
        block = GlmImageTransformerBlock(
            dim=2560,
            num_attention_heads=64,
            attention_head_dim=40,
            time_embed_dim=512,
            quant_config=None,
        )
        assert block.norm1.linear.quant_config is None


class TestGlmImagePrepareModule:
    """Test GlmImagePrepare module."""

    def test_prepare_module_exists(self):
        """Verify GlmImagePrepare module exists and works."""
        projector = GlmImageImageProjector(in_channels=16, hidden_size=2560, patch_size=2)
        rope = GlmImageRotaryPosEmbed(dim=40, patch_size=2)
        prepare = GlmImagePrepare(
            image_projector=projector,
            rope=rope,
            patch_size=2,
        )

        # Test forward pass
        hidden_states = torch.randn(1, 16, 64, 64)  # [B, C, H, W]
        result = prepare(hidden_states)

        assert len(result) == 5
        hidden_out, rope_cos, rope_sin, height, width = result
        assert hidden_out.shape[0] == 1  # batch size
        assert hidden_out.shape[1] == 1024  # seq_len (32 * 32)
        assert hidden_out.shape[2] == 2560  # hidden_dim


class TestGlmImagePipelineConfig:
    """Test GLM_IMAGE_PIPELINE configuration."""

    def test_glm_image_pipeline_config_exists(self):
        """Verify GLM_IMAGE_PIPELINE config exists."""
        assert GLM_IMAGE_PIPELINE is not None

    def test_pipeline_has_correct_model_type(self):
        """Verify pipeline has correct model_type."""
        assert GLM_IMAGE_PIPELINE.model_type == "glm_image"

    def test_pipeline_has_correct_model_arch(self):
        """Verify pipeline has correct model_arch."""
        assert GLM_IMAGE_PIPELINE.model_arch == "GlmImageForConditionalGeneration"

    def test_pipeline_has_two_stages(self):
        """Verify pipeline has two stages (AR + DiT)."""
        assert len(GLM_IMAGE_PIPELINE.stages) == 2

    def test_stage_0_is_ar_llm(self):
        """Verify stage 0 is the AR LLM stage."""
        stage_0 = GLM_IMAGE_PIPELINE.stages[0]
        assert stage_0.stage_id == 0
        assert stage_0.model_stage == "ar"
        assert stage_0.execution_type.value == "llm_ar"
        assert stage_0.owns_tokenizer is True

    def test_stage_1_is_diffusion(self):
        """Verify stage 1 is the DiT diffusion stage."""
        stage_1 = GLM_IMAGE_PIPELINE.stages[1]
        assert stage_1.stage_id == 1
        assert stage_1.model_stage == "dit"
        assert stage_1.execution_type.value == "diffusion"
        assert stage_1.final_output is True
        assert stage_1.final_output_type == "image"
        assert stage_1.input_sources == (0,)  # Takes input from stage 0

    def test_pipeline_has_diffusers_class_name(self):
        """Verify pipeline has diffusers_class_name."""
        assert GLM_IMAGE_PIPELINE.diffusers_class_name == "GlmImagePipeline"


class TestGlmImageImageProjector:
    """Test GlmImageImageProjector module."""

    def test_projector_output_shape(self):
        """Verify projector produces correct output shape."""
        projector = GlmImageImageProjector(
            in_channels=16,
            hidden_size=2560,
            patch_size=2,
        )

        # Input: [B, C, H, W] = [1, 16, 64, 64]
        hidden_states = torch.randn(1, 16, 64, 64)
        output = projector(hidden_states)

        # After patchify: [B, H/2*W/2, D] = [1, 32*32, 2560] = [1, 1024, 2560]
        assert output.shape == (1, 1024, 2560)

    def test_projector_with_different_patch_sizes(self):
        """Verify projector works with different patch sizes."""
        for patch_size in [2, 4]:
            projector = GlmImageImageProjector(
                in_channels=16,
                hidden_size=2560,
                patch_size=patch_size,
            )
            h, w = 64, 64
            hidden_states = torch.randn(1, 16, h, w)
            output = projector(hidden_states)
            expected_seq_len = (h // patch_size) * (w // patch_size)
            assert output.shape[1] == expected_seq_len


class TestGlmImageRotaryPosEmbed:
    """Test GlmImageRotaryPosEmbed module."""

    def test_rope_output_shape(self):
        """Verify RoPE produces correct output shape."""
        rope = GlmImageRotaryPosEmbed(dim=40, patch_size=2, theta=10000.0)
        hidden_states = torch.randn(1, 16, 64, 64)  # [B, C, H, W]

        cos, sin = rope(hidden_states)

        # After patchify: height=32, width=32
        # Output: [32*32, dim] = [1024, 40]
        assert cos.shape[0] == 1024
        assert cos.shape[1] == 40
        assert sin.shape == cos.shape

    def test_rope_value_range(self):
        """Verify RoPE produces values in valid range."""
        rope = GlmImageRotaryPosEmbed(dim=40, patch_size=2, theta=10000.0)
        hidden_states = torch.randn(2, 16, 32, 32)

        cos, sin = rope(hidden_states)

        # cos and sin should be in [-1, 1]
        assert cos.min() >= -1.0 and cos.max() <= 1.0
        assert sin.min() >= -1.0 and sin.max() <= 1.0

    def test_rope_consistency(self):
        """Verify RoPE is consistent for same input."""
        rope = GlmImageRotaryPosEmbed(dim=40, patch_size=2, theta=10000.0)
        hidden_states = torch.randn(1, 16, 32, 32)

        cos1, sin1 = rope(hidden_states)
        cos2, sin2 = rope(hidden_states)

        # Same input should produce same output
        assert torch.allclose(cos1, cos2)
        assert torch.allclose(sin1, sin2)


class TestGlmImageTransformer2DModelQuantization:
    """Test GlmImageTransformer2DModel with quantization config."""

    def test_accepts_quant_config_parameter(self, mocker: MockerFixture):
        """Verify the model accepts quant_config parameter."""
        from vllm_omni.diffusion.data import OmniDiffusionConfig

        mock_quant_config = mocker.MagicMock()
        parallel_config = DiffusionParallelConfig(
            tensor_parallel_size=1,
            sequence_parallel_size=1,
        )

        # Create a minimal mock od_config
        mock_tf_config = mocker.MagicMock()
        mock_tf_config.patch_size = 2
        mock_tf_config.in_channels = 16
        mock_tf_config.out_channels = 16
        mock_tf_config.num_attention_heads = 64
        mock_tf_config.attention_head_dim = 40
        mock_tf_config.time_embed_dim = 512
        mock_tf_config.condition_dim = 256
        mock_tf_config.prior_vq_quantizer_codebook_size = 16384
        mock_tf_config.text_embed_dim = 1024
        mock_tf_config.num_layers = 2  # Small number for testing

        mock_od_config = mocker.MagicMock(spec=OmniDiffusionConfig)
        mock_od_config.tf_model_config = mock_tf_config
        mock_od_config.parallel_config = parallel_config

        model = GlmImageTransformer2DModel(
            od_config=mock_od_config,
            quant_config=mock_quant_config,
        )

        # Check that quantization config was passed to transformer blocks
        for block in model.transformer_blocks:
            assert block.norm1.linear.quant_config is mock_quant_config
            assert block.attn1.to_qkv.quant_config is mock_quant_config
            assert block.ff.net[0].proj.quant_config is mock_quant_config

    def test_accepts_none_quant_config(self, mocker: MockerFixture):
        """Verify quant_config=None is accepted."""
        from vllm_omni.diffusion.data import OmniDiffusionConfig

        parallel_config = DiffusionParallelConfig(
            tensor_parallel_size=1,
            sequence_parallel_size=1,
        )

        mock_tf_config = mocker.MagicMock()
        mock_tf_config.patch_size = 2
        mock_tf_config.in_channels = 16
        mock_tf_config.out_channels = 16
        mock_tf_config.num_attention_heads = 64
        mock_tf_config.attention_head_dim = 40
        mock_tf_config.time_embed_dim = 512
        mock_tf_config.condition_dim = 256
        mock_tf_config.prior_vq_quantizer_codebook_size = 16384
        mock_tf_config.text_embed_dim = 1024
        mock_tf_config.num_layers = 2

        mock_od_config = mocker.MagicMock(spec=OmniDiffusionConfig)
        mock_od_config.tf_model_config = mock_tf_config
        mock_od_config.parallel_config = parallel_config

        model = GlmImageTransformer2DModel(
            od_config=mock_od_config,
            quant_config=None,
        )

        # Check that quantization config is None
        for block in model.transformer_blocks:
            assert block.norm1.linear.quant_config is None
            assert block.attn1.to_qkv.quant_config is None

    def test_norm_out_has_no_quantization(self, mocker: MockerFixture):
        """Verify norm_out (output layer) does NOT use quantization to preserve precision."""
        from vllm_omni.diffusion.data import OmniDiffusionConfig

        mock_quant_config = mocker.MagicMock()
        parallel_config = DiffusionParallelConfig(
            tensor_parallel_size=1,
            sequence_parallel_size=1,
        )

        mock_tf_config = mocker.MagicMock()
        mock_tf_config.patch_size = 2
        mock_tf_config.in_channels = 16
        mock_tf_config.out_channels = 16
        mock_tf_config.num_attention_heads = 64
        mock_tf_config.attention_head_dim = 40
        mock_tf_config.time_embed_dim = 512
        mock_tf_config.condition_dim = 256
        mock_tf_config.prior_vq_quantizer_codebook_size = 16384
        mock_tf_config.text_embed_dim = 1024
        mock_tf_config.num_layers = 2

        mock_od_config = mocker.MagicMock(spec=OmniDiffusionConfig)
        mock_od_config.tf_model_config = mock_tf_config
        mock_od_config.parallel_config = parallel_config

        model = GlmImageTransformer2DModel(
            od_config=mock_od_config,
            quant_config=mock_quant_config,
        )

        # norm_out.linear should NOT have quant_config to preserve output precision
        assert model.norm_out.linear.quant_config is None

    def test_model_has_sp_plan(self, mocker: MockerFixture):
        """Verify model has _sp_plan defined for sequence parallelism."""
        from vllm_omni.diffusion.data import OmniDiffusionConfig

        parallel_config = DiffusionParallelConfig(
            tensor_parallel_size=1,
            sequence_parallel_size=1,
        )

        mock_tf_config = mocker.MagicMock()
        mock_tf_config.patch_size = 2
        mock_tf_config.in_channels = 16
        mock_tf_config.out_channels = 16
        mock_tf_config.num_attention_heads = 64
        mock_tf_config.attention_head_dim = 40
        mock_tf_config.time_embed_dim = 512
        mock_tf_config.condition_dim = 256
        mock_tf_config.prior_vq_quantizer_codebook_size = 16384
        mock_tf_config.text_embed_dim = 1024
        mock_tf_config.num_layers = 2

        mock_od_config = mocker.MagicMock(spec=OmniDiffusionConfig)
        mock_od_config.tf_model_config = mock_tf_config
        mock_od_config.parallel_config = parallel_config

        model = GlmImageTransformer2DModel(
            od_config=mock_od_config,
            quant_config=None,
        )

        # Verify _sp_plan exists
        assert hasattr(model, "_sp_plan")
        assert "prepare" in model._sp_plan
        assert "proj_out" in model._sp_plan

    def test_model_has_hsdp_shard_conditions(self, mocker: MockerFixture):
        """Verify model has _hsdp_shard_conditions for HSDP parallelism."""
        from vllm_omni.diffusion.data import OmniDiffusionConfig

        parallel_config = DiffusionParallelConfig(
            tensor_parallel_size=1,
            sequence_parallel_size=1,
        )

        mock_tf_config = mocker.MagicMock()
        mock_tf_config.patch_size = 2
        mock_tf_config.in_channels = 16
        mock_tf_config.out_channels = 16
        mock_tf_config.num_attention_heads = 64
        mock_tf_config.attention_head_dim = 40
        mock_tf_config.time_embed_dim = 512
        mock_tf_config.condition_dim = 256
        mock_tf_config.prior_vq_quantizer_codebook_size = 16384
        mock_tf_config.text_embed_dim = 1024
        mock_tf_config.num_layers = 2

        mock_od_config = mocker.MagicMock(spec=OmniDiffusionConfig)
        mock_od_config.tf_model_config = mock_tf_config
        mock_od_config.parallel_config = parallel_config

        model = GlmImageTransformer2DModel(
            od_config=mock_od_config,
            quant_config=None,
        )

        assert hasattr(model, "_hsdp_shard_conditions")
        assert len(model._hsdp_shard_conditions) > 0

    def test_model_creates_kv_cache(self, mocker: MockerFixture):
        """Verify model can create KV cache for image editing."""
        from vllm_omni.diffusion.data import OmniDiffusionConfig

        parallel_config = DiffusionParallelConfig(
            tensor_parallel_size=1,
            sequence_parallel_size=1,
        )

        mock_tf_config = mocker.MagicMock()
        mock_tf_config.patch_size = 2
        mock_tf_config.in_channels = 16
        mock_tf_config.out_channels = 16
        mock_tf_config.num_attention_heads = 64
        mock_tf_config.attention_head_dim = 40
        mock_tf_config.time_embed_dim = 512
        mock_tf_config.condition_dim = 256
        mock_tf_config.prior_vq_quantizer_codebook_size = 16384
        mock_tf_config.text_embed_dim = 1024
        mock_tf_config.num_layers = 2

        mock_od_config = mocker.MagicMock(spec=OmniDiffusionConfig)
        mock_od_config.tf_model_config = mock_tf_config
        mock_od_config.parallel_config = parallel_config

        model = GlmImageTransformer2DModel(
            od_config=mock_od_config,
            quant_config=None,
        )

        kv_cache = model.create_kv_cache()
        assert kv_cache is not None
        assert len(kv_cache) == 2  # num_layers
