# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for Ovis-Image tensor parallelism.

This module tests:
1. Weight loading with TP sharding for special layers
2. Packed module mapping for QKV projections
3. Forward pass shape consistency

The weight loading logic for Ovis-Image requires manual sharding for:
1. proj_out in single_transformer_blocks: splits [attn, mlp] features separately
2. SwiGLU layers: re-interleaves [hidden, gate] pairs after sharding
"""

import pytest
import torch
from pytest_mock import MockerFixture

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.models.ovis_image.ovis_image_transformer import (
    OvisImagePosEmbed,
    OvisImageTransformer2DModel,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def setup_tp_group(mocker: MockerFixture):
    """Set up TP group for each test function (default TP=2)."""
    mocker.patch(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
        return_value=2,
    )
    mocker.patch(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
        return_value=0,
    )
    mock_get_tp_group = mocker.patch("vllm.distributed.parallel_state.get_tp_group")
    mock_tp_group = mocker.MagicMock()
    mock_tp_group.world_size = 2
    mock_get_tp_group.return_value = mock_tp_group
    yield


@pytest.fixture(scope="function")
def setup_tp1(mocker: MockerFixture):
    """Set up TP=1 environment."""
    mocker.patch(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    mocker.patch(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
        return_value=0,
    )
    mocker.patch(
        "vllm_omni.diffusion.models.ovis_image.ovis_image_transformer.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    mocker.patch(
        "vllm_omni.diffusion.models.ovis_image.ovis_image_transformer.get_tensor_model_parallel_rank",
        return_value=0,
    )


@pytest.fixture(scope="function")
def setup_tp2(mocker: MockerFixture):
    """Set up TP=2 environment for ovis_image_transformer."""
    mocker.patch(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
        return_value=2,
    )
    mocker.patch(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
        return_value=0,
    )
    mocker.patch(
        "vllm_omni.diffusion.models.ovis_image.ovis_image_transformer.get_tensor_model_parallel_world_size",
        return_value=2,
    )
    mocker.patch(
        "vllm_omni.diffusion.models.ovis_image.ovis_image_transformer.get_tensor_model_parallel_rank",
        return_value=0,
    )


# ============================================================================
# Test Classes
# ============================================================================


class TestOvisImageProjOutWeightSharding:
    """Test proj_out weight sharding for single_transformer_blocks.

    proj_out input is torch.cat([attn_output, mlp_hidden_states], dim=-1) where:
    - attn_output: [batch, seq, inner_dim / tp_size] (TP-sharded)
    - mlp_hidden_states: [batch, seq, mlp_hidden_dim / tp_size] (TP-sharded)

    The weight must be split into attn and mlp portions, each sharded separately.
    """

    pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

    def test_proj_out_weight_split_correctness(self, setup_tp2):
        """Verify proj_out weight is correctly split for TP=2."""
        inner_dim = 512
        mlp_ratio = 4.0
        mlp_hidden_dim = int(inner_dim * mlp_ratio)

        # Full weight shape: [inner_dim, inner_dim + mlp_hidden_dim]
        full_weight = torch.randn(inner_dim, inner_dim + mlp_hidden_dim)

        # Simulate the load_weights logic for rank 0
        tp_size = 2
        tp_rank = 0

        w_attn, w_mlp = full_weight.split([inner_dim, mlp_hidden_dim], dim=1)
        w_attn_local = w_attn.chunk(tp_size, dim=1)[tp_rank]
        w_mlp_local = w_mlp.chunk(tp_size, dim=1)[tp_rank]
        local_weight_rank0 = torch.cat([w_attn_local, w_mlp_local], dim=1)

        # Verify shapes
        assert w_attn_local.shape == (inner_dim, inner_dim // tp_size)
        assert w_mlp_local.shape == (inner_dim, mlp_hidden_dim // tp_size)
        assert local_weight_rank0.shape == (
            inner_dim,
            (inner_dim + mlp_hidden_dim) // tp_size,
        )

    def test_proj_out_weight_different_ranks(self, setup_tp2):
        """Verify different TP ranks get different weight shards."""
        inner_dim = 512
        mlp_hidden_dim = inner_dim * 4
        full_weight = torch.randn(inner_dim, inner_dim + mlp_hidden_dim)

        tp_size = 2

        # Rank 0
        w_attn, w_mlp = full_weight.split([inner_dim, mlp_hidden_dim], dim=1)
        local_rank0 = torch.cat([w_attn.chunk(tp_size, dim=1)[0], w_mlp.chunk(tp_size, dim=1)[0]], dim=1)

        # Rank 1
        local_rank1 = torch.cat([w_attn.chunk(tp_size, dim=1)[1], w_mlp.chunk(tp_size, dim=1)[1]], dim=1)

        # Ranks should have different weights
        assert not torch.allclose(local_rank0, local_rank1)

    def test_proj_out_weight_reconstruction(self, setup_tp2):
        """Verify full weight can be reconstructed from TP shards."""
        inner_dim = 512
        mlp_hidden_dim = inner_dim * 4
        full_weight = torch.randn(inner_dim, inner_dim + mlp_hidden_dim)

        tp_size = 2
        w_attn, w_mlp = full_weight.split([inner_dim, mlp_hidden_dim], dim=1)

        # Get all shards
        attn_shards = w_attn.chunk(tp_size, dim=1)
        mlp_shards = w_mlp.chunk(tp_size, dim=1)

        # Reconstruct
        reconstructed = torch.cat(
            [
                torch.cat([attn_shards[0], attn_shards[1]], dim=1),
                torch.cat([mlp_shards[0], mlp_shards[1]], dim=1),
            ],
            dim=1,
        )

        assert torch.allclose(reconstructed, full_weight)


class TestOvisImageSwiGLUWeightSharding:
    """Test SwiGLU weight interleaving for tensor parallelism.

    SwiGLU weight layout is [hidden, gate] interleaved. For TP, we shard each
    portion separately then re-interleave so each rank gets matching hidden/gate pairs.
    """

    pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

    def test_swiglu_weight_split_correctness(self, setup_tp2):
        """Verify SwiGLU weight is correctly interleaved for TP=2."""
        hidden_dim = 512
        inner_dim = 2048

        # Full weight shape: [inner_dim * 2, hidden_dim] for ColumnParallelLinear
        full_weight = torch.randn(inner_dim * 2, hidden_dim)

        tp_size = 2
        tp_rank = 0

        w_hidden, w_gate = full_weight.chunk(2, dim=0)
        w_h_local = w_hidden.chunk(tp_size, dim=0)[tp_rank]
        w_g_local = w_gate.chunk(tp_size, dim=0)[tp_rank]
        local_weight = torch.cat([w_h_local, w_g_local], dim=0)

        # Verify shapes
        assert w_h_local.shape == (inner_dim // tp_size, hidden_dim)
        assert w_g_local.shape == (inner_dim // tp_size, hidden_dim)
        assert local_weight.shape == (inner_dim, hidden_dim)

    def test_swiglu_bias_split_correctness(self, setup_tp2):
        """Verify SwiGLU bias is correctly interleaved for TP=2."""
        inner_dim = 2048
        full_bias = torch.randn(inner_dim * 2)

        tp_size = 2
        tp_rank = 0

        b_hidden, b_gate = full_bias.chunk(2, dim=0)
        b_h_local = b_hidden.chunk(tp_size, dim=0)[tp_rank]
        b_g_local = b_gate.chunk(tp_size, dim=0)[tp_rank]
        local_bias = torch.cat([b_h_local, b_g_local], dim=0)

        assert local_bias.shape == (inner_dim,)

    def test_swiglu_weight_reconstruction(self, setup_tp2):
        """Verify full SwiGLU weight can be reconstructed from TP shards."""
        hidden_dim = 512
        inner_dim = 2048
        full_weight = torch.randn(inner_dim * 2, hidden_dim)

        tp_size = 2
        w_hidden, w_gate = full_weight.chunk(2, dim=0)

        # Get all shards and reconstruct
        h_shards = w_hidden.chunk(tp_size, dim=0)
        g_shards = w_gate.chunk(tp_size, dim=0)

        reconstructed = torch.cat(
            [
                torch.cat([h_shards[0], h_shards[1]], dim=0),
                torch.cat([g_shards[0], g_shards[1]], dim=0),
            ],
            dim=0,
        )

        assert torch.allclose(reconstructed, full_weight)


class TestOvisImageWeightLoadingIntegration:
    """Integration tests for Ovis-Image weight loading with TP."""

    @pytest.mark.core_model
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_weight_loading_tp2(self, mocker: MockerFixture, setup_tp2):
        """Verify weights load correctly with TP=2."""
        # Mock distributed functions
        mocker.patch(
            "vllm.distributed.get_tensor_model_parallel_world_size",
            return_value=2,
        )
        mocker.patch(
            "vllm.distributed.get_tensor_model_parallel_rank",
            return_value=0,
        )

        inner_dim = 8 * 64  # num_heads * head_dim
        mlp_hidden_dim = inner_dim * 4

        # Create minimal config
        od_config = OmniDiffusionConfig(
            model="test",
            model_class_name="OvisImagePipeline",
            tf_model_config=TransformerConfig(
                num_layers=1,
                num_single_layers=1,
            ),
        )

        model = OvisImageTransformer2DModel(
            od_config=od_config,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=64,
            num_attention_heads=8,
            quant_config=None,
        )

        # Create mock weights
        mock_weights = [
            # Embeddings
            ("time_proj.weight", torch.randn(256, inner_dim)),
            ("timestep_embedder.linear_1.weight", torch.randn(inner_dim, 256)),
            ("timestep_embedder.linear_2.weight", torch.randn(inner_dim * 4, inner_dim)),
            ("context_embedder.weight", torch.randn(inner_dim, 2048)),
            ("x_embedder.weight", torch.randn(inner_dim, 64)),
            ("proj_out.weight", torch.randn(64, inner_dim)),
            # Transformer block
            ("transformer_blocks.0.norm1.linear.weight", torch.randn(inner_dim * 6, inner_dim)),
            ("transformer_blocks.0.attn.to_qkv.weight", torch.randn(inner_dim * 3, inner_dim)),
            ("transformer_blocks.0.attn.add_kv_proj.weight", torch.randn(inner_dim * 3, inner_dim)),
            ("transformer_blocks.0.attn.to_out.0.weight", torch.randn(inner_dim, inner_dim)),
            ("transformer_blocks.0.ff.net.0.proj.weight", torch.randn(mlp_hidden_dim * 2, inner_dim)),
            ("transformer_blocks.0.ff.net.2.weight", torch.randn(inner_dim, mlp_hidden_dim)),
            # Single block
            ("single_transformer_blocks.0.proj_mlp.weight", torch.randn(mlp_hidden_dim * 2, inner_dim)),
            ("single_transformer_blocks.0.attn.to_qkv.weight", torch.randn(inner_dim * 3, inner_dim)),
            ("single_transformer_blocks.0.proj_out.weight", torch.randn(inner_dim, inner_dim + mlp_hidden_dim)),
        ]

        loaded_params = model.load_weights(mock_weights)

        assert len(loaded_params) > 0, "Parameters should be loaded"

        # Verify proj_out weight shape after TP sharding
        proj_out_weight = model.single_transformer_blocks[0].proj_out.weight
        expected_shape = (inner_dim, (inner_dim + mlp_hidden_dim) // 2)
        assert proj_out_weight.shape == expected_shape, (
            f"Expected proj_out shape {expected_shape}, got {proj_out_weight.shape}"
        )

        # Verify proj_mlp weight shape
        proj_mlp_weight = model.single_transformer_blocks[0].proj_mlp.weight
        expected_mlp_shape = (mlp_hidden_dim, inner_dim)
        assert proj_mlp_weight.shape == expected_mlp_shape

    @pytest.mark.core_model
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_packed_module_mapping(self, mocker: MockerFixture, setup_tp_group):
        """Verify QKV packing matches expected configuration."""
        mocker.patch(
            "vllm_omni.diffusion.models.ovis_image.ovis_image_transformer.get_tensor_model_parallel_world_size",
            return_value=2,
        )
        mocker.patch(
            "vllm_omni.diffusion.models.ovis_image.ovis_image_transformer.get_tensor_model_parallel_rank",
            return_value=0,
        )

        od_config = OmniDiffusionConfig(
            model="test",
            model_class_name="OvisImagePipeline",
            tf_model_config=TransformerConfig(num_layers=1, num_single_layers=0),
        )

        model = OvisImageTransformer2DModel(
            od_config=od_config,
            num_layers=1,
            num_single_layers=0,
            attention_head_dim=64,
            num_attention_heads=8,
            quant_config=None,
        )

        model.load_weights([])

        # Verify stacked_params_mapping
        assert model.stacked_params_mapping is not None
        expected_mappings = [
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
            (".add_kv_proj", ".add_q_proj", "q"),
            (".add_kv_proj", ".add_k_proj", "k"),
            (".add_kv_proj", ".add_v_proj", "v"),
        ]
        assert model.stacked_params_mapping == expected_mappings

    @pytest.mark.core_model
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_unexpected_parameter_warning(self, mocker: MockerFixture, setup_tp2):
        """Test that unexpected parameters trigger a warning."""
        mocker.patch(
            "vllm.distributed.get_tensor_model_parallel_world_size",
            return_value=2,
        )
        mocker.patch(
            "vllm.distributed.get_tensor_model_parallel_rank",
            return_value=0,
        )

        od_config = OmniDiffusionConfig(
            model="test",
            model_class_name="OvisImagePipeline",
            tf_model_config=TransformerConfig(num_layers=1, num_single_layers=0),
        )

        model = OvisImageTransformer2DModel(
            od_config=od_config,
            num_layers=1,
            num_single_layers=0,
            attention_head_dim=64,
            num_attention_heads=8,
            quant_config=None,
        )

        # Load weights with invalid names
        invalid_weights = [("invalid.weight", torch.randn(10, 10))]
        loaded_params = model.load_weights(invalid_weights)

        assert len(loaded_params) == 0, "Should not load invalid weights"


class TestOvisImageRopePositionEmbedding:
    """Test Ovis-Image RoPE position embedding functionality."""

    pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

    def test_rope_position_embedding(self):
        """Verify RoPE produces correct embeddings for 3D coordinates."""
        # Ovis-Image default: axes_dims_rope = (16, 56, 56)
        axes_dims = (16, 56, 56)
        rope_theta = 10000
        pos_embed = OvisImagePosEmbed(theta=rope_theta, axes_dim=axes_dims)

        seq_len = 10
        ids = torch.randint(0, 100, (seq_len, 3))

        freqs_cos, freqs_sin = pos_embed(ids)

        # Expected dimension: sum(axes_dims) // 2
        expected_dim = sum(axes_dims) // 2  # 128/2 = 64
        assert freqs_cos.shape == (seq_len, expected_dim)
        assert freqs_sin.shape == (seq_len, expected_dim)

        # Verify value range
        assert torch.all(freqs_cos >= -1) and torch.all(freqs_cos <= 1)
        assert torch.all(freqs_sin >= -1) and torch.all(freqs_sin <= 1)

        # Verify trigonometric relationship
        cos_sq_sin_sq = freqs_cos**2 + freqs_sin**2
        assert torch.allclose(cos_sq_sin_sq, torch.ones_like(cos_sq_sin_sq), atol=1e-6)

    def test_rope_different_positions(self):
        """Verify different positions produce different embeddings."""
        axes_dims = (16, 56, 56)
        pos_embed = OvisImagePosEmbed(theta=10000, axes_dim=axes_dims)

        ids1 = torch.randint(0, 100, (10, 3))
        ids2 = torch.randint(0, 100, (10, 3))

        freqs_cos1, _ = pos_embed(ids1)
        freqs_cos2, _ = pos_embed(ids2)

        assert not torch.allclose(freqs_cos1, freqs_cos2)

    def test_rope_same_positions(self):
        """Verify same positions produce same embeddings."""
        axes_dims = (16, 56, 56)
        pos_embed = OvisImagePosEmbed(theta=10000, axes_dim=axes_dims)

        ids = torch.randint(0, 100, (10, 3))

        freqs_cos1, freqs_sin1 = pos_embed(ids)
        freqs_cos2, freqs_sin2 = pos_embed(ids.clone())

        assert torch.allclose(freqs_cos1, freqs_cos2)
        assert torch.allclose(freqs_sin1, freqs_sin2)


class TestOvisImageForwardShape:
    """Test Ovis-Image forward pass shape consistency."""

    @pytest.mark.core_model
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_forward_shape_consistency(self, mocker: MockerFixture, setup_tp1):
        """Verify forward pass produces correct shapes."""
        mocker.patch(
            "vllm.distributed.get_tensor_model_parallel_world_size",
            return_value=1,
        )
        mocker.patch(
            "vllm.distributed.get_tensor_model_parallel_rank",
            return_value=0,
        )

        od_config = OmniDiffusionConfig(
            model="test",
            model_class_name="OvisImagePipeline",
            tf_model_config=TransformerConfig(num_layers=1, num_single_layers=1),
        )

        model = OvisImageTransformer2DModel(
            od_config=od_config,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=64,
            num_attention_heads=8,
            quant_config=None,
        )

        inner_dim = 8 * 64

        # Create dummy inputs
        batch_size = 1
        seq_len = 16
        encoder_seq_len = 8

        hidden_states = torch.randn(batch_size, seq_len, inner_dim)
        encoder_hidden_states = torch.randn(batch_size, encoder_seq_len, 2048)
        temb = torch.randn(batch_size, inner_dim)

        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
        )

        assert output.sample.shape == hidden_states.shape
