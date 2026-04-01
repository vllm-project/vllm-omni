from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.utils import hardware_test
from vllm_omni.diffusion.models.flux2.flux2_transformer import (
    Flux2PosEmbed,
    Flux2Transformer2DModel,
)


# Initialize TP group before tests
@pytest.fixture(scope="function", autouse=True)
def setup_tp_group():
    """Set up TP group for each test function"""
    with patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", return_value=2):
        with patch("vllm.distributed.parallel_state.get_tp_group") as mock_get_tp_group:
            mock_tp_group = MagicMock()
            mock_tp_group.world_size = 2
            mock_get_tp_group.return_value = mock_tp_group
            yield


class TestFlux2TransformerWeightLoading:
    """Test Flux2Transformer weight loading functionality"""

    @pytest.mark.core_model
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_weight_loading_tp2(self, setup_tp_group):
        """Verify weights load correctly with TP=2"""
        # Prepare test data
        model = Flux2Transformer2DModel(
            num_layers=2,
            num_single_layers=1,
            num_attention_heads=48,
            attention_head_dim=128,
            joint_attention_dim=15360,
        )

        # Mock TP=2 weight loading
        mock_weights = []

        # 1. Test regular weight loading
        mock_weights.append(("x_embedder.weight", torch.randn(6144, 128)))
        mock_weights.append(("context_embedder.weight", torch.randn(6144, 15360)))
        mock_weights.append(("proj_out.weight", torch.randn(128, 6144)))

        # 2. Test stacked_params_mapping weight loading
        # Full weights - load_weights handles sharding internally
        to_qkv_weight = torch.randn(18432, 6144)  # Full weights
        mock_weights.append(("transformer_blocks.0.attn.to_qkv.weight", to_qkv_weight))

        # Add_kv_proj weights
        add_kv_proj_weight = torch.randn(18432, 6144)  # Full weights
        mock_weights.append(("transformer_blocks.0.attn.add_kv_proj.weight", add_kv_proj_weight))

        # 3. Test single block weight loading
        single_block_qkv_mlp_weight = torch.randn(18432 + 18432 * 3, 6144)  # Full weights
        mock_weights.append(("single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight", single_block_qkv_mlp_weight))

        # Execute weight loading
        loaded_params = model.load_weights(mock_weights)

        # Verify
        assert len(loaded_params) > 0, "Parameters should be loaded"

        # Verify stacked_params_mapping is correctly set
        assert model.stacked_params_mapping is not None
        # Should have 6 mappings: 3 for to_qkv + 3 for add_kv_proj
        assert len(model.stacked_params_mapping) == 6, "Should have 6 mappings"

        # Verify weights are correctly loaded to corresponding layers
        assert hasattr(model.transformer_blocks[0].attn.to_qkv, "weight")
        # With TP=2, weight dimension on each GPU should be 18432/2 = 9216
        assert model.transformer_blocks[0].attn.to_qkv.weight.shape == (9216, 6144), (
            f"With TP=2, to_qkv weight dimension should be (9216, 6144), got {model.transformer_blocks[0].attn.to_qkv.weight.shape}"
        )


class TestFlux2RopePositionEmbedding:
    """Test Flux2 RoPE position embedding functionality"""

    @pytest.mark.core_model
    @pytest.mark.cpu
    def test_rope_position_embedding(self):
        """Verify RoPE produces correct embeddings for 4D coordinates"""
        # Prepare test data - use model default configuration
        # axes_dims_rope default is (32, 32, 32, 32)
        # get_1d_rotary_pos_embed outputs half the dimension for real/imag parts
        # So actual output dimension should be (16+16+16+16) = 64
        axes_dims = (32, 32, 32, 32)  # Use model default
        rope_theta = 2000  # Model default is 2000
        pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims)

        # Create test IDs
        seq_len = 10
        ids = torch.randint(0, 100, (seq_len, 4))  # [S, 4]

        # Forward pass
        freqs_cos, freqs_sin = pos_embed(ids)

        # Verify output shape - based on model config, expected dimension is 64
        # Each axes_dim=32 outputs 16-dim real part, sum of 4 dimensions = 64
        expected_dim = sum(axes_dims) // 2  # 128/2 = 64
        assert freqs_cos.shape == (seq_len, expected_dim), (
            f"Expected shape {(seq_len, expected_dim)}, got {freqs_cos.shape}"
        )
        assert freqs_sin.shape == (seq_len, expected_dim), (
            f"Expected shape {(seq_len, expected_dim)}, got {freqs_sin.shape}"
        )

        # Verify output type - NPU may use float32, other devices use float64
        assert freqs_cos.dtype in [torch.float32, torch.float64], "Should be float type"
        assert freqs_sin.dtype in [torch.float32, torch.float64], "Should be float type"

        # Verify value range
        assert torch.all(freqs_cos >= -1) and torch.all(freqs_cos <= 1), "cos values should be in [-1, 1]"
        assert torch.all(freqs_sin >= -1) and torch.all(freqs_sin <= 1), "sin values should be in [-1, 1]"

        # Verify trigonometric relationship: cos^2 + sin^2 ≈ 1
        cos_sq_sin_sq = freqs_cos**2 + freqs_sin**2
        assert torch.allclose(cos_sq_sin_sq, torch.ones_like(cos_sq_sin_sq), atol=1e-6), "cos^2 + sin^2 should ≈ 1"

        # Verify different positions produce different embeddings
        ids_diff = torch.randint(0, 100, (seq_len, 4))
        freqs_cos_diff, freqs_sin_diff = pos_embed(ids_diff)
        assert not torch.allclose(freqs_cos, freqs_cos_diff), "Different positions should produce different embeddings"

        # Verify same positions produce same embeddings
        ids_same = ids.clone()
        freqs_cos_same, freqs_sin_same = pos_embed(ids_same)
        assert torch.allclose(freqs_cos, freqs_cos_same), "Same positions should produce same embeddings"
        assert torch.allclose(freqs_sin, freqs_sin_same), "Same positions should produce same embeddings"


class TestFlux2PackedModuleMapping:
    """Test Flux2 packed module mapping functionality"""

    @pytest.mark.core_model
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_packed_module_mapping(self, setup_tp_group):
        """Verify to_qkv packing matches HF checkpoint"""
        model = Flux2Transformer2DModel(
            num_layers=1,
            num_single_layers=0,
            num_attention_heads=48,
            attention_head_dim=128,
            joint_attention_dim=15360,
        )

        # Verify stacked_params_mapping is correctly initialized
        model.load_weights([])  # Trigger stacked_params_mapping initialization
        assert model.stacked_params_mapping is not None

        # Verify mapping configuration
        expected_mappings = [
            (".to_qkv.", ".to_q.", "q"),
            (".to_qkv.", ".to_k.", "k"),
            (".to_qkv.", ".to_v.", "v"),
            (".add_kv_proj", ".add_q_proj", "q"),
            (".add_kv_proj", ".add_k_proj", "k"),
            (".add_kv_proj", ".add_v_proj", "v"),
        ]
        assert model.stacked_params_mapping == expected_mappings

        # Create mock HF checkpoint weights
        hf_weights = []

        # Mock HF separated Q/K/V weights
        attn_block = model.transformer_blocks[0].attn
        head_dim = 128
        num_heads = 48
        hidden_size = 6144
        # Full weight dimension
        full_shard_size = num_heads * head_dim  # 6144

        # Q projection weights (full weights)
        q_weight = torch.randn(full_shard_size, hidden_size)
        hf_weights.append(("transformer_blocks.0.attn.to_q.weight", q_weight))

        # K projection weights (full weights)
        k_weight = torch.randn(full_shard_size, hidden_size)
        hf_weights.append(("transformer_blocks.0.attn.to_k.weight", k_weight))

        # V projection weights (full weights)
        v_weight = torch.randn(full_shard_size, hidden_size)
        hf_weights.append(("transformer_blocks.0.attn.to_v.weight", v_weight))

        # Mock HF separated add_q/k/v projection weights
        add_q_weight = torch.randn(full_shard_size, hidden_size)
        hf_weights.append(("transformer_blocks.0.attn.add_q_proj.weight", add_q_weight))

        add_k_weight = torch.randn(full_shard_size, hidden_size)
        hf_weights.append(("transformer_blocks.0.attn.add_k_proj.weight", add_k_weight))

        add_v_weight = torch.randn(full_shard_size, hidden_size)
        hf_weights.append(("transformer_blocks.0.attn.add_v_proj.weight", add_v_weight))

        # Execute weight loading
        loaded_params = model.load_weights(hf_weights)

        # Verify weights are loaded
        assert len(loaded_params) > 0

        # Verify final QKV weights are correctly combined (considering TP sharding)
        # With TP=2, dimension on each GPU should be half of full dimension
        expected_qkv_shape = (full_shard_size * 3 // 2, hidden_size)  # 18432/2 = 9216
        assert attn_block.to_qkv.weight.shape == expected_qkv_shape, (
            f"to_qkv weight dimension should be {expected_qkv_shape}, got {attn_block.to_qkv.weight.shape}"
        )

        expected_add_kv_shape = (full_shard_size * 3 // 2, hidden_size)
        assert attn_block.add_kv_proj.weight.shape == expected_add_kv_shape, (
            f"add_kv_proj weight dimension should be {expected_add_kv_shape}, got {attn_block.add_kv_proj.weight.shape}"
        )

    @pytest.mark.core_model
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_packed_mapping_edge_cases(self, setup_tp_group):
        """Test edge cases for packed mapping"""
        model = Flux2Transformer2DModel(
            num_layers=1,
            num_single_layers=1,
            num_attention_heads=48,
            attention_head_dim=128,
            joint_attention_dim=15360,
        )
        model.load_weights([])

        # Test invalid weight names
        invalid_weights = [("invalid.weight", torch.randn(10, 10))]
        loaded_params = model.load_weights(invalid_weights)
        assert len(loaded_params) == 0, "Should not load invalid weights"

        # Test to_out weight renaming
        to_out_weight = torch.randn(6144, 6144)
        weights = [("transformer_blocks.0.attn.to_out.0.weight", to_out_weight)]
        loaded_params = model.load_weights(weights)

        # Check if to_out related weights are loaded
        to_out_loaded = any("to_out" in p for p in loaded_params)
        assert to_out_loaded, "to_out weights should be correctly renamed and loaded"

        # Test partial weight loading
        partial_weights = [
            ("x_embedder.weight", torch.randn(6144, 128)),
            ("transformer_blocks.0.attn.to_q.weight", torch.randn(6144, 6144)),  # Full weights
        ]
        loaded_params = model.load_weights(partial_weights)
        assert len(loaded_params) == 2, "Should load two weights"
        assert "x_embedder.weight" in loaded_params
        assert any("to_qkv" in p for p in loaded_params), "to_q should be mapped to to_qkv"
