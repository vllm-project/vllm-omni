import pytest

from vllm_omni.diffusion.models.ernie_image.ernie_image_transformer import (
    validate_ernie_image_tp_constraints,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestErnieImageTPConstraints:
    """Test ErnieImage tensor parallelism constraints"""

    def test_tp1_heads_divisible(self):
        """Verify TP=1 works with default heads=32"""
        heads_per_gpu = validate_ernie_image_tp_constraints(heads=32, tensor_parallel_size=1)
        assert heads_per_gpu == 32

    def test_tp2_heads_divisible(self):
        """Verify TP=2 works with heads=32"""
        heads_per_gpu = validate_ernie_image_tp_constraints(heads=32, tensor_parallel_size=2)
        assert heads_per_gpu == 16

    def test_tp4_heads_divisible(self):
        """Verify TP=4 works with heads=32"""
        heads_per_gpu = validate_ernie_image_tp_constraints(heads=32, tensor_parallel_size=4)
        assert heads_per_gpu == 8

    def test_tp8_heads_divisible(self):
        """Verify TP=8 works with heads=32"""
        heads_per_gpu = validate_ernie_image_tp_constraints(heads=32, tensor_parallel_size=8)
        assert heads_per_gpu == 4

    def test_tp3_heads_not_divisible(self):
        """Verify TP=3 fails when heads=32 not divisible by 3"""
        with pytest.raises(ValueError, match=r"num_attention_heads.*must be divisible by tensor_parallel_size"):
            validate_ernie_image_tp_constraints(heads=32, tensor_parallel_size=3)

    def test_tp5_heads_not_divisible(self):
        """Verify TP=5 fails when heads=32 not divisible by 5"""
        with pytest.raises(ValueError, match=r"num_attention_heads.*must be divisible by tensor_parallel_size"):
            validate_ernie_image_tp_constraints(heads=32, tensor_parallel_size=5)

    def test_tp6_heads_not_divisible(self):
        """Verify TP=6 fails when heads=32 not divisible by 6"""
        with pytest.raises(ValueError, match=r"num_attention_heads.*must be divisible by tensor_parallel_size"):
            validate_ernie_image_tp_constraints(heads=32, tensor_parallel_size=6)

    def test_custom_heads_tp2_divisible(self):
        """Verify TP=2 works with custom heads=24"""
        heads_per_gpu = validate_ernie_image_tp_constraints(heads=24, tensor_parallel_size=2)
        assert heads_per_gpu == 12

    def test_custom_heads_tp3_divisible(self):
        """Verify TP=3 works with custom heads=24"""
        heads_per_gpu = validate_ernie_image_tp_constraints(heads=24, tensor_parallel_size=3)
        assert heads_per_gpu == 8

    def test_tp_invalid_zero(self):
        """Verify TP=0 fails"""
        with pytest.raises(ValueError, match=r"tensor_parallel_size must be > 0"):
            validate_ernie_image_tp_constraints(heads=32, tensor_parallel_size=0)

    def test_tp_invalid_negative(self):
        """Verify negative TP fails"""
        with pytest.raises(ValueError, match=r"tensor_parallel_size must be > 0"):
            validate_ernie_image_tp_constraints(heads=32, tensor_parallel_size=-1)
