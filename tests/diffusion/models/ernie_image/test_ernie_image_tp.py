# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.diffusion.models.ernie_image.ernie_image_transformer import (
    validate_ernie_image_tp_constraints,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestErnieImageTPConstraints:
    """Test ErnieImage tensor parallelism constraints."""

    @pytest.mark.parametrize(
        ("heads", "tensor_parallel_size", "expected_heads_per_gpu"),
        [(32, 1, 32), (32, 2, 16), (32, 4, 8), (32, 8, 4), (24, 2, 12), (24, 3, 8)],
    )
    def test_heads_divisible(self, heads, tensor_parallel_size, expected_heads_per_gpu):
        assert (
            validate_ernie_image_tp_constraints(heads=heads, tensor_parallel_size=tensor_parallel_size)
            == expected_heads_per_gpu
        )

    @pytest.mark.parametrize("tensor_parallel_size", [3, 5, 6])
    def test_heads_not_divisible(self, tensor_parallel_size):
        with pytest.raises(ValueError, match=r"num_attention_heads.*must be divisible by tensor_parallel_size"):
            validate_ernie_image_tp_constraints(heads=32, tensor_parallel_size=tensor_parallel_size)

    @pytest.mark.parametrize("tensor_parallel_size", [0, -1])
    def test_tp_must_be_positive(self, tensor_parallel_size):
        with pytest.raises(ValueError, match=r"tensor_parallel_size must be > 0"):
            validate_ernie_image_tp_constraints(heads=32, tensor_parallel_size=tensor_parallel_size)
