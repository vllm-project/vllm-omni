# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the shared rotate_half function."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class TestRotateHalf:
    """Verify rotate_half produces correct results for both variants."""

    def test_non_interleaved_basic(self) -> None:
        """Non-interleaved: swaps and negates halves."""
        from vllm_omni.diffusion.layers.rope import rotate_half

        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = rotate_half(x)
        expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
        torch.testing.assert_close(result, expected)

    def test_interleaved_basic(self) -> None:
        """Interleaved: negates odd elements, swaps with even."""
        from vllm_omni.diffusion.layers.rope import rotate_half

        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = rotate_half(x, interleaved=True)
        expected = torch.tensor([-2.0, 1.0, -4.0, 3.0])
        torch.testing.assert_close(result, expected)

    def test_non_interleaved_involution(self) -> None:
        """Applying rotate_half twice negates the input."""
        from vllm_omni.diffusion.layers.rope import rotate_half

        torch.manual_seed(42)
        x = torch.randn(2, 8, 4, 64)
        result = rotate_half(rotate_half(x))
        torch.testing.assert_close(result, -x)

    def test_interleaved_involution(self) -> None:
        """Applying rotate_half(interleaved=True) twice negates the input."""
        from vllm_omni.diffusion.layers.rope import rotate_half

        torch.manual_seed(42)
        x = torch.randn(2, 8, 4, 64)
        result = rotate_half(rotate_half(x, interleaved=True), interleaved=True)
        torch.testing.assert_close(result, -x)

    @pytest.mark.parametrize(
        "shape",
        [
            (4,),
            (8, 64),
            (2, 16, 8, 64),
            (1, 8192, 4, 128),
        ],
    )
    def test_output_shape_preserved(self, shape: tuple[int, ...]) -> None:
        from vllm_omni.diffusion.layers.rope import rotate_half

        x = torch.randn(shape)
        assert rotate_half(x).shape == shape
        assert rotate_half(x, interleaved=True).shape == shape

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_dtype_preserved(self, dtype: torch.dtype) -> None:
        from vllm_omni.diffusion.layers.rope import rotate_half

        x = torch.randn(2, 8, dtype=dtype)
        assert rotate_half(x).dtype == dtype
