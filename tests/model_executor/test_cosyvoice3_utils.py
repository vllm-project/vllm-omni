# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CosyVoice3 utility functions."""

import pytest
import torch


class TestMakePadMask:
    """Tests for make_pad_mask utility."""

    def test_basic_mask_creation(self):
        """Test basic padding mask creation."""
        from vllm_omni.model_executor.models.cosyvoice3.utils import make_pad_mask

        lengths = torch.tensor([5, 3, 2])
        mask = make_pad_mask(lengths)

        # Expected: [[F,F,F,F,F], [F,F,F,T,T], [F,F,T,T,T]]
        expected = torch.tensor(
            [
                [False, False, False, False, False],
                [False, False, False, True, True],
                [False, False, True, True, True],
            ]
        )
        assert torch.equal(mask, expected)

    def test_single_sequence(self):
        """Test mask for single sequence."""
        from vllm_omni.model_executor.models.cosyvoice3.utils import make_pad_mask

        lengths = torch.tensor([3])
        mask = make_pad_mask(lengths)

        expected = torch.tensor([[False, False, False]])
        assert torch.equal(mask, expected)

    def test_explicit_max_len(self):
        """Test with explicit max_len parameter."""
        from vllm_omni.model_executor.models.cosyvoice3.utils import make_pad_mask

        lengths = torch.tensor([2, 3])
        mask = make_pad_mask(lengths, max_len=5)

        expected = torch.tensor(
            [
                [False, False, True, True, True],
                [False, False, False, True, True],
            ]
        )
        assert torch.equal(mask, expected)

    def test_zero_max_len_uses_max_length(self):
        """Test that max_len=0 falls back to max(lengths)."""
        from vllm_omni.model_executor.models.cosyvoice3.utils import make_pad_mask

        lengths = torch.tensor([2, 4])
        mask = make_pad_mask(lengths, max_len=0)

        # Should use max(lengths) = 4
        assert mask.shape == (2, 4)

    def test_full_length_sequence(self):
        """Test sequence that fills the entire length (no padding)."""
        from vllm_omni.model_executor.models.cosyvoice3.utils import make_pad_mask

        lengths = torch.tensor([5])
        mask = make_pad_mask(lengths, max_len=5)

        # No positions should be masked
        assert not mask.any()

    def test_empty_sequence(self):
        """Test sequence with zero length (all padding)."""
        from vllm_omni.model_executor.models.cosyvoice3.utils import make_pad_mask

        lengths = torch.tensor([0])
        mask = make_pad_mask(lengths, max_len=3)

        # All positions should be masked
        assert mask.all()

    def test_device_preservation(self):
        """Test that mask is created on the same device as input."""
        from vllm_omni.model_executor.models.cosyvoice3.utils import make_pad_mask

        lengths = torch.tensor([3, 2])
        mask = make_pad_mask(lengths)

        assert mask.device == lengths.device


class TestDynamicRangeCompression:
    """Tests for dynamic_range_compression_torch utility."""

    def test_basic_compression(self):
        """Test basic compression with default parameters."""
        from vllm_omni.model_executor.models.cosyvoice3.utils import dynamic_range_compression_torch

        x = torch.tensor([1.0, 10.0, 100.0])
        result = dynamic_range_compression_torch(x)

        # Result should be log(x * 1) = log(x)
        expected = torch.log(x)
        assert torch.allclose(result, expected)

    def test_clipping_small_values(self):
        """Test that small values are clipped."""
        from vllm_omni.model_executor.models.cosyvoice3.utils import dynamic_range_compression_torch

        x = torch.tensor([0.0, 1e-10, 1e-6])
        clip_val = 1e-5
        result = dynamic_range_compression_torch(x, clip_val=clip_val)

        # All values below clip_val should be clipped
        expected = torch.log(torch.tensor([clip_val, clip_val, clip_val]))
        assert torch.allclose(result, expected)

    def test_scaling_factor(self):
        """Test with different scaling factor c."""
        from vllm_omni.model_executor.models.cosyvoice3.utils import dynamic_range_compression_torch

        x = torch.tensor([1.0, 10.0])
        c = 2.0
        result = dynamic_range_compression_torch(x, c=c)

        expected = torch.log(x * c)
        assert torch.allclose(result, expected)


class TestSpectralNormalize:
    """Tests for spectral_normalize_torch utility."""

    def test_is_wrapper_for_compression(self):
        """Test that spectral_normalize is a wrapper for dynamic_range_compression."""
        from vllm_omni.model_executor.models.cosyvoice3.utils import (
            dynamic_range_compression_torch,
            spectral_normalize_torch,
        )

        magnitudes = torch.tensor([1.0, 10.0, 100.0])
        result = spectral_normalize_torch(magnitudes)
        expected = dynamic_range_compression_torch(magnitudes)

        assert torch.allclose(result, expected)


class TestExactDiv:
    """Tests for exact_div utility."""

    def test_exact_division(self):
        """Test exact division with valid inputs."""
        from vllm_omni.model_executor.models.cosyvoice3.utils import exact_div

        assert exact_div(10, 2) == 5
        assert exact_div(100, 25) == 4
        assert exact_div(0, 5) == 0

    def test_non_exact_division_raises(self):
        """Test that non-exact division raises AssertionError."""
        from vllm_omni.model_executor.models.cosyvoice3.utils import exact_div

        with pytest.raises(AssertionError):
            exact_div(10, 3)

        with pytest.raises(AssertionError):
            exact_div(7, 2)


class TestConcatTextWithPromptIds:
    """Tests for concat_text_with_prompt_ids utility."""

    def test_basic_concat(self):
        """Test basic concatenation."""
        from vllm_omni.model_executor.models.cosyvoice3.utils import concat_text_with_prompt_ids

        text = torch.tensor([[5, 6, 7]], dtype=torch.int64)
        text_len = torch.tensor([3], dtype=torch.int32)
        prompt_text = torch.tensor([[1, 2]], dtype=torch.int64)
        prompt_text_len = torch.tensor([2], dtype=torch.int32)

        result, result_len = concat_text_with_prompt_ids(text, text_len, prompt_text, prompt_text_len)

        assert result.tolist() == [[1, 2, 5, 6, 7]]
        assert result_len.item() == 5

    def test_empty_prompt(self):
        """Test with empty prompt."""
        from vllm_omni.model_executor.models.cosyvoice3.utils import concat_text_with_prompt_ids

        text = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        text_len = torch.tensor([3], dtype=torch.int32)
        prompt_text = torch.tensor([[]], dtype=torch.int64).reshape(1, 0)
        prompt_text_len = torch.tensor([0], dtype=torch.int32)

        result, result_len = concat_text_with_prompt_ids(text, text_len, prompt_text, prompt_text_len)

        assert result.tolist() == [[1, 2, 3]]
        assert result_len.item() == 3

    def test_batch_concat(self):
        """Test batched concatenation."""
        from vllm_omni.model_executor.models.cosyvoice3.utils import concat_text_with_prompt_ids

        text = torch.tensor([[5, 6], [7, 8]], dtype=torch.int64)
        text_len = torch.tensor([2, 2], dtype=torch.int32)
        prompt_text = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        prompt_text_len = torch.tensor([2, 2], dtype=torch.int32)

        result, result_len = concat_text_with_prompt_ids(text, text_len, prompt_text, prompt_text_len)

        assert result.tolist() == [[1, 2, 5, 6], [3, 4, 7, 8]]
        assert result_len.tolist() == [4, 4]

    def test_prompt_comes_first(self):
        """Test that prompt tokens come before text tokens."""
        from vllm_omni.model_executor.models.cosyvoice3.utils import concat_text_with_prompt_ids

        text = torch.tensor([[100, 200]], dtype=torch.int64)
        text_len = torch.tensor([2], dtype=torch.int32)
        prompt_text = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        prompt_text_len = torch.tensor([3], dtype=torch.int32)

        result, _ = concat_text_with_prompt_ids(text, text_len, prompt_text, prompt_text_len)

        # First 3 tokens should be prompt
        assert result[0, :3].tolist() == [1, 2, 3]
        # Last 2 tokens should be text
        assert result[0, 3:].tolist() == [100, 200]
