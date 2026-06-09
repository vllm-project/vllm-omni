# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for payload_builder shared utilities."""

import torch

from vllm_omni.model_executor.stage_input_processors.payload_builder import (
    count_trailing_placeholders,
    ensure_list,
    filter_invalid_tokens,
    layer_tensor,
    strip_boundary_tokens,
    to_cpu_tensor,
    to_tensor_or_none,
)


def test_ensure_list():
    """Test ensure_list helper function."""
    # Test with list
    assert ensure_list([1, 2, 3]) == [1, 2, 3]

    # Test with tuple
    assert ensure_list((1, 2, 3)) == [1, 2, 3]

    # Test with None
    assert ensure_list(None) == []

    # Test with single value
    assert ensure_list(5) == [5]

    # Test with ConstantList-like object
    class ConstantList:
        def __init__(self, x):
            self._x = x

    assert ensure_list(ConstantList([1, 2, 3])) == [1, 2, 3]


def test_layer_tensor():
    """Test layer_tensor helper function."""
    # Test with int key
    layers = {0: torch.tensor([1, 2, 3]), 24: torch.tensor([4, 5, 6])}
    assert torch.equal(layer_tensor(layers, 0), torch.tensor([1, 2, 3]))
    assert torch.equal(layer_tensor(layers, 24), torch.tensor([4, 5, 6]))

    # Test with str key
    layers_str = {"0": torch.tensor([1, 2, 3]), "24": torch.tensor([4, 5, 6])}
    assert torch.equal(layer_tensor(layers_str, "0"), torch.tensor([1, 2, 3]))

    # Test with missing key
    assert layer_tensor(layers, 1) is None

    # Test with non-dict input
    assert layer_tensor(None, 0) is None


def test_to_cpu_tensor():
    """Test to_cpu_tensor helper function."""
    # Test with GPU tensor (if CUDA available)
    if torch.cuda.is_available():
        gpu_tensor = torch.tensor([1, 2, 3]).cuda()
        cpu_tensor = to_cpu_tensor(gpu_tensor)
        assert cpu_tensor is not None
        assert not cpu_tensor.is_cuda
        assert torch.equal(cpu_tensor, torch.tensor([1, 2, 3]))

    # Test with CPU tensor
    cpu_tensor = torch.tensor([1, 2, 3])
    result = to_cpu_tensor(cpu_tensor)
    assert result is not None
    assert not result.is_cuda
    assert torch.equal(result, torch.tensor([1, 2, 3]))

    # Test with list of tensors
    tensor_list = [torch.tensor([1, 2, 3])]
    result = to_cpu_tensor(tensor_list)
    assert result is not None
    assert torch.equal(result, torch.tensor([1, 2, 3]))

    # Test with None
    assert to_cpu_tensor(None) is None

    # Test with empty list
    assert to_cpu_tensor([]) is None


def test_to_tensor_or_none():
    """Test to_tensor_or_none helper function."""
    # Test with tensor
    tensor = torch.tensor([1, 2, 3])
    result = to_tensor_or_none(tensor)
    assert result is not None
    assert torch.equal(result, torch.tensor([1, 2, 3]))

    # Test with list of tensors
    tensor_list = [torch.tensor([1, 2, 3])]
    result = to_tensor_or_none(tensor_list)
    assert result is not None
    assert torch.equal(result, torch.tensor([1, 2, 3]))

    # Test with None
    assert to_tensor_or_none(None) is None

    # Test with empty list
    assert to_tensor_or_none([]) is None


def test_strip_boundary_tokens():
    """Test strip_boundary_tokens helper function."""
    # Test with start token
    tokens = [100, 1, 2, 3]
    result = strip_boundary_tokens(tokens, start_token_id=100)
    assert result == [1, 2, 3]

    # Test with end token
    tokens = [1, 2, 3, 200]
    result = strip_boundary_tokens(tokens, end_token_id=200)
    assert result == [1, 2, 3]

    # Test with pad token
    tokens = [1, 2, -1, 3, -1]
    result = strip_boundary_tokens(tokens, pad_token_id=-1)
    assert result == [1, 2, 3]

    # Test with all boundary tokens
    tokens = [100, 1, 2, -1, 3, 200]
    result = strip_boundary_tokens(
        tokens,
        start_token_id=100,
        pad_token_id=-1,
        end_token_id=200,
    )
    assert result == [1, 2, 3]

    # Test with no boundary tokens
    tokens = [1, 2, 3]
    result = strip_boundary_tokens(tokens, start_token_id=100, end_token_id=200)
    assert result == [1, 2, 3]


def test_filter_invalid_tokens():
    """Test filter_invalid_tokens helper function."""
    # Test with min_valid
    tokens = [-1, 0, 1, 2, -1]
    result = filter_invalid_tokens(tokens, min_valid=0)
    assert result == [0, 1, 2]

    # Test with max_valid
    tokens = [0, 1, 2, 1000, 3]
    result = filter_invalid_tokens(tokens, max_valid=100)
    assert result == [0, 1, 2, 3]

    # Test with both min and max
    tokens = [-1, 0, 1, 2, 1000, 3]
    result = filter_invalid_tokens(tokens, min_valid=0, max_valid=100)
    assert result == [0, 1, 2, 3]

    # Test with no invalid tokens
    tokens = [0, 1, 2, 3]
    result = filter_invalid_tokens(tokens, min_valid=0, max_valid=100)
    assert result == [0, 1, 2, 3]


def test_count_trailing_placeholders():
    """Test count_trailing_placeholders helper function."""
    # Test with trailing placeholders
    tokens = [1, 2, 3, -1, -1]
    result = count_trailing_placeholders(tokens, placeholder=-1)
    assert result == 2

    # Test with no trailing placeholders
    tokens = [1, 2, 3, -1, 4]
    result = count_trailing_placeholders(tokens, placeholder=-1)
    assert result == 0

    # Test with all placeholders
    tokens = [-1, -1, -1]
    result = count_trailing_placeholders(tokens, placeholder=-1)
    assert result == 3

    # Test with empty list
    tokens = []
    result = count_trailing_placeholders(tokens, placeholder=-1)
    assert result == 0

    # Test with different placeholder value
    tokens = [1, 2, 3, 0, 0]
    result = count_trailing_placeholders(tokens, placeholder=0)
    assert result == 2
