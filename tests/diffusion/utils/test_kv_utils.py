# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for kv utils."""

import pytest
import torch

from vllm_omni.diffusion.utils.kv_utils import left_pad_stack

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_uniform_lengths():
    """Ensure uniform 3D tensors correctly stack to 4D and have no mask."""
    tensors = [torch.randn(10, 4, 128) for _ in range(3)]
    stacked, mask = left_pad_stack(tensors)
    assert stacked.shape == (3, 10, 4, 128)
    assert mask is None
    for i in range(3):
        assert torch.equal(stacked[i], tensors[i])


def test_variable_lengths():
    """Ensure variable 3D tensors correctly stack to 4D with a mask."""
    t1 = torch.ones(5, 2, 4)
    t2 = torch.ones(8, 2, 4)
    t3 = torch.ones(3, 2, 4)
    stacked, mask = left_pad_stack([t1, t2, t3])

    assert stacked.shape == (3, 8, 2, 4)
    assert mask is not None
    assert mask.shape == (3, 8)
    # Ensure summing over dim 1 gives our seq lens back
    assert mask.sum(dim=1).tolist() == [5, 8, 3]


def test_single_tensor_is_4d():
    """Ensure a single 3D tensor is expanded to 4D."""
    t = torch.randn(7, 4, 128)
    stacked, mask = left_pad_stack([t])
    assert stacked.shape == (1, 7, 4, 128)
    assert mask is None
    assert torch.equal(stacked[0], t)


def test_preserves_device_and_dtype():
    """Ensure device/dtype is preserved."""
    t1 = torch.randn(3, 2, dtype=torch.bfloat16)
    t2 = torch.randn(5, 2, dtype=torch.bfloat16)
    stacked, mask = left_pad_stack([t1, t2])
    assert stacked.dtype == torch.bfloat16
    assert stacked.device == t1.device
    assert mask.device == t1.device


def test_mismatched_trailing_shapes_raises():
    """Ensure that mismatched dims outside of 0 explodes."""
    t1 = torch.randn(5, 4, 128)
    t2 = torch.randn(5, 8, 128)
    with pytest.raises(ValueError):
        left_pad_stack([t1, t2])


def test_mismatched_ndim_raises():
    """Ensure that mismatched ndims explodes."""
    t1 = torch.randn(5, 4)
    t2 = torch.randn(5, 4, 128)
    with pytest.raises(ValueError):
        left_pad_stack([t1, t2])


def test_empty_list_raises():
    """Ensure that tensors must be nonempty."""
    with pytest.raises(ValueError):
        left_pad_stack([])
