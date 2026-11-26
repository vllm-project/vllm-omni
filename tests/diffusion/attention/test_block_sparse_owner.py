# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Owner identity/lifetime tests with a CPU executor; no FA4 dependency."""

import copy
import gc
import io
import weakref

import pytest
import torch

from vllm_omni.diffusion.attention.block_sparse import (
    BlockSparseAttention,
    _block_sparse_request,
    _request_owners,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


class CPUOwner(BlockSparseAttention):
    """Exercise the production handle/state protocol without initializing CUDA."""

    def __init__(self, bias):
        self.bias = bias
        self._request_preparation = {}
        self._register_request_owner()

    def _dispatch_request(self, query, key, value, prefix):
        return query + self.bias


def dispatch(owner):
    q = torch.zeros(1, 2, 1, 1)
    return _block_sparse_request(q, q, q, 0, owner._request_owner_handle)


@pytest.mark.parametrize("clone_method", ["deepcopy", "torch_save"])
def test_copied_owner_is_independent_and_survives_original_collection(clone_method):
    original = CPUOwner(1)
    original._request_preparation[("prepared",)] = None
    original._request_preparation[("failed",)] = "previous failure"
    original_id = original._request_owner_handle.item()
    original_ref = weakref.ref(original)
    stale_handle = original._request_owner_handle.clone()
    if clone_method == "deepcopy":
        cloned = copy.deepcopy(original)
    else:
        buffer = io.BytesIO()
        torch.save(original, buffer)
        buffer.seek(0)
        # Only deserialize the object created in this test, never external data.
        cloned = torch.load(buffer, weights_only=False)
    clone_id = cloned._request_owner_handle.item()
    assert clone_id != original_id and not cloned._request_preparation
    assert _request_owners[original_id] is original and _request_owners[clone_id] is cloned
    cloned.bias = 7
    cloned._request_preparation[("clone only",)] = None
    assert ("clone only",) not in original._request_preparation
    torch.testing.assert_close(dispatch(original), torch.ones(1, 2, 1, 1))
    torch.testing.assert_close(dispatch(cloned), torch.full((1, 2, 1, 1), 7.0))
    del original
    gc.collect()
    assert original_ref() is None and original_id not in _request_owners
    torch.testing.assert_close(dispatch(cloned), torch.full((1, 2, 1, 1), 7.0))
    q = torch.zeros(1, 2, 1, 1)
    with pytest.raises(RuntimeError, match="owner was released"):
        _block_sparse_request(q, q, q, 0, stale_handle)
    clone_ref = weakref.ref(cloned)
    del cloned
    gc.collect()
    assert clone_ref() is None and clone_id not in _request_owners


def test_shallow_copy_rejected_without_changing_owner():
    owner = CPUOwner(3)
    handle = owner._request_owner_handle
    with pytest.raises(TypeError, match="copy.deepcopy"):
        copy.copy(owner)
    assert owner._request_owner_handle is handle
    torch.testing.assert_close(dispatch(owner), torch.full((1, 2, 1, 1), 3.0))


def test_deepcopy_preserves_aliases_within_copied_model():
    original = CPUOwner(1)
    first, second = copy.deepcopy([original, original])
    assert first is second and first is not original
    assert first._request_owner_handle.item() != original._request_owner_handle.item()
