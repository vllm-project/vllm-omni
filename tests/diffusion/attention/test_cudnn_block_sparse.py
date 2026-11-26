# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Public cuDNN ABI and sparse metadata contracts, without a CUDA dependency."""

import pytest
import torch

from vllm_omni.diffusion.attention.ops import cudnn_block_sparse
from vllm_omni.diffusion.attention.ops.block_sparse import block_sparse_attn_bshd
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("block_size", [64, 128])
def test_cudnn_preserves_selected_blocks_and_excludes_empty_keys(monkeypatch, block_size):
    q = torch.randn(1, block_size * 3, 2, 128, dtype=torch.bfloat16)
    mask = torch.tensor([[[[True, True, True], [False, False, False], [True, False, True]]]]).expand(1, 2, 3, 3)
    sizes = torch.tensor([block_size, 0, 17], dtype=torch.int32)
    calls = []

    def kernel(query, key, value, indices, **kwargs):
        calls.append(kwargs)
        assert indices[0, 0].tolist() == [[0, 2, -1], [-1, -1, -1], [0, 2, -1]]
        assert kwargs["q2k_block_nums"][0, 0].tolist() == [2, 0, 2]
        assert kwargs["sparse_block_size"] == block_size
        assert kwargs["allow_empty_block_nums"] is True
        assert kwargs["layout"] == "bshd"
        assert kwargs["softmax_scale"] == 0.125
        assert torch.equal(kwargs["block_sizes"], sizes)
        return {"o_tensor": query.clone()}

    monkeypatch.setattr(cudnn_block_sparse, "require_cudnn_sparse", lambda device: kernel)
    result = block_sparse_attn_bshd(q, q, q, mask, sizes, 0.125, provider="cudnn", block_size=block_size)
    torch.testing.assert_close(result, q)
    assert len(calls) == 1
    assert mask[0, 0, 0, 1], "provider conversion must not mutate the model's routing mask"


def test_cudnn_selection_round_trips_existing_attention_config():
    config = AttentionConfig(default=AttentionSpec(backend="FASTVIDEO_VSA", fastvideo_vsa_provider="cudnn"))
    assert config.default.backend_kwargs()["provider"] == "cudnn"
    with pytest.raises(ValueError, match="FlashInfer"):
        AttentionSpec(backend="FASTVIDEO_VSA", fastvideo_vsa_provider="cudnn", fastvideo_vsa_precision="sage")


def test_cudnn_missing_dependency_is_actionable(monkeypatch):
    cudnn_block_sparse.require_cudnn_sparse.cache_clear()
    monkeypatch.setattr(cudnn_block_sparse.current_omni_platform, "get_device_capability", lambda index: (12, 0))
    monkeypatch.setattr(cudnn_block_sparse, "version", lambda name: "1.26.0")
    with pytest.raises(ImportError, match=r"vllm-omni\[cudnn-bsa\]"):
        cudnn_block_sparse.require_cudnn_sparse(torch.device("cuda:3"))
    cudnn_block_sparse.require_cudnn_sparse.cache_clear()


def test_cudnn_kernel_failure_is_propagated(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("kernel failure")

    monkeypatch.setattr(cudnn_block_sparse, "require_cudnn_sparse", lambda device: fail)
    q = torch.ones(1, 128, 1, 128, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="kernel failure"):
        block_sparse_attn_bshd(
            q,
            q,
            q,
            torch.ones(1, 1, 1, 1, dtype=torch.bool),
            None,
            0.125,
            provider="cudnn",
            block_size=128,
        )
