# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Public cuDNN ABI and sparse metadata contracts, without a CUDA dependency."""

import sys
from types import ModuleType

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

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
        return {
            "o_tensor": query.clone(),
            "lse_tensor": torch.zeros(query.shape[0], query.shape[2], query.shape[1], dtype=torch.float32),
        }

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


@pytest.mark.parametrize("frontend_version", ["1.31.0", "1.32.0"])
def test_cudnn_loads_public_bsa_namespace_without_torch_alias(monkeypatch, frontend_version):
    public_api = ModuleType("cudnn.block_sparse_attention")

    def kernel(*args, **kwargs):
        pytest.fail("dependency preflight must not launch the kernel")

    monkeypatch.setattr(public_api, "block_sparse_attention_forward", kernel, raising=False)
    monkeypatch.setitem(sys.modules, "cudnn.block_sparse_attention", public_api)
    monkeypatch.setitem(sys.modules, "cudnn.torch", None)
    monkeypatch.setattr(cudnn_block_sparse, "version", lambda name: frontend_version)
    monkeypatch.setattr(cudnn_block_sparse.current_omni_platform, "get_device_capability", lambda index: (12, 0))
    cudnn_block_sparse.require_cudnn_sparse.cache_clear()
    try:
        assert cudnn_block_sparse.require_cudnn_sparse(torch.device("cuda:3")) is kernel
    finally:
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


@pytest.mark.parametrize("compiled", [False, True])
def test_cudnn_lse_preserves_natural_log_values_and_layout(monkeypatch, compiled):
    query = torch.ones(2, 129, 3, 128, dtype=torch.bfloat16)
    expected_lse = torch.arange(2 * 3 * 129, dtype=torch.float32).view(2, 129, 3).transpose(1, 2)

    def kernel(*args, **kwargs):
        return {"o_tensor": query.clone(), "lse_tensor": expected_lse}

    monkeypatch.setattr(cudnn_block_sparse, "require_cudnn_sparse", lambda device: kernel)
    forward = (
        torch.compile(block_sparse_attn_bshd, backend="aot_eager", fullgraph=True)
        if compiled
        else block_sparse_attn_bshd
    )
    output, lse = forward(
        query,
        query,
        query,
        torch.ones(2, 3, 3, 3, dtype=torch.bool),
        None,
        0.125,
        provider="cudnn",
        return_lse=True,
    )
    torch.testing.assert_close(output, query)
    torch.testing.assert_close(lse, expected_lse, rtol=0, atol=0)
    assert lse.is_contiguous()


@pytest.mark.parametrize("bad_lse", [torch.zeros(1, 64, 2), torch.zeros(1, 2, 64, dtype=torch.bfloat16)])
def test_cudnn_rejects_incompatible_lse_abi(monkeypatch, bad_lse):
    query = torch.ones(1, 64, 2, 128, dtype=torch.bfloat16)
    monkeypatch.setattr(
        cudnn_block_sparse,
        "require_cudnn_sparse",
        lambda device: lambda *args, **kwargs: {"o_tensor": query.clone(), "lse_tensor": bad_lse},
    )
    with pytest.raises(ValueError, match="FP32 natural-log LSE"):
        block_sparse_attn_bshd(
            query,
            query,
            query,
            torch.ones(1, 2, 1, 1, dtype=torch.bool),
            None,
            0.125,
            provider="cudnn",
            return_lse=True,
        )


def test_cudnn_fake_lse_matches_real_output_contract(monkeypatch):
    def unexpected_launch(device):
        pytest.fail("fake dispatch must not load a CUDA kernel")

    monkeypatch.setattr(cudnn_block_sparse, "require_cudnn_sparse", unexpected_launch)
    with FakeTensorMode():
        query = torch.empty(2, 129, 3, 128, dtype=torch.bfloat16, device="cuda")
        block_map = torch.ones(2, 3, 3, 3, dtype=torch.bool, device="cuda")
        output, lse = block_sparse_attn_bshd(
            query,
            query,
            query,
            block_map,
            None,
            0.125,
            provider="cudnn",
            return_lse=True,
        )
        assert output.shape == query.shape and output.dtype == query.dtype
        assert lse.shape == (2, 3, 129) and lse.dtype == torch.float32
        assert output.device == lse.device == query.device
