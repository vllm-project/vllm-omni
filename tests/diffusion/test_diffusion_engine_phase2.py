"""
Unit tests for Phase 2 fixes:
- Recursive CPU offload (_move_tensors_to_cpu)
- Batched audio output slicing (_split_batched_output)
"""

import numpy as np
import torch

from vllm_omni.diffusion.diffusion_engine import (
    _move_tensors_to_cpu,
    _split_batched_output,
)


class TestMoveTensorsToCpu:
    def test_single_tensor_on_cpu(self):
        t = torch.randn(4)
        result = _move_tensors_to_cpu(t)
        assert result.device.type == "cpu"

    def test_single_tensor_already_cpu(self):
        t = torch.randn(4)  # already cpu
        result = _move_tensors_to_cpu(t)
        assert result is t  # no-op, same object
        assert result.device.type == "cpu"

    def test_dict_with_tensors(self):
        data = {"video": torch.randn(2, 3), "score": torch.tensor(0.9)}
        result = _move_tensors_to_cpu(data)
        for v in result.values():
            assert v.device.type == "cpu"

    def test_tuple_with_tensors(self):
        data = (torch.randn(3), torch.randn(5))
        result = _move_tensors_to_cpu(data)
        assert isinstance(result, tuple)
        for v in result:
            assert v.device.type == "cpu"

    def test_list_with_tensors(self):
        data = [torch.randn(3), torch.randn(5)]
        result = _move_tensors_to_cpu(data)
        assert isinstance(result, list)
        for v in result:
            assert v.device.type == "cpu"

    def test_nested_structure(self):
        data = {"outputs": (torch.randn(2), {"extra": torch.randn(1)})}
        result = _move_tensors_to_cpu(data)
        assert result["outputs"][0].device.type == "cpu"
        assert result["outputs"][1]["extra"].device.type == "cpu"

    def test_non_tensor_passthrough(self):
        data = {"label": "hello", "count": 42}
        result = _move_tensors_to_cpu(data)
        assert result == data


class TestSplitBatchedOutput:
    def test_batched_torch_tensor_split(self):
        batch = torch.randn(3, 16000)  # 3 requests
        results = _split_batched_output(batch, n=3)
        assert len(results) == 3
        for r in results:
            assert r.shape == (16000,)

    def test_single_sample_not_split(self):
        single = torch.randn(1, 16000)
        results = _split_batched_output(single, n=1)
        assert len(results) == 1

    def test_numpy_batched_split(self):
        batch = np.random.randn(2, 8000).astype(np.float32)
        results = _split_batched_output(batch, n=2)
        assert len(results) == 2
        assert results[0].shape == (8000,)

    def test_list_passthrough(self):
        lst = [torch.randn(16000), torch.randn(16000)]
        results = _split_batched_output(lst, n=2)
        assert results == lst

    def test_none_returns_empty(self):
        results = _split_batched_output(None, n=1)
        assert results == []

    def test_mismatched_batch_size(self):
        batch = torch.randn(5, 16000)
        results = _split_batched_output(batch, n=3)
        # batch dim != n, so treated as single output
        assert len(results) == 1
        assert results[0] is batch
