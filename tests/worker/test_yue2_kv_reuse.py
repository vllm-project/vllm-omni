# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Runner KV snapshot routing and copier reuse."""

from unittest.mock import Mock

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.kv_snapshot import PagedKVCopy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("layout", ["stacked", "packed"])
def test_runner_uses_device_block_table_and_reuses_copier(mocker, layout):
    length = 6
    runner = object.__new__(GPUARModelRunner)
    group = Mock(layer_names=["layer0", "layer1"])
    group.kv_cache_spec = FullAttentionSpec(block_size=4, num_kv_heads=2, head_size=3, dtype=torch.float32)
    runner.kv_cache_config = Mock(kv_cache_groups=[group])
    runner.cache_config = Mock(cache_dtype="auto")
    runner._kernel_block_sizes = [4]
    runner.requests = {"song": Mock(block_ids=([2, 0],))}
    runner.input_batch = Mock(req_id_to_index={"song": 1}, num_reqs=2)
    block_table = torch.tensor([[1, 1], [2, 0]], dtype=torch.int32)
    group_table = Mock()
    group_table.get_device_tensor.return_value = block_table
    runner.input_batch.block_table = [group_table]
    factory = mocker.patch("vllm_omni.worker.gpu_ar_model_runner.PagedKVCopy", wraps=PagedKVCopy)

    copy = mocker.patch.object(PagedKVCopy, "copy", autospec=True, return_value=True)
    contexts = {}
    for index, name in enumerate(group.layer_names):
        k = torch.arange(3 * 4 * 2 * 3, dtype=torch.float32).reshape(3, 4, 2, 3) + index * 100
        v = -k
        cache = torch.stack((k, v)) if layout == "stacked" else torch.cat((k, v), dim=-1).transpose(1, 2)
        contexts[name] = Mock(kv_cache=cache)
    runner.compilation_config = Mock(static_forward_context=contexts)
    keys = torch.full((2, 12, 2, 3), -999.0)
    values = torch.full_like(keys, -999.0)
    assert runner._copy_request_kv("song", group.layer_names, length, keys, values)
    assert copy.call_args.args[1] is block_table
    assert copy.call_args.args[2] == 1
    assert runner._copy_request_kv("song", group.layer_names, length, keys, values)
    assert factory.call_count == 1  # No new source-pointer upload per request.
    args = copy.call_args.args
    assert args[3] == length
    assert args[4] is keys and args[5] is values
    runner.cache_config.cache_dtype = "fp8"
    assert not runner._copy_request_kv("song", group.layer_names, length, keys, values)
    assert copy.call_count == 2  # Unsupported cache dtype never reaches the copier.
