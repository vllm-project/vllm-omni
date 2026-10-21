# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA-only GPUARModelRunner tests, split from the CPU-marked test module."""

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.core.prefix_cache import ModelCachePolicy
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

pytestmark = [pytest.mark.core_model]


def _mtp_runner(buffers: dict[str, dict]) -> GPUARModelRunner:
    runner = object.__new__(GPUARModelRunner)
    runner._omni_cache_policy = ModelCachePolicy(needs_full_hidden_states=True)
    runner._pooler_payload_include_hidden_flag = True
    runner.use_async_scheduling = False
    runner.requests = {req_id: SimpleNamespace() for req_id in buffers}
    runner.model_intermediate_buffer = buffers
    return runner


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_post_sample_talker_mtp_moves_device_codes_with_one_pending_copy() -> None:
    runner = _mtp_runner({"a": {"duplex": {"data_plane": True}}, "b": {"duplex": {"data_plane": True}}})
    runner.model = SimpleNamespace(
        post_sample_talker_mtp=lambda *, input_ids, **_: input_ids[:, None].expand(-1, 3) * 10
    )
    multimodal = GPUARModelRunner._run_post_sample_talker_mtp(
        runner,
        req_ids=["a", "b"],
        valid_sampled_token_ids=[[1], [2]],
        sampled_token_ids=torch.tensor([[1], [2]], dtype=torch.long, device="cuda"),
        invalid_req_indices=[],
        sample_hidden_states=torch.zeros((2, 2), device="cuda"),
        multimodal_outputs=None,
    )
    runner._omni_post_sample_host_copies.wait()
    audio = multimodal["codes"]["audio"]
    from vllm.utils.platform_utils import is_pin_memory_available

    assert all(row.device.type == "cpu" and row.is_pinned() == is_pin_memory_available() for row in audio)
    assert [row.tolist() for row in audio] == [[[10] * 3], [[20] * 3]]
    assert runner.model_intermediate_buffer["b"]["codes"]["audio"].tolist() == [[20] * 3]
