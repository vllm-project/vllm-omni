# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU AR model runner tests that need a CUDA device (CPU tests: test_gpu_ar_model_runner.py)."""

from types import SimpleNamespace

import pytest
import torch
from vllm.utils.platform_utils import is_pin_memory_available

from tests.helpers.mark import hardware_test
from tests.worker.test_gpu_ar_model_runner import _mtp_runner
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

pytestmark = [pytest.mark.core_model]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_post_sample_talker_mtp_moves_device_codes_with_one_pending_copy() -> None:
    runner, _ = _mtp_runner(
        async_scheduling=False,
        buffers={"a": {"duplex": {"data_plane": True}}, "b": {"duplex": {"data_plane": True}}},
    )
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
    assert all(row.device.type == "cpu" and row.is_pinned() == is_pin_memory_available() for row in audio)
    assert [row.tolist() for row in audio] == [[[10] * 3], [[20] * 3]]
    assert runner.model_intermediate_buffer["b"]["codes"]["audio"].tolist() == [[20] * 3]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_reused_output_worker_preserves_cuda_snapshots_and_sampled_tokens() -> None:
    from concurrent.futures import ThreadPoolExecutor

    from vllm_omni.outputs import OmniModelRunnerOutput
    from vllm_omni.worker.gpu_ar_model_runner import (
        OmniAsyncGPUModelRunnerOutput,
        _snapshot_tensor_payload_to_cpu_async,
    )

    copy_stream = torch.cuda.Stream()
    outputs = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        for token in (7, 8):
            codes = torch.full((2, 3), token, device="cuda")
            snapshot = _snapshot_tensor_payload_to_cpu_async({"audio": codes}, copy_stream=copy_stream, pin_memory=True)

            def build(snapshot=snapshot):
                snapshot.wait()
                return OmniModelRunnerOutput(
                    req_ids=["r"], req_id_to_index={"r": 0}, multimodal_outputs=[snapshot.payload]
                )

            outputs.append(
                OmniAsyncGPUModelRunnerOutput(
                    model_runner_output_builder=build,
                    output_executor=executor,
                    cuda_device=torch.device("cuda", torch.accelerator.current_device_index()),
                    sampled_token_ids=torch.tensor([[token]], device="cuda"),
                    logprobs_tensors=None,
                    invalid_req_indices=[],
                    async_output_copy_stream=copy_stream,
                    vocab_size=10,
                )
            )
            codes.fill_(-1)
        for token, output in zip((7, 8), outputs):
            result = output.get_output()
            assert result.sampled_token_ids == [[token]]
            torch.testing.assert_close(result.multimodal_outputs[0]["audio"], torch.full((2, 3), token))
