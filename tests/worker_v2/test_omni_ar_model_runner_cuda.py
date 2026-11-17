# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA coverage for MRV2 AR output transfer."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker

import vllm_omni.worker_v2.omni_ar_model_runner as omni_ar_model_runner
from tests.helpers.mark import hardware_test
from vllm_omni.worker_v2.omni_ar_model_runner import OmniAsyncOutput

pytestmark = pytest.mark.core_model


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_chunked_fixed_token_scores_are_copied_from_cuda() -> None:
    device = torch.device("cuda")
    hidden = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0], [4.0, 5.0, 6.0]], device=device)
    worker = PromptLogprobsWorker(1, device)
    worker.add_request(
        "req", 0, SamplingParams(prompt_logprob_token_ids=np.array([[2, 0], [2, 0]]), prompt_logprob_start=1)
    )
    batch = SimpleNamespace(
        req_ids=["req"],
        idx_mapping_np=np.array([0]),
        num_computed_prefill_tokens_np=np.array([0]),
        num_scheduled_tokens=np.array([2]),
        prefill_len_np=np.array([4]),
        query_start_loc_np=np.array([0, 2]),
    )
    assert worker.compute_prompt_token_id_logprobs(lambda hidden: hidden, hidden[:2], batch, np.array([4])) == {}
    batch.num_computed_prefill_tokens_np = np.array([2])
    scores = worker.compute_prompt_token_id_logprobs(lambda hidden: hidden, hidden[2:], batch, np.array([4]))
    mro = omni_ar_model_runner.OmniModelRunnerOutput(
        req_ids=["req"],
        req_id_to_index={"req": 0},
        sampled_token_ids=None,
        prompt_logprobs_dict={},
        prompt_token_id_logprobs_dict=scores,
    )
    num_sampled = torch.tensor([1], device=device, dtype=torch.int32)
    sampler_output = SamplerOutput(
        torch.tensor([[1]], device=device), None, None, num_sampled, torch.zeros_like(num_sampled)
    )
    output = OmniAsyncOutput(
        model_runner_output=mro,
        sampler_output=sampler_output,
        num_sampled_tokens=num_sampled,
        main_stream=torch.cuda.current_stream(),
        copy_stream=torch.cuda.Stream(),
    ).get_output()

    expected = hidden[1:3].log_softmax(dim=-1)[:, [2, 0]].cpu()
    assert output.prompt_token_id_logprobs_dict["req"].device.type == "cpu"
    torch.testing.assert_close(output.prompt_token_id_logprobs_dict["req"], expected)
