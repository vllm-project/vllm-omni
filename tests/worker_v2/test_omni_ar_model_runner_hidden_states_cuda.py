# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA coverage for the per-request teacher hidden-state host copy.

Lives outside ``test_omni_ar_model_runner_cuda.py`` because that file is one
of the AMD worker-copy curated files: the AMD pipeline lints its hardware
decorators and diffs the executed one-card tests against a fixed list, so a
CUDA-only test cannot be added there. Run it with:

    pytest tests/worker_v2/test_omni_ar_model_runner_hidden_states_cuda.py -m "core_model and cuda and L4"
"""

import pytest
import torch
from vllm.v1.worker.gpu.sample.output import SamplerOutput

import vllm_omni.worker_v2.omni_ar_model_runner as omni_ar_model_runner
from tests.helpers.mark import hardware_test
from vllm_omni.worker_v2.omni_ar_model_runner import TEACHER_HIDDEN_STATES_KEY, OmniAsyncOutput

pytestmark = pytest.mark.core_model


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_hidden_state_slices_reach_the_wire_channel_from_cuda() -> None:
    """Opted-in rows are copied to host memory under the teacher key; other requests get none."""
    device = torch.device("cuda")
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], device=device)
    mro = omni_ar_model_runner.OmniModelRunnerOutput(
        req_ids=["plain", "opted"],
        sampled_token_ids=None,
        prompt_logprobs_dict={},
        prompt_token_id_logprobs_dict={},
    )
    num_sampled = torch.tensor([1, 1], device=device, dtype=torch.int32)
    sampler_output = SamplerOutput(
        torch.tensor([[1], [2]], device=device), None, None, num_sampled, torch.zeros_like(num_sampled)
    )
    output = OmniAsyncOutput(
        model_runner_output=mro,
        sampler_output=sampler_output,
        num_sampled_tokens=num_sampled,
        main_stream=torch.cuda.current_stream(),
        copy_stream=torch.cuda.Stream(),
        hidden_state_slices={"opted": hidden[1:3]},
    ).get_output()
    payload = output.multimodal_outputs[1][TEACHER_HIDDEN_STATES_KEY]
    # A later step reusing the device buffer must not alter the delivered payload.
    hidden.zero_()

    assert output.multimodal_outputs[0] is None
    assert payload.device.type == "cpu"
    torch.testing.assert_close(payload, torch.tensor([[3.0, 4.0], [5.0, 6.0]]), rtol=0, atol=0)
