# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Talker context must leave room for both conditioning and codec output."""

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.request import Request, RequestStatus

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import load_deploy_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("prompt_len", [892, 1011])
def test_long_qwen3_tts_prompt_can_finish_within_codec_budget(prompt_len):
    deploy = load_deploy_config(get_deploy_config_path("qwen3_tts.yaml"))
    talker = next(stage for stage in deploy.stages if stage.stage_id == 0)
    params = SamplingParams(**talker.default_sampling_params, stop_token_ids=[4198])
    request = Request("long-tts", [1] * prompt_len, params, pooling_params=None)

    # #10781: 881 text tokens produce a 892-token conditioned prompt. A
    # 4096-token total context stops after only 3204 of the 4096 codec tokens.
    request.append_output_token_ids([42] * 3204)
    assert not check_stop(request, talker.max_model_len)

    # Even the last codec position must remain available for a natural EOS.
    request.append_output_token_ids([42] * (params.max_tokens - 1 - request.num_output_tokens))
    assert not check_stop(request, talker.max_model_len)
    request.append_output_token_ids(4198)
    assert check_stop(request, talker.max_model_len)
    assert request.status == RequestStatus.FINISHED_STOPPED


def test_qwen3_tts_context_headroom_preserves_codec_limit():
    deploy = load_deploy_config(get_deploy_config_path("qwen3_tts.yaml"))
    talker = next(stage for stage in deploy.stages if stage.stage_id == 0)
    params = SamplingParams(**talker.default_sampling_params, stop_token_ids=[4198])
    request = Request("runaway-tts", [1] * 892, params, pooling_params=None)

    request.append_output_token_ids([42] * params.max_tokens)
    assert check_stop(request, talker.max_model_len)
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
