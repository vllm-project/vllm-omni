# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest
import torch
import yaml
from vllm import SamplingParams
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.request import Request, RequestStatus
from vllm.v1.sample.logits_processor import BatchUpdate, MinTokensLogitsProcessor

from vllm_omni.config.stage_config import merge_sampling_constraints
from vllm_omni.model_executor.models.breeze_tts_2.pipeline import BREEZE_TTS_2_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def talker_sampling_params():
    deploy_path = Path(__file__).parents[3] / "vllm_omni" / "deploy" / "breeze_tts_2.yaml"
    deploy = yaml.safe_load(deploy_path.read_text(encoding="utf-8"))
    params = SamplingParams(
        **merge_sampling_constraints(
            deploy["stages"][0]["default_sampling_params"],
            BREEZE_TTS_2_PIPELINE.stages[0].sampling_constraints,
        )
    )
    # Mirror the input processor: Breeze's text tokenizer uses EOS 1,
    # which is also an ordinary token in the generated codec vocabulary.
    params.update_from_generation_config({"eos_token_id": [1]}, eos_token_id=1)
    return params


@pytest.mark.parametrize("codec_token, should_stop", [(0, False), (1, False), (2047, False), (2051, True)])
def test_scheduler_stops_only_on_codec_eos(talker_sampling_params, codec_token, should_stop):
    request = Request(
        request_id="breeze-eos-test",
        prompt_token_ids=[7, 8, 9],
        sampling_params=talker_sampling_params,
        pooling_params=None,
    )
    request.append_output_token_ids([42, codec_token])
    request.status = RequestStatus.RUNNING

    assert check_stop(request, max_model_len=4096) is should_stop
    if should_stop:
        assert request.status == RequestStatus.FINISHED_STOPPED
        assert request.stop_reason == 2051
    else:
        assert request.status == RequestStatus.RUNNING


@pytest.mark.parametrize("codec_token", [1, 2051])
def test_first_step_keeps_codec_token_and_eos_logits(talker_sampling_params, codec_token, monkeypatch):
    # vLLM's tensor helper uses a global flag rather than is_pin_memory,
    # and pinned allocations require a CUDA device even for this CPU test.
    monkeypatch.setattr("vllm.utils.torch_utils.PIN_MEMORY", False)
    # ignore_eos still leaves text EOS in all_stop_token_ids. Exercise the
    # actual processor so the first codec frame and immediate EOS stay valid.
    processor = MinTokensLogitsProcessor(None, device=torch.device("cpu"), is_pin_memory=False)
    processor.update_state(
        BatchUpdate(batch_size=1, removed=[], added=[(0, talker_sampling_params, None, [])], moved=[])
    )
    logits = torch.full((1, 2052), -float("inf"))
    logits[0, codec_token] = 0.0

    processed = processor.apply(logits.clone())

    assert torch.equal(processed, logits)
    assert processed.argmax(dim=-1).item() == codec_token


def test_pipeline_disallows_minimum_tokens_that_mask_selected_codec_ids():
    params = merge_sampling_constraints({"min_tokens": 10}, BREEZE_TTS_2_PIPELINE.stages[0].sampling_constraints)
    assert params["min_tokens"] == 0
