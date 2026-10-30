# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from transformers import Qwen3Config
from vllm import SamplingParams
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_omni.model_executor.models.soprano.contracts import select_predictors, transform_prompt
from vllm_omni.model_executor.stage_input_processors.soprano import ar_to_decoder_full_payload

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def request_snapshot(prompt_ids, sampled_ids):
    return CachedRequestState(
        req_id="internal",
        prompt_token_ids=prompt_ids,
        output_token_ids=sampled_ids,
        mm_features=[],
        sampling_params=SamplingParams(),
        generator=None,
        block_ids=([],),
        num_computed_tokens=0,
    )


@pytest.mark.parametrize(
    "prompt_length,valid,eos", [(1, 0, True), (40, 16, True), (40, 512, False), (1, 1, False), (4, 1, True)]
)
def test_predictors_belong_to_sampled_step_not_consumed_token(prompt_length, valid, eos):
    sampled = [100] * valid + ([3] if eos else [])
    rows = prompt_length + len(sampled) - 1
    hidden = torch.arange(rows, dtype=torch.float32).unsqueeze(1).expand(rows, 512)
    retained = select_predictors(hidden, prompt_length, sampled)
    assert retained[:, 0].tolist() == list(range(prompt_length - 1, prompt_length - 1 + valid))


def test_missing_final_length_vector_is_rejected():
    with pytest.raises(ValueError, match="consumed trajectory"):
        select_predictors(torch.zeros(41, 512), 40, [100, 101, 102])


def test_nonfinite_handoff_and_tokens_past_eos_are_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        select_predictors(torch.full((1, 512), float("nan")), 1, [100])
    with pytest.raises(ValueError, match="after Soprano EOS"):
        select_predictors(torch.zeros(2, 512), 1, [3, 100])


def test_handoff_carries_failure_instead_of_silently_waiting():
    request = request_snapshot([3, 1], [10, 3])
    failed = ar_to_decoder_full_payload(None, {"hidden": torch.zeros(2, 512)}, request)
    assert "soprano_error" in failed["additional_information"]
    assert failed["meta"]["finished"] is True
    valid = ar_to_decoder_full_payload(None, {"hidden": torch.ones(3, 512)}, request)
    assert valid["hidden_states"]["output"].shape == (1, 512)


def test_eos_in_prompt_does_not_discard_prompt_tail():
    hidden = torch.arange(3, dtype=torch.float32).unsqueeze(1).expand(3, 512)
    request = request_snapshot([3, 1], [100, 3])
    output = ar_to_decoder_full_payload(None, {"hidden": hidden}, request)
    assert output["hidden_states"]["output"][0, 0] == 1


def test_explicit_pipeline_does_not_capture_plain_qwen3(monkeypatch):
    from vllm_omni.config.config_factory import StageConfigFactory
    from vllm_omni.config.stage_config import DeployConfig

    config = Qwen3Config()
    monkeypatch.setattr(StageConfigFactory, "try_infer_model_type", classmethod(lambda cls, **kw: "qwen3"))
    monkeypatch.setattr(StageConfigFactory, "get_hf_config", classmethod(lambda cls, **kw: config))
    assert StageConfigFactory.get_pipeline_config("plain-qwen", False) is None
    selected = StageConfigFactory.get_pipeline_config(
        "local-soprano", False, user_deploy_config=DeployConfig(pipeline="soprano")
    )
    assert selected.model_type == "soprano"
    assert not selected.hf_architectures


def test_prompt_is_built_once_and_numbers_follow_official_normalization():
    pytest.importorskip("inflect")
    pytest.importorskip("unidecode")
    params = [SamplingParams()]
    assert transform_prompt(" Hello world. ", params)["prompt"] == "[STOP][TEXT]hello world.[START]"
    tokenized = {"prompt_token_ids": [3, 1, 2]}
    assert transform_prompt(tokenized, params) is tokenized
    assert transform_prompt("I have 12 apples.", params)["prompt"] == "[STOP][TEXT]i have twelve apples.[START]"


def test_omni_metadata_does_not_change_native_forward(monkeypatch):
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

    from vllm_omni.model_executor.models.soprano.soprano_ar import SopranoForCausalLM

    observed = []

    def forward(self, *args):
        observed.append(args)
        return "native-result"

    monkeypatch.setattr(Qwen3ForCausalLM, "forward", forward)
    instance = object.__new__(SopranoForCausalLM)
    assert instance.forward("tokens", "positions", model_intermediate_buffer=[{}]) == "native-result"
    assert observed == [("tokens", "positions", None, None)]
