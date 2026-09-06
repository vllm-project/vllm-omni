# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MRv2 Omni AR text + payload outputs."""

from unittest.mock import MagicMock

import pytest
import torch
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import EngineCoreOutput, FinishReason

from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.engine.output_modality import OutputModality
from vllm_omni.engine.output_processor import (
    MultimodalOutputProcessor,
    OmniRequestState,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Detokenizer:
    def __init__(self):
        self.output_token_ids = []
        self.output_text = ""
        self.update_calls = []

    def update(self, token_ids, stop):
        self.update_calls.append((list(token_ids), stop))
        self.output_token_ids.extend(token_ids)
        self.output_text += "X" * len(token_ids)
        return None

    def get_next_output_text(self, finished, delta):
        return self.output_text

    def num_output_tokens(self):
        return len(self.output_token_ids)


def _make_processor_state(detokenizer):
    logprobs = MagicMock(
        logprobs=None,
        cumulative_logprob=None,
        prompt_logprobs=None,
    )
    logprobs.update_from_output = MagicMock()

    state = OmniRequestState(
        request_id="r",
        external_req_id="r",
        parent_req=None,
        request_index=0,
        lora_request=None,
        output_kind=RequestOutputKind.FINAL_ONLY,
        prompt="prompt",
        prompt_token_ids=[1],
        prompt_embeds=None,
        logprobs_processor=logprobs,
        detokenizer=detokenizer,
        max_tokens_param=None,
        arrival_time=0.0,
        queue=None,
        log_stats=False,
        stream_interval=1,
    )
    processor = MultimodalOutputProcessor(
        tokenizer=None,
        log_stats=False,
        output_modality=OutputModality.LATENT,
    )
    processor.request_states["r"] = state
    processor.external_req_ids["r"].append("r")
    return processor, state


def _make_generation_processor_state():
    state = OmniRequestState(
        request_id="r",
        external_req_id="r",
        parent_req=None,
        request_index=0,
        lora_request=None,
        output_kind=RequestOutputKind.FINAL_ONLY,
        prompt=None,
        prompt_token_ids=[],
        prompt_embeds=None,
        logprobs_processor=None,
        detokenizer=None,
        max_tokens_param=None,
        arrival_time=0.0,
        queue=None,
        log_stats=False,
        stream_interval=1,
    )
    processor = MultimodalOutputProcessor(
        tokenizer=None,
        log_stats=False,
        output_modality=OutputModality.AUDIO,
    )
    processor.request_states["r"] = state
    processor.external_req_ids["r"].append("r")
    return processor, state


def test_text_tokens_are_detokenized_when_mrv2_ar_output_has_pooling_payload():
    detokenizer = _Detokenizer()
    processor, state = _make_processor_state(detokenizer)

    output = EngineCoreOutput(
        request_id="r",
        new_token_ids=[42],
        pooling_output={"hidden": torch.ones(1, 4)},
        finish_reason=FinishReason.STOP,
    )

    processed = processor.process_outputs([output])

    assert detokenizer.update_calls == [([42], True)]
    completion = processed.request_outputs[0].outputs[0]
    assert list(completion.token_ids) == [42]
    assert completion.text == "X"
    assert list(completion.cumulative_token_ids) == [42]
    assert not state.mm_accumulated.is_empty


def test_generation_stage_mm_only_output_is_returned_without_queue():
    processor, _ = _make_generation_processor_state()
    audio = torch.ones(1, 320)
    sr = torch.tensor(24000, dtype=torch.int32)

    output = OmniEngineCoreOutput(
        request_id="r",
        new_token_ids=[],
        multimodal_output={"model_outputs": audio, "sr": sr},
        finish_reason=FinishReason.STOP,
    )

    processed = processor.process_outputs([output])

    assert len(processed.request_outputs) == 1
    request_output = processed.request_outputs[0]
    assert request_output.finished
    completion = request_output.outputs[0]
    assert completion.text == ""
    assert "audio" in completion.multimodal_output
    assert torch.equal(completion.multimodal_output["audio"], audio)
    assert completion.multimodal_output["sr"].item() == 24000


def test_generation_stage_accumulates_mm_until_terminal_output():
    processor, _ = _make_generation_processor_state()
    audio = torch.ones(1, 320)
    sr = torch.tensor(24000, dtype=torch.int32)

    output = OmniEngineCoreOutput(
        request_id="r",
        new_token_ids=[],
        multimodal_output={"model_outputs": audio, "sr": sr},
        finish_reason=None,
    )
    terminal = OmniEngineCoreOutput(
        request_id="r",
        new_token_ids=[],
        finish_reason=FinishReason.STOP,
    )

    processed = processor.process_outputs([output])
    assert processed.request_outputs == []

    processed = processor.process_outputs([terminal])

    assert len(processed.request_outputs) == 1
    completion = processed.request_outputs[0].outputs[0]
    assert "audio" in completion.multimodal_output
    assert torch.equal(completion.multimodal_output["audio"], audio)
    assert completion.multimodal_output["sr"].item() == 24000
