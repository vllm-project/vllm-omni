# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for MRv2 Omni AR text detach and generation payload outputs."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import FinishReason

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


def _make_processor(output_modality, detokenizer=None):
    state = OmniRequestState(
        request_id="r",
        external_req_id="r",
        parent_req=None,
        request_index=0,
        lora_request=None,
        output_kind=RequestOutputKind.FINAL_ONLY,
        prompt="prompt" if detokenizer else None,
        prompt_token_ids=[1] if detokenizer else [],
        prompt_embeds=None,
        logprobs_processor=MagicMock(logprobs=None, cumulative_logprob=None, prompt_logprobs=None)
        if detokenizer
        else None,
        detokenizer=detokenizer,
        max_tokens_param=None,
        arrival_time=0.0,
        queue=None,
        log_stats=False,
        stream_interval=1,
    )
    processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False, output_modality=output_modality)
    processor.request_states["r"] = state
    processor.external_req_ids["r"].append("r")
    return processor, state


def _rehydrated_output(*, pooling_output=None, finish_reason=FinishReason.STOP):
    """Build the post-StagePool shape without violating vLLM's wire schema."""
    return SimpleNamespace(
        request_id="r",
        new_token_ids=[42],
        pooling_output=pooling_output,
        finish_reason=finish_reason,
        stop_reason=None,
        kv_transfer_params=None,
        ec_transfer_params=None,
        routed_experts=None,
        trace_headers=None,
        prefill_stats=None,
        spec_decode_metrics=None,
        new_sampling_mask=None,
        num_nans_in_logits=0,
        multimodal_output=None,
        is_segment_finished=False,
        is_non_final_audio_chunk=False,
        output_type=None,
        num_generation_tokens=None,
        finished=True,
    )


def test_text_tokens_are_detokenized_when_mrv2_ar_output_has_pooling_payload():
    detokenizer = _Detokenizer()
    processor, state = _make_processor(OutputModality.LATENT, detokenizer)

    # StagePool has already rehydrated the MRv2 carrier.  Use a structural
    # output here because upstream EngineCoreOutput strictly validates
    # pooling_output as torch.Tensor | None on construction.
    output = _rehydrated_output(pooling_output={"hidden": torch.ones(1, 4)})

    processed = processor.process_outputs([output])

    assert detokenizer.update_calls == [([42], True)]
    completion = processed.request_outputs[0].outputs[0]
    assert list(completion.token_ids) == [42]
    assert completion.text == "X"
    assert not state.mm_accumulated.is_empty


@pytest.mark.parametrize("streaming", [False, True])
def test_generation_stage_mm_delivered_only_with_terminal(streaming):
    # Non-terminal chunks accumulate; the terminal output carries the audio.
    # When the first output is already terminal it is delivered directly.
    processor, _ = _make_processor(OutputModality.AUDIO)
    audio = torch.ones(1, 320)
    sr = torch.tensor(24000, dtype=torch.int32)
    mm_output = OmniEngineCoreOutput(
        request_id="r",
        new_token_ids=[],
        multimodal_output={"model_outputs": audio, "sr": sr},
        finish_reason=None if streaming else FinishReason.STOP,
    )

    processed = processor.process_outputs([mm_output])
    if streaming:
        assert processed.request_outputs == []
        processed = processor.process_outputs(
            [OmniEngineCoreOutput(request_id="r", new_token_ids=[], finish_reason=FinishReason.STOP)]
        )

    assert len(processed.request_outputs) == 1
    completion = processed.request_outputs[0].outputs[0]
    assert completion.text == ""
    assert torch.equal(completion.multimodal_output["audio"], audio)
    assert completion.multimodal_output["sr"].item() == 24000
