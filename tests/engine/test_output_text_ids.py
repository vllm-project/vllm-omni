# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Client text deltas may differ from the AR scheduler's control tokens."""

from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, decoders, models
from transformers import PreTrainedTokenizerFast
from vllm.outputs import CompletionOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import FinishReason
from vllm.v1.engine.detokenizer import SlowIncrementalDetokenizer
from vllm.v1.engine.logprobs import LogprobsProcessor

from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.outputs.output_processor import MultimodalOutputProcessor, OmniRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def text_processor():
    # Small real tokenizer; no model weights or external downloads are needed
    # to exercise native incremental decoding and STOP handling.
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "你": 1, "好": 2, "🙂": 3}, unk_token="[UNK]"))
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    return MultimodalOutputProcessor(tokenizer=tokenizer, log_stats=False)


def add_text_request(processor, request_id, output_kind, include_stop=True):
    params = SamplingParams(output_kind=output_kind, include_stop_str_in_output=include_stop)
    request = SimpleNamespace(prompt_token_ids=[], prompt_embeds=None, sampling_params=params)
    state = OmniRequestState(
        request_id=request_id,
        external_req_id=request_id,
        parent_req=None,
        request_index=0,
        lora_request=None,
        output_kind=output_kind,
        prompt=None,
        prompt_token_ids=[],
        prompt_embeds=None,
        detokenizer=SlowIncrementalDetokenizer(processor.tokenizer, request),
        logprobs_processor=LogprobsProcessor.from_new_request(processor.tokenizer, request),
        max_tokens_param=None,
        arrival_time=0.0,
        queue=None,
        log_stats=False,
        stream_interval=1,
    )
    processor.request_states[request_id] = state
    processor.external_req_ids[request_id].append(request_id)


@pytest.mark.parametrize("kind", list(RequestOutputKind))
@pytest.mark.parametrize("finish", [FinishReason.STOP, FinishReason.LENGTH])
@pytest.mark.parametrize("final_text", [True, False])
def test_visible_text_delta_preserves_tail_and_scheduler_output(text_processor, kind, finish, final_text):
    add_text_request(text_processor, "text", kind)
    # Cover both a final step with text and audio continuing after text ends.
    # An empty delta must never fall back to decoding the scheduler's blank.
    chunks = [[1], [2, 3]] if final_text else [[1], [2, 3], []]
    emitted: list[CompletionOutput] = []
    for index, ids in enumerate(chunks):
        core = OmniEngineCoreOutput(
            request_id="text",
            new_token_ids=[999],
            finish_reason=finish if index == len(chunks) - 1 else None,
            stop_reason=999 if index == len(chunks) - 1 else None,
            # Match the runner's list-to-tensor conversion, including []
            # inferring a floating dtype after the text stream has ended.
            multimodal_output={"ids.output": torch.tensor(ids), "codes.audio": torch.tensor([index])},
        )
        result = text_processor.process_outputs([core])
        assert core.new_token_ids == [999]  # The original scheduler output is not rewritten.
        emitted.extend(output.outputs[0] for output in result.request_outputs)

    text = "".join(output.text for output in emitted) if kind == RequestOutputKind.DELTA else emitted[-1].text
    assert text == "你好🙂"
    assert emitted[-1].cumulative_token_ids == [1, 2, 3]
    assert emitted[-1].finish_reason == str(finish)
    assert emitted[-1].multimodal_output["ids"]["output"].dtype == torch.long
    torch.testing.assert_close(emitted[-1].multimodal_output["codes"]["audio"], torch.arange(len(chunks)))
    assert not text_processor.request_states


def test_native_text_request_without_separate_ids_is_unchanged(text_processor):
    add_text_request(text_processor, "plain", RequestOutputKind.FINAL_ONLY, include_stop=False)
    output = text_processor.process_outputs(
        [OmniEngineCoreOutput(request_id="plain", new_token_ids=[1, 2, 3, 0], finish_reason=FinishReason.STOP)]
    ).request_outputs[0]
    assert output.outputs[0].text == "你好🙂"


def test_separate_text_requires_preserving_the_final_token(text_processor):
    add_text_request(text_processor, "bad", RequestOutputKind.FINAL_ONLY, include_stop=False)
    with pytest.raises(ValueError, match="include_stop_str_in_output"):
        text_processor.process_outputs(
            [
                OmniEngineCoreOutput(
                    request_id="bad", new_token_ids=[999], multimodal_output={"ids.output": torch.tensor([1])}
                )
            ]
        )


def test_nonempty_text_delta_rejects_floating_token_ids(text_processor):
    add_text_request(text_processor, "bad", RequestOutputKind.FINAL_ONLY)
    with pytest.raises(ValueError, match="integer tensor"):
        text_processor.process_outputs(
            [
                OmniEngineCoreOutput(
                    request_id="bad", new_token_ids=[999], multimodal_output={"ids.output": torch.tensor([1.5])}
                )
            ]
        )
