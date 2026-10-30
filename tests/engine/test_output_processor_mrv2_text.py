# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for MRv2 Omni AR text detach and generation payload outputs."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreRequest, FinishReason
from vllm.v1.engine.logprobs import LogprobsProcessor

from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.engine.output_modality import OutputModality
from vllm_omni.engine.output_processor import (
    MultimodalOutputProcessor,
    OmniRequestState,
)
from vllm_omni.outputs import OmniRequestOutput

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


def test_mrv2_payload_mirrored_on_both_channels_is_accumulated_once():
    # The MRv2 AR full-payload fallback publishes one payload as both
    # pooling_output and multimodal_output; each row must reach the next stage once.
    processor, _ = _make_processor(OutputModality.LATENT, _Detokenizer())
    hidden = torch.arange(12.0).reshape(3, 4)
    ids = torch.tensor([[11], [12], [13]])
    output = _rehydrated_output(pooling_output={"latent": hidden, "latent_input_ids": ids})
    output.multimodal_output = {"latent": hidden, "latent_input_ids": ids}

    payload = processor.process_outputs([output]).request_outputs[0].outputs[0].multimodal_output

    torch.testing.assert_close(payload["latent"], hidden)
    torch.testing.assert_close(payload["latent_input_ids"], ids)


def _process_mrv2_mirror(pooling_output, multimodal_output):
    processor, _ = _make_processor(OutputModality.LATENT, _Detokenizer())
    output = _rehydrated_output(pooling_output=pooling_output)
    output.multimodal_output = multimodal_output
    return processor.process_outputs([output]).request_outputs[0].outputs[0].multimodal_output


def test_mrv2_partially_mirrored_nested_payload_keeps_every_layer_once():
    # The channels are decoded separately, so the mirror is equal, not identical.
    hidden = torch.arange(12.0).reshape(3, 4)
    layer = torch.arange(6.0).reshape(3, 2)
    payload = _process_mrv2_mirror(
        {"latent": hidden, "hidden_states": {"layers": {0: layer, 1: layer + 1}}},
        {"latent": hidden.clone(), "hidden_states": {"layers": {0: layer.clone()}}},
    )

    torch.testing.assert_close(payload["latent"], hidden)
    # _to_cpu stringifies nested dict keys on both channels.
    layers = payload["hidden_states"]["layers"]
    assert set(layers) == {"0", "1"}
    torch.testing.assert_close(layers["0"], layer)
    torch.testing.assert_close(layers["1"], layer + 1)


def test_mrv2_mirror_with_nan_and_deep_nesting_is_accumulated_once():
    hidden = torch.tensor([[1.0, float("nan")], [2.0, 3.0]])
    speakers = [torch.tensor([1, 2]), torch.tensor([3])]
    # Two levels down, so consolidate_metadata leaves the list as is.
    payload = _process_mrv2_mirror(
        {"latent": hidden, "extra": {"deep": {"t": hidden, "speakers": speakers}}},
        {
            "latent": hidden.clone(),
            "extra": {"deep": {"t": hidden.clone(), "speakers": [t.clone() for t in speakers]}},
        },
    )

    torch.testing.assert_close(payload["latent"], hidden, equal_nan=True)
    deep = payload["extra"]["deep"]
    torch.testing.assert_close(deep["t"], hidden, equal_nan=True)
    assert len(deep["speakers"]) == 2


def test_mrv2_distinct_same_key_values_are_not_dropped():
    first, second = torch.ones(1, 4), torch.zeros(1, 4)
    payload = _process_mrv2_mirror(
        {"latent": second, "codes": {"audio": torch.tensor([1, 2])}},
        {"latent": first, "codes": {"audio": torch.tensor([7, 8]), "text": torch.tensor([5])}},
    )

    # A top-level tensor accumulates after the mm_output one, as before.
    torch.testing.assert_close(payload["latent"], torch.cat([first, second]))
    # Nested metadata keeps both siblings; the pooling value wins on conflict.
    torch.testing.assert_close(payload["codes"]["audio"], torch.tensor([1, 2]))
    torch.testing.assert_close(payload["codes"]["text"], torch.tensor([5]))


def test_mrv2_differing_top_level_metadata_matches_separate_accumulation():
    # Same result as accumulating the two channels one after the other: the
    # pooling value replaces differing metadata, and a tensor and a non-tensor
    # under one key keep both partitions, where the snapshot shows the metadata.
    processor, state = _make_processor(OutputModality.LATENT, _Detokenizer())
    hidden = torch.ones(1, 4)
    output = _rehydrated_output(
        pooling_output={"latent": hidden, "spk": ["b", "c"], "a": torch.tensor([9]), "b": ["y"]}
    )
    output.multimodal_output = {"latent": hidden.clone(), "spk": ["a"], "a": ["x"], "b": torch.tensor([4])}

    payload = processor.process_outputs([output]).request_outputs[0].outputs[0].multimodal_output

    torch.testing.assert_close(payload["latent"], hidden)
    assert payload.metadata == {"spk": ["b", "c"], "a": ["x"], "b": ["y"]}
    # The tensor side of each mismatched key is accumulated once, not dropped.
    accumulated = state.mm_accumulated
    assert set(accumulated.tensors) == {"latent", "a", "b"}
    torch.testing.assert_close(accumulated.tensors["latent"], hidden)
    torch.testing.assert_close(accumulated.tensors["a"], torch.tensor([9]))
    torch.testing.assert_close(accumulated.tensors["b"], torch.tensor([4]))


def test_fixed_token_scores_survive_output_processing_and_stage_wrapping():
    processor, state = _make_processor(OutputModality.TEXT, _Detokenizer())
    request = EngineCoreRequest(
        request_id="r",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(prompt_logprob_token_ids=[4, 5]),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    state.logprobs_processor = LogprobsProcessor.from_new_request(None, request)
    scores = torch.tensor([[-0.5, -1.5], [-0.2, -2.2]])
    processed = processor.process_outputs(
        [
            OmniEngineCoreOutput(
                request_id="r",
                new_token_ids=[42],
                prompt_token_id_logprobs=scores,
                finish_reason=FinishReason.STOP,
            )
        ]
    )

    (raw_output,) = processed.request_outputs
    final_output = OmniRequestOutput.from_stage_output(raw_output, stage_id=0)
    torch.testing.assert_close(torch.from_numpy(final_output.prompt_token_id_logprobs), scores)


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
