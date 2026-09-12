# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Snapshot metadata must not accumulate as if it were generated content."""

import pytest
import torch
from vllm.outputs import PoolingRequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import FinishReason

from vllm_omni.outputs.mm_outputs import MultimodalPayload
from vllm_omni.outputs.output_modality import OutputModality, TensorAccumulationStrategy
from vllm_omni.outputs.output_processor import MultimodalOutputProcessor, OmniRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
RATE_KEYS = ("sr", "sample_rate", "audio_sample_rate")


def _state(kind: RequestOutputKind) -> OmniRequestState:
    return OmniRequestState(
        request_id="audio",
        external_req_id="audio",
        parent_req=None,
        request_index=0,
        lora_request=None,
        prompt=None,
        prompt_token_ids=[0],
        prompt_embeds=None,
        logprobs_processor=None,
        detokenizer=None,
        max_tokens_param=None,
        arrival_time=0.0,
        queue=None,
        log_stats=False,
        stream_interval=1,
        output_kind=kind,
    )


@pytest.mark.parametrize("key", RATE_KEYS)
@pytest.mark.parametrize("representation", ["integer", "scalar_tensor", "vector_tensor", "metadata_tensor"])
def test_rate_is_latest_snapshot_before_consolidation(key, representation):
    payload = MultimodalPayload()
    for index, rate in enumerate((16000, 24000, 48000)):
        value = (
            rate if representation == "integer" else torch.tensor([rate] if representation == "vector_tensor" else rate)
        )
        incoming = (
            MultimodalPayload(tensors={"audio": torch.full((2,), index)}, metadata={key: value})
            if representation == "metadata_tensor"
            else MultimodalPayload.from_dict({"audio": torch.full((2,), index), key: value})
        )
        assert incoming is not None
        payload = payload.merged_with(incoming)
        snapshot = payload[key]
        assert not isinstance(snapshot, list)
        assert int(snapshot) == rate
        assert payload.get(key) is payload.to_dict()[key]
    payload = payload.merged_with(MultimodalPayload(tensors={"audio": torch.full((2,), 3)}))
    assert int(payload[key]) == 48000  # Missing metadata means retain, not clear.
    payload.consolidate_tensors(TensorAccumulationStrategy.CONCAT_LAST)
    payload.consolidate_metadata()
    torch.testing.assert_close(payload["audio"], torch.arange(4).repeat_interleave(2))


@pytest.mark.parametrize("key", RATE_KEYS)
@pytest.mark.parametrize("tensor_first", [False, True])
def test_rate_representation_change_removes_stale_partition(key, tensor_first):
    first = torch.tensor(16000) if tensor_first else 16000
    last = 24000 if tensor_first else torch.tensor(24000)
    payload = MultimodalPayload.from_dict({key: first})
    incoming = MultimodalPayload.from_dict({key: last})
    assert payload is not None and incoming is not None
    incoming_keys = (set(incoming.tensors), set(incoming.metadata))
    payload = payload.merged_with(incoming)
    assert int(payload[key]) == 24000
    assert payload[key] is payload.to_dict()[key]
    assert (key in payload.tensors) != (key in payload.metadata)
    assert incoming[key] is last
    assert incoming_keys == (set(incoming.tensors), set(incoming.metadata))


@pytest.mark.parametrize("key", RATE_KEYS)
def test_self_merge_retains_snapshot_and_content(key):
    payload = MultimodalPayload.from_dict({key: torch.tensor(24000), "audio": torch.tensor([1.0, 2.0])})
    assert payload is not None
    payload = payload.merged_with(payload)
    assert int(payload[key]) == 24000
    payload.consolidate_tensors(TensorAccumulationStrategy.CONCAT_LAST)
    torch.testing.assert_close(payload["audio"], torch.tensor([1.0, 2.0, 1.0, 2.0]))


@pytest.mark.parametrize("key", RATE_KEYS)
@pytest.mark.parametrize("kind", [RequestOutputKind.DELTA, RequestOutputKind.CUMULATIVE])
def test_long_stream_keeps_rate_bounded_without_losing_audio(key, kind):
    state = _state(kind)
    for index in range(1000):
        state.add_multimodal_tensor(
            {"model_outputs": torch.full((3,), index / 1000), key: torch.tensor(24000)}, "audio"
        )
        output = state.make_request_output([], None, None, None)
        assert output is not None and not isinstance(output, PoolingRequestOutput)
        payload = output.outputs[0].multimodal_output
        assert isinstance(payload[key], torch.Tensor) and payload[key].numel() == 1
        audio = payload["audio"]
        assert audio.numel() == (3 if kind == RequestOutputKind.DELTA else 3 * (index + 1))
        torch.testing.assert_close(audio[-3:], torch.full((3,), index / 1000))
        assert ("audio" in state.mm_accumulated) is (kind == RequestOutputKind.CUMULATIVE)
    final = state.make_request_output([], None, FinishReason.STOP, None)
    assert final is not None and final.finished
    if kind == RequestOutputKind.DELTA:
        assert "audio" not in final.outputs[0].multimodal_output


@pytest.mark.parametrize("pending", [0, 3])
def test_abort_flushes_only_pending_content_with_latest_rate(pending):
    state = _state(RequestOutputKind.DELTA)
    processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False, output_modality=OutputModality.AUDIO)
    processor.request_states[state.request_id] = state
    processor.external_req_ids[state.external_req_id] = [state.request_id]
    for rate in (16000, 24000):
        state.add_multimodal_tensor({"model_outputs": torch.ones(3), "sr": torch.tensor(rate)}, "audio")
        output = state.make_request_output([], None, None, None)
        assert output is not None
    if pending:
        state.add_multimodal_tensor({"model_outputs": torch.ones(pending)}, "audio")
    aborted, outputs = processor.abort_requests_collecting_outputs([state.request_id], internal=True)
    assert aborted == [state.request_id] and len(outputs) == 1 and outputs[0].finished
    payload = outputs[0].outputs[0].multimodal_output
    assert int(payload["sr"]) == 24000
    assert (payload["audio"].numel() if "audio" in payload else 0) == pending
    assert state.request_id not in processor.request_states


def test_non_snapshot_tensors_still_accumulate_and_empty_merge_keeps_identity():
    first = MultimodalPayload.from_dict({"latent": torch.tensor([1.0]), "audio": torch.tensor([2.0])})
    second = MultimodalPayload.from_dict({"latent": torch.tensor([3.0]), "audio": torch.tensor([4.0])})
    assert first is not None and second is not None
    assert MultimodalPayload().merged_with(first) is first
    merged = first.merged_with(second)
    merged.consolidate_tensors(TensorAccumulationStrategy.CONCAT_LAST)
    torch.testing.assert_close(merged["latent"], torch.tensor([1.0, 3.0]))
    torch.testing.assert_close(merged["audio"], torch.tensor([2.0, 4.0]))
