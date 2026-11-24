# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_talker import (
    HiggsAudioRMSNorm,
    HiggsAudioV2DecoderLayer,
    HiggsAudioV2Sampler,
    HiggsAudioV2TalkerForConditionalGeneration,
)
from vllm_omni.model_executor.models.higgs_audio_v2.model_state import HiggsAudioV2ModelState
from vllm_omni.model_executor.models.higgs_audio_v2.pipeline import HIGGS_AUDIO_V2_PIPELINE
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.omni_sampler import OmniSamplingOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _RoutingTestAttention(torch.nn.Module):
    def forward(self, positions, hidden_states):
        return hidden_states


class _RoutingTestFinalNorm(torch.nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states + residual


@pytest.mark.parametrize("token_ids", [[4, 4], [9, 4], [3, 4], [1, 1]])
@torch.inference_mode()
def test_uniform_audio_routing_avoids_layer_indexing_without_rerouting_text(token_ids):
    """Pure audio skips boolean indexing; BOS and mixed spans keep their experts."""
    model = object.__new__(HiggsAudioV2TalkerForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = _Config()
    layers = []
    for _ in range(2):
        layer = object.__new__(HiggsAudioV2DecoderLayer)
        torch.nn.Module.__init__(layer)
        layer.self_attn = _RoutingTestAttention()
        layer.input_layernorm = HiggsAudioRMSNorm(4)
        layer.audio_input_layernorm = HiggsAudioRMSNorm(4)
        layer.post_attention_layernorm = HiggsAudioRMSNorm(4)
        layer.audio_post_attention_layernorm = HiggsAudioRMSNorm(4)
        layer.mlp = torch.nn.Linear(4, 4, bias=False)
        layer.audio_mlp = torch.nn.Linear(4, 4, bias=False)
        torch.nn.init.constant_(layer.mlp.weight, 2)
        torch.nn.init.constant_(layer.audio_mlp.weight, 3)
        layers.append(layer)
    model.layers = torch.nn.ModuleList(layers)
    model.norm = _RoutingTestFinalNorm()
    input_ids = torch.tensor(token_ids)
    positions = torch.arange(len(token_ids))
    embeddings = torch.arange(len(token_ids) * 4, dtype=torch.float32).reshape(-1, 4)
    mask = model.audio_token_mask(input_ids)
    expected = embeddings
    residual = None
    for layer in model.layers:
        expected, residual = layer(positions, expected, residual, audio_token_mask=mask)
    expected = model.norm(expected, residual)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        actual = model(input_ids, positions, inputs_embeds=embeddings)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    if all(token in (4, 9) for token in token_ids):
        assert not any(event.key == "aten::nonzero" for event in profile.key_averages())


@dataclass
class _Config:
    audio_bos_token_id: int = 3
    audio_token_id: int = 4
    audio_eos_token_id: int = 5
    audio_delay_token_id: int = 9
    audio_stream_bos_id: int = 6
    audio_stream_eos_id: int = 7
    num_codebooks: int = 2
    codebook_size: int = 8


@dataclass
class _SamplerOutput:
    sampled_token_ids: torch.Tensor
    num_sampled: torch.Tensor | None = None
    num_rejected: torch.Tensor | None = None


@dataclass
class _SamplerBatch:
    req_ids: list[str]
    num_reqs: int
    num_draft_tokens: int
    logits_indices: torch.Tensor
    input_ids: torch.Tensor


@dataclass
class _CodesBatch:
    num_reqs: int
    cu_num_logits_np: np.ndarray


@dataclass
class _StateBatch:
    num_reqs: int
    req_ids: list[str]
    query_start_loc_np: np.ndarray
    input_ids: torch.Tensor
    positions: torch.Tensor


def _model() -> HiggsAudioV2TalkerForConditionalGeneration:
    return object.__new__(HiggsAudioV2TalkerForConditionalGeneration)


def test_higgs_audio_v2_registers_model_owned_mrv2_sampler(mocker) -> None:
    model = _model()
    base_sampler = mocker.Mock()

    sampler, draft_sampler = model.mrv2_custom_sampler(base_sampler)

    assert isinstance(sampler, HiggsAudioV2Sampler)
    assert sampler.base_sampler is base_sampler
    assert sampler.model is model
    assert draft_sampler is None


def test_higgs_audio_v2_sampler_publishes_current_step_codes(mocker) -> None:
    model = _model()
    hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    logits = torch.zeros(2, 8)
    model.compute_logits = mocker.Mock(return_value=logits)
    model._row_req_ids = mocker.Mock(return_value=["second", "first"])
    model._apply_audio_mode_bias = mocker.Mock()
    sampled = _SamplerOutput(
        sampled_token_ids=torch.tensor([[4], [4]]),
        num_sampled=torch.tensor([1, 1]),
        num_rejected=torch.tensor([0, 0]),
    )
    model._dispatch_audio_codes = mocker.Mock(return_value=sampled)
    current_codes = {"codes": {"audio": [torch.tensor([[1, 2]]), torch.tensor([[3, 4]])]}}
    model._current_codes_by_request = mocker.Mock(return_value=current_codes)
    base_sampler = mocker.Mock(return_value=sampled)
    batch = _SamplerBatch(
        req_ids=["second", "first", "stale"],
        num_reqs=2,
        num_draft_tokens=0,
        logits_indices=torch.tensor([0, 2]),
        input_ids=torch.tensor([3, 99, 4]),
    )

    result = HiggsAudioV2Sampler(base_sampler, model).sample_step(
        hidden,
        batch,
        req_states=object(),
        grammar_output=None,
        standard_sample=mocker.Mock(),
    )

    assert isinstance(result, OmniSamplingOutput)
    assert result.sampler_output is sampled
    assert result.multimodal_outputs is current_codes
    assert result.include_hidden_states is False
    assert result.owns_multimodal_outputs is True
    compute_args = model.compute_logits.call_args.args
    assert torch.equal(compute_args[0], hidden[batch.logits_indices])
    assert compute_args[1] is None
    model._apply_audio_mode_bias.assert_called_once_with(
        logits,
        None,
        request_ids=["second", "first"],
        row_prev_tokens=[3, 4],
    )
    dispatch_args = model._dispatch_audio_codes.call_args.args
    dispatch_kwargs = model._dispatch_audio_codes.call_args.kwargs
    assert dispatch_args[0] is sampled
    assert torch.equal(dispatch_args[1], hidden[batch.logits_indices])
    assert dispatch_args[2] is None
    assert dispatch_kwargs == {"request_ids": ["second", "first"]}


def test_higgs_audio_v2_mrv2_sample_emits_request_keyed_codec_frame() -> None:
    model = _model()
    model.config = _Config()
    model._audio_state_uses_request_ids = True
    model._mrv2_req_ids = ["request-a"]
    model._mrv2_query_start_loc = [0, 1]
    model._last_step_input_ids = torch.tensor([4])
    model._last_step_query_start_loc = torch.tensor([0, 1])
    model._last_logits_hidden = torch.ones(1, 2)
    model.__dict__["audio_codebook_logits"] = lambda hidden, mask: torch.zeros((int(mask.sum()), 2, 8))
    model.__dict__["_sample_audio_codes_upstream"] = lambda logits, **_: torch.tensor([1, 2])
    sampler_output = _SamplerOutput(sampled_token_ids=torch.tensor([[4]]))

    result = model._dispatch_audio_codes(
        sampler_output,
        model._last_logits_hidden,
        None,
        request_ids=["request-a"],
    )

    assert result is sampler_output
    assert model._last_audio_codes.tolist() == [[1, 6]]
    assert model._audio_state["request-a"]["audio_out_ids"].tolist() == [[1], [6]]


def test_higgs_audio_v2_partial_prefill_does_not_advance_audio_state() -> None:
    model = _model()
    model.config = _Config()
    model._last_first_audio_after_bos = torch.tensor([False])
    sampler_output = _SamplerOutput(
        sampled_token_ids=torch.tensor([[4]]),
        num_sampled=torch.tensor([0]),
        num_rejected=torch.tensor([0]),
    )

    result = model._dispatch_audio_codes(
        sampler_output,
        torch.ones(1, 2),
        None,
        request_ids=["request-a"],
    )

    assert result is sampler_output
    assert model._last_audio_codes is None
    assert model._audio_state == {}


def test_higgs_audio_v2_current_codes_keep_empty_request_rows() -> None:
    model = _model()
    model._last_audio_codes = torch.tensor([[-1, -1], [3, 4]])
    batch = _CodesBatch(
        num_reqs=2,
        cu_num_logits_np=np.array([0, 1, 2]),
    )

    result = model._current_codes_by_request(batch)

    rows = result["codes"]["audio"]
    assert [row.tolist() for row in rows] == [[], [[3, 4]]]
    assert rows[0].shape == (0, 2)
    assert all(row.dtype == torch.int32 for row in rows)


def test_higgs_audio_v2_mrv2_postprocess_does_not_replay_previous_codes() -> None:
    model = _model()
    model._mrv2_active = True
    model._last_audio_codes = torch.tensor([[1, 2]])

    assert model.postprocess(torch.zeros(1, 2)) == {}


def test_higgs_audio_v2_recompute_preemption_fails_closed() -> None:
    assert HIGGS_AUDIO_V2_PIPELINE.stages[0].recompute_preemption == "fail"


def test_higgs_audio_v2_maps_flat_token_positions_to_request_ids() -> None:
    model = _model()
    model._audio_state_uses_request_ids = True
    model._mrv2_req_ids = ["prefill", "decode"]
    model._mrv2_query_start_loc = [0, 3, 4]

    assert model._audio_state_key_for_position(0) == "prefill"
    assert model._audio_state_key_for_position(2) == "prefill"
    assert model._audio_state_key_for_position(3) == "decode"


def test_higgs_audio_v2_prepare_runner_inputs_refreshes_compacted_layout() -> None:
    model = _model()
    input_ids = object()
    positions = object()

    result = model.prepare_runner_inputs(
        req_ids=["second", "first"],
        query_start_loc=[0, 1, 5],
        input_ids=input_ids,
        positions=positions,
    )

    assert result == (input_ids, positions)
    assert model._mrv2_req_ids == ["second", "first"]
    assert model._mrv2_query_start_loc == [0, 1, 5]
    assert model._audio_state_uses_request_ids is True


def test_higgs_audio_v2_model_state_forwards_authoritative_batch_layout(mocker) -> None:
    state = object.__new__(HiggsAudioV2ModelState)
    state.model = mocker.Mock()
    batch = _StateBatch(
        num_reqs=2,
        req_ids=["second", "first", "stale"],
        query_start_loc_np=np.array([0, 1, 5, 99]),
        input_ids=torch.arange(5),
        positions=torch.arange(5),
    )

    parent_prepare = mocker.patch.object(OmniModelState, "prepare_inputs", return_value={"ready": True})
    result = state.prepare_inputs(batch, req_states=object())

    assert result == {"ready": True}
    parent_prepare.assert_called_once()
    state.model.prepare_runner_inputs.assert_called_once()
    kwargs = state.model.prepare_runner_inputs.call_args.kwargs
    assert kwargs["req_ids"] == ["second", "first"]
    assert kwargs["query_start_loc"].tolist() == [0, 1, 5]
    assert kwargs["input_ids"] is batch.input_ids
    assert kwargs["positions"] is batch.positions


def test_higgs_audio_v2_model_state_forwards_request_cleanup(mocker) -> None:
    state = object.__new__(HiggsAudioV2ModelState)
    state.model = mocker.Mock()

    parent_cleanup = mocker.patch.object(OmniModelState, "on_requests_finished")
    state.on_requests_finished({"finished"})

    parent_cleanup.assert_called_once_with({"finished"})
    state.model.on_requests_finished.assert_called_once_with({"finished"})


def test_higgs_audio_v2_releases_request_keyed_audio_history() -> None:
    model = _model()
    model._audio_state = {"finished": {"audio_out_ids": object()}, "live": {}}
    model._slot_output_len = {"finished": 7, "live": 2}

    model.on_requests_finished({"finished"})

    assert model._audio_state == {"live": {}}
    assert model._slot_output_len == {"live": 2}
