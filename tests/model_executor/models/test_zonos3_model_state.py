# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native ModelState with real Backbone/sampler math and CPU-owned inputs."""

from dataclasses import replace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.config import ModelConfig, SchedulerConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import SamplingStates
from vllm.v1.worker.gpu.states import RequestState

from tests.model_executor.models.test_zonos3_modeling import CPUAttentionOracle, small_ar_config, valid_runtime
from vllm_omni.core.sched.output import OmniNewRequestData
from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.model_executor.models.zonos3.ar_config import ZONOS3_PROMPT_KEY
from vllm_omni.model_executor.models.zonos3.model_state import Zonos3ARModelState
from vllm_omni.model_executor.models.zonos3.modeling_zonos3_mini import Zonos3Backbone
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def owner(monkeypatch):
    monkeypatch.setattr("vllm.v1.worker.gpu.buffer_utils.is_uva_available", lambda: False)
    return make_owner()


def make_owner(capacity=4, device="cpu", dtype=torch.float32):
    device = torch.device(device)
    torch.manual_seed(31)
    architecture = small_ar_config()
    model = (
        Zonos3Backbone(architecture, lambda config, index: CPUAttentionOracle(config))
        .eval()
        .to(device=device, dtype=dtype)
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.1)
    config = valid_runtime()
    config.model_config = MagicMock(spec=ModelConfig)
    config.model_config.enable_prompt_embeds = False
    config.model_config.uses_mrope = False
    config.model_config.get_inputs_embeds_size.return_value = architecture.dim
    config.model_config.max_model_len = 128
    config.model_config.dtype = dtype
    config.speculative_config = None
    config.scheduler_config = SchedulerConfig(
        max_model_len=128, is_encoder_decoder=False, max_num_seqs=capacity, max_num_batched_tokens=max(32, capacity)
    )
    config.compilation_config.cudagraph_capture_sizes = [1, 2, 4]
    state = Zonos3ARModelState(config, model, None, device)
    native = Mock(spec=Sampler)
    native.sampling_states = SamplingStates(capacity, 3)
    native.add_request.side_effect = native.sampling_states.add_request
    native.apply_staged_writes.side_effect = native.sampling_states.apply_staged_writes
    native.req_states = MagicMock(spec=RequestState)
    native.req_states.index_to_req_id = {}
    from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor

    native.req_states.prefill_len = UvaBackedTensor(capacity, dtype=torch.int32)
    return state, native, native.req_states


def materialize_codes(snapshot):
    """Use the runner's real generic D2H/partition contract for model snapshots."""
    from vllm_omni.data_entry_keys import unflatten_payload
    from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner

    cpu = snapshot.copy_to_cpu(lambda tensor: tensor.clone())
    n = len(cpu["codes"]["audio"])
    outputs, _ = OmniARModelRunner._build_async_chunk_outputs_from_mm(
        cpu, np.arange(n + 1, dtype=np.int32), np.ones(n, dtype=np.int32), n, n
    )
    return [unflatten_payload(payload) for payload in outputs]


def add(owner, req_id, slot, length):
    state, sampler, requests = owner
    c = state.model.config
    ids = torch.tensor([[c.audio_pad_id] * 10 + [i % c.text_vocab] for i in range(length)])
    payload = {
        "frame_ids": ids,
        "speaker_embedding": torch.randn(c.speaker_embedding_dim),
        "speaker_row": 0,
        "condition_start": 1,
        "condition_buckets": [0, 1, 2],
        "has_audio_prefix": True,
    }
    params = SamplingParams(temperature=0, seed=42)
    request = OmniNewRequestData(
        req_id=req_id,
        prompt_token_ids=[0] * length,
        prefill_token_ids=[0] * length,
        mm_features=[],
        sampling_params=params,
        pooling_params=None,
        block_ids=(),
        num_computed_tokens=0,
        lora_request=None,
        model_intermediate_buffer={ZONOS3_PROMPT_KEY: payload},
    )
    state.add_request(slot, request)
    requests.index_to_req_id[slot] = req_id
    requests.prefill_len.np[slot] = length
    requests.prefill_len.copy_to_uva()
    sampler.add_request(slot, params)
    sampler.apply_staged_writes()
    state.apply_staged_writes()
    return request


def batch(owner, entries, padded=None):
    state, _, _ = owner
    # Entries: (request id, slot, tokens, absolute offset, is_prefill).
    counts = np.array([entry[2] for entry in entries], dtype=np.int32)
    tokens = int(counts.sum())
    padded = padded or tokens
    result = InputBatch.make_dummy(len(entries), tokens, InputBuffers(4, 32, torch.device("cpu")))
    result.req_ids = [entry[0] for entry in entries]
    result.idx_mapping_np = np.array([entry[1] for entry in entries], dtype=np.intp)
    result.idx_mapping = torch.tensor(result.idx_mapping_np)
    result.num_scheduled_tokens = counts
    result.query_start_loc_np = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
    result.query_start_loc = torch.tensor(result.query_start_loc_np)
    result.num_tokens_after_padding = padded
    result.num_reqs_after_padding = padded if np.all(counts == 1) else len(entries)
    result.positions = torch.cat(
        [torch.arange(offset, offset + n) for _, _, n, offset, _ in entries]
        + [torch.zeros(padded - tokens, dtype=torch.int64)]
    )
    result.num_computed_prefill_tokens_np = np.array([entry[3] for entry in entries], dtype=np.int32)
    result.prefill_len_np = np.array([state._lengths[entry[0]] for entry in entries], dtype=np.int32)
    result.is_prefilling_np = np.array([entry[4] for entry in entries], dtype=bool)
    result.has_prefill = bool(result.is_prefilling_np.any())
    result.decode_graph_eligible = bool(np.all(counts == 1))
    result.seq_lens = torch.tensor([entry[3] + entry[2] for entry in entries], dtype=torch.int32)
    result.logits_indices = result.query_start_loc[1:].long() - 1
    return result


def step(owner, data):
    state, sampler, requests = owner
    inputs = state.prepare_inputs(data, requests)
    state.run_preprocess(data, inputs, requests)
    hidden = state.model(inputs["inputs_embeds"], data.positions, inputs["zonos3_batch"])
    state.postprocess_model_output(hidden, data, requests)
    return state.sample_determined_tokens(data, sampler), inputs


@torch.inference_mode()
def test_conditioning_once_chunked_mixed_prefill_feedback_and_static_bucket(owner, monkeypatch):
    state, sampler, _ = owner
    capture_inputs = state.prepare_dummy_inputs(4, 4)
    slot_pointer = capture_inputs["zonos3_batch"].decode_slots.data_ptr()
    add(owner, "a", 0, 6)
    add(owner, "b", 1, 5)
    compiled = []
    original = state._compile_prompt

    def compile_once(req_id, info):
        compiled.append(req_id)
        return original(req_id, info)

    monkeypatch.setattr(state, "_compile_prompt", compile_once)
    first, _ = step(owner, batch(owner, [("a", 0, 2, 0, True), ("b", 1, 5, 0, True)]))
    assert first.num_sampled.tolist() == [0, 1]
    feedback = state.next_embeddings[1].clone()
    second_data = batch(owner, [("a", 0, 4, 2, True), ("b", 1, 1, 5, False)])
    second, inputs = step(owner, second_data)
    assert second.num_sampled.tolist() == [1, 1]
    torch.testing.assert_close(inputs["inputs_embeds"][4], feedback)
    assert compiled == ["a", "b"]
    third_data = batch(owner, [("b", 1, 1, 6, False), ("a", 0, 1, 6, False)], padded=4)
    _, inputs = step(owner, third_data)
    assert inputs["zonos3_batch"] is capture_inputs["zonos3_batch"]
    assert inputs["zonos3_batch"].decode_slots.data_ptr() == slot_pointer
    assert inputs["zonos3_batch"].decode_slots.tolist() == [1, 0, 4, 5]
    assert state.sampling_state.position[:2].tolist() == [2, 3]
    state.on_requests_finished({"a"})
    state.remove_request(0)
    add(owner, "replacement", 0, 5)
    assert "a" not in state._prompts and int(state.sampling_state.position[0]) == 0
    assert all(not entry.window[0].any() for entry in state.gdn_states.values())


@torch.inference_mode()
def test_raw_speaker_encoding_once_across_prefill_chunks_and_cleanup(owner):
    state, _, _ = owner
    vector = torch.arange(state.model.config.speaker_embedding_dim, dtype=torch.float32)
    encoder = Mock(return_value=vector[None])
    state.model.speaker_encoder = encoder
    add(owner, "raw", 0, 6)
    payload = state.intermediate_buffer.buffers[0][ZONOS3_PROMPT_KEY]
    payload.pop("speaker_embedding")
    waveform = torch.zeros(16000)
    payload.update(reference_waveform=waveform, reference_sample_rate=16000)
    step(owner, batch(owner, [("raw", 0, 2, 0, True)]))
    step(owner, batch(owner, [("raw", 0, 4, 2, True)]))
    encoder.assert_called_once_with(waveform, 16000)
    torch.testing.assert_close(state._prompts["raw"].embeddings[0], state.model.speaker_projection(vector))
    state.on_requests_finished({"raw"})
    assert "raw" not in state._prompts


@torch.inference_mode()
def test_raw_speaker_reference_enforces_checkpoint_duration_limit(owner):
    state, _, _ = owner
    state.model.speaker_encoder = Mock()
    add(owner, "long-reference", 0, 6)
    payload = state.intermediate_buffer.buffers[0][ZONOS3_PROMPT_KEY]
    payload.pop("speaker_embedding")
    payload.update(reference_waveform=torch.zeros(30 * 16000 + 1), reference_sample_rate=16000)
    with pytest.raises(ValueError, match="no longer than 30 seconds"):
        state._compile_prompt("long-reference", state.intermediate_buffer.buffers[0])
    state.model.speaker_encoder.assert_not_called()


@torch.inference_mode()
def test_one_token_prefill_graph_keeps_frontier_for_later_chunk(owner):
    state, _, _ = owner
    add(owner, "a", 0, 6)
    first, _ = step(owner, batch(owner, [("a", 0, 1, 0, True)]))
    assert first.num_sampled.item() == 0
    assert all(entry.prefill_tokens[0] == 1 for entry in state.gdn_states.values())
    last, _ = step(owner, batch(owner, [("a", 0, 5, 1, True)]))
    assert last.num_sampled.item() == 1
    assert all(entry.prefill_tokens[0] == 6 for entry in state.gdn_states.values())


@torch.inference_mode()
def test_native_state_rejects_length_mismatch_preemption_and_decode_fallback(owner):
    state, _, _ = owner
    add(owner, "a", 0, 6)
    state.intermediate_buffer.buffers[0][ZONOS3_PROMPT_KEY]["frame_ids"] = torch.zeros((5, 11), dtype=torch.int64)
    with pytest.raises(ValueError, match="placeholder length"):
        step(owner, batch(owner, [("a", 0, 2, 0, True)]))
    with pytest.raises(RuntimeError, match="preemption"):
        state.on_request_preempted("a", 0)
    pure = batch(owner, [("a", 0, 1, 6, False)])
    with pytest.raises(RuntimeError, match="one native FULL"):
        state.prepare_attn(pure, CUDAGraphMode.NONE, (), torch.empty(0), [], None)


@torch.inference_mode()
def test_prompt_conditioning_survives_engine_core_metadata_serialization(owner):
    state, sampler, _ = owner
    original = add(owner, "before", 0, 6)
    payload = state.intermediate_buffer.buffers[0][ZONOS3_PROMPT_KEY]
    serialized = serialize_additional_information({ZONOS3_PROMPT_KEY: payload})
    state.on_requests_finished({"before"})
    state.remove_request(0)
    request = replace(original, req_id="serialized", model_intermediate_buffer=None, additional_information=serialized)
    state.add_request(0, request)
    sampler.add_request(0, request.sampling_params)
    sampler.apply_staged_writes()
    state.apply_staged_writes()
    output, _ = step(owner, batch(owner, [("serialized", 0, 6, 0, True)]))
    assert output.num_sampled.tolist() == [1]
    assert state._prompts["serialized"].embeddings.shape == (6, state.model.config.dim)
    assert bool(state.sampling_state.has_audio_prefix[0])


@torch.inference_mode()
def test_request_owned_codes_survive_mixed_prefill_partition_and_later_steps(owner):
    from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner

    state, native, requests = owner
    add(owner, "partial", 0, 6)
    add(owner, "complete", 1, 5)
    data = batch(owner, [("partial", 0, 2, 0, True), ("complete", 1, 5, 0, True)])
    inputs = state.prepare_inputs(data, requests)
    state.run_preprocess(data, inputs, requests)
    hidden = state.model(inputs["inputs_embeds"], data.positions, inputs["zonos3_batch"])
    _, snapshot = state.postprocess_model_output(hidden, data, requests)
    tokens = state.sample_determined_tokens(data, native)
    assert tokens.num_sampled.tolist() == [0, 1]
    cpu = snapshot.copy_to_cpu(lambda tensor: tensor.clone())
    outputs, _ = OmniARModelRunner._build_async_chunk_outputs_from_mm(
        cpu, data.query_start_loc_np, data.num_scheduled_tokens, 2, 7
    )
    assert [p["codes.audio"].shape for p in outputs] == [(1, 10), (1, 10)]
    assert [p["meta.codec_frame_valid"].item() for p in outputs] == [0, 1]
    saved = outputs[1]["codes.audio"].clone()
    step(owner, batch(owner, [("partial", 0, 4, 2, True), ("complete", 1, 1, 5, False)]))
    assert torch.equal(materialize_codes(snapshot)[1]["codes"]["audio"], saved)


def test_native_startup_decode_warmup_can_run_before_full_graph_capture(owner, monkeypatch):
    state, _, _ = owner
    add(owner, "real", 0, 5)
    data = batch(owner, [("real", 0, 1, 5, False)])
    upstream = Mock(return_value={})
    monkeypatch.setattr(OmniModelState, "prepare_attn", upstream)
    with pytest.raises(RuntimeError, match="one native FULL"):
        state.prepare_attn(data, CUDAGraphMode.NONE, (), torch.empty(0), [], None)
    data.req_ids = ["_warmup_0_"]
    assert state.prepare_attn(data, CUDAGraphMode.NONE, (), torch.empty(0), [], None) == {}
    assert upstream.call_count == 1
