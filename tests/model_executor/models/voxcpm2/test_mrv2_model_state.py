# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression coverage for VoxCPM2's MRV2 slot and output ownership."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.voxcpm2.model_state import (
    VoxCPM2AudioOutput,
    VoxCPM2BatchSlots,
    VoxCPM2ModelState,
)
from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import (
    VoxCPM2TalkerForConditionalGeneration,
    _encode_raw_audio,
    _encode_raw_audio_batch,
    _PrefillInputs,
)
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _talker(device: torch.device) -> VoxCPM2TalkerForConditionalGeneration:
    model = VoxCPM2TalkerForConditionalGeneration.__new__(VoxCPM2TalkerForConditionalGeneration)
    nn.Module.__init__(model)
    model._tts = SimpleNamespace(audio_vae=SimpleNamespace(decode_chunk_size=8))
    model._device = device
    model._patch_size = 4
    model._feat_dim = 2
    model._side_dtype = torch.float32
    model.config = SimpleNamespace(hidden_size=6)
    model._n_decode_pad_frames = 12
    model._vae_decode_every = 1
    model._audio_emit_every = 1
    model._sample_rate = 48000
    model.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=100))
    model._active_states = {}
    model._audio_queue = []
    return model


def _state(
    model: VoxCPM2TalkerForConditionalGeneration, count: int, device: torch.device, monkeypatch: pytest.MonkeyPatch
) -> VoxCPM2ModelState:
    def init_base(owner, _config, talker, _cache, _device) -> None:
        owner.model = talker
        owner.scheduler_config = SimpleNamespace(max_num_seqs=count)

    monkeypatch.setattr(OmniModelState, "__init__", init_base)
    state = VoxCPM2ModelState(None, model, None, device)
    state.intermediate_buffer = SimpleNamespace(req_id_to_index={}, buffers=[{} for _ in range(count)])
    monkeypatch.setattr(
        OmniModelState,
        "add_request",
        lambda owner, index, request: owner.intermediate_buffer.req_id_to_index.update({request.req_id: index}),
    )
    monkeypatch.setattr(OmniModelState, "remove_request", lambda *_args: None)
    monkeypatch.setattr(OmniModelState, "on_request_preempted", lambda *_args: None)
    return state


def _audio_output(
    state: VoxCPM2ModelState, slots: list[int], sampled: list[int], *, max_seq_len: int = 100
) -> list[dict | None]:
    batch = SimpleNamespace(
        num_reqs=len(slots),
        idx_mapping_np=np.asarray(slots),
        idx_mapping=torch.tensor(slots, device=state.audio_buffer.device),
        num_computed_tokens_np=np.zeros(len(slots), dtype=np.int32),
        num_scheduled_tokens=[1] * len(slots),
    )
    req_states = SimpleNamespace(max_seq_len=np.full(len(state.slots), max_seq_len, dtype=np.int32))
    # Production resolves tail stops before VAE collection and forces their
    # sampled stop token. Mirror that host state in these output-only tests.
    for slot, token in zip(slots, sampled, strict=True):
        if token == 1:
            state.slots[slot].is_stopping = True
    return state.prepare_streaming_audio_output(batch, req_states, {"voxcpm2_audio_pending": True}).get_output()


def test_slots_allocate_up_front_and_reuse_without_cross_request_state(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _talker(torch.device("cpu"))
    state = _state(model, 2, torch.device("cpu"), monkeypatch)
    slots = state.slots
    assert slots is not None
    assert len(slots) == 2
    assert state.audio_buffer.shape == (2, 32)
    assert state.vae_batch_input.shape == (2, 2, 16)

    state.add_request(0, SimpleNamespace(req_id="first"))
    slot = model._active_states["first"]
    slot.decode_step_count = 7
    state.append_pending_latent(slot, torch.ones(4, 2))
    state.remove_request(0)
    state.add_request(0, SimpleNamespace(req_id="second"))
    assert model._active_states["second"] is slot
    assert slot.decode_step_count == 0
    assert slot.pending_vae_count == 0
    assert state.audio_buffer.shape == (2, 32)
    with pytest.raises(RuntimeError, match="still owned"):
        state.add_request(0, SimpleNamespace(req_id="third"))


def test_pending_batch_preserves_reordered_slots_and_individual_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _talker(torch.device("cpu"))
    model._vae_decode_every = 3
    state = _state(model, 3, torch.device("cpu"), monkeypatch)
    for slot in range(3):
        state.add_request(slot, SimpleNamespace(req_id=f"req-{slot}"))
    state.pending_latents.fill_(-1)
    state.prefix_feat[2].fill_(3)
    state.prefix_feat[0].fill_(1)
    state.slots[2].pending_vae_count = 2
    state.append_pending_batch([state.slots[2], state.slots[0]])
    assert state.pending_latents[2, 2].tolist() == [[3.0] * 2] * 4
    assert state.pending_latents[0, 0].tolist() == [[1.0] * 2] * 4
    assert state.pending_latents[1].eq(-1).all()
    assert [owner.pending_vae_count for owner in state.slots] == [1, 0, 3]
    with pytest.raises(RuntimeError, match="slot is full"):
        state.append_pending_batch([state.slots[2], state.slots[0]])
    assert state.slots[0].pending_vae_count == 1


@pytest.mark.parametrize("pending_patch", [False, True])
def test_batched_decode_state_scatter_and_preemption_restore(
    monkeypatch: pytest.MonkeyPatch, pending_patch: bool
) -> None:
    model = _talker(torch.device("cpu"))
    state = _state(model, 2, torch.device("cpu"), monkeypatch)
    state.add_request(0, SimpleNamespace(req_id="first"))
    state.add_request(1, SimpleNamespace(req_id="second"))
    states = [state.slots[1], state.slots[0]]
    state.store_decode_batch(
        states,
        torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.bfloat16),
        torch.stack([torch.full((6,), 2.0), torch.full((6,), 1.0)]).to(torch.bfloat16),
        torch.stack([torch.full((4, 2), 2.0), torch.full((4, 2), 1.0)]).to(torch.bfloat16),
    )
    assert state.next_embed[:, 0].tolist() == [1.0, 2.0]
    assert state.prefix_feat[:, 0, 0].tolist() == [1.0, 2.0]
    assert state.stop_logits.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert state.slots[0].curr_embed_for_next is None
    assert state.tensor_for(state.slots[0], "curr_embed_for_next").tolist() == [[1.0] * 6]
    assert state.gather_embeddings([state.slots[1], state.slots[0]])[:, 0].tolist() == [2.0, 1.0]
    assert state.tensor_for(state.slots[0], "last_audio_patch_gpu") is not None
    state.clear_tensor(state.slots[0], "last_audio_patch_gpu")
    assert state.tensor_for(state.slots[0], "last_audio_patch_gpu") is None
    state.slots[0].audio_patch_ready = pending_patch
    state.append_pending_latent(state.slots[0], torch.full((4, 2), 3.0))

    state.on_request_preempted("first", 0)
    parked = state.suspended["first"]
    if pending_patch:
        assert parked.last_audio_patch_gpu.data_ptr() == parked.curr_prefix_feat_cond.data_ptr()
    state.remove_request(0)
    state.add_request(0, SimpleNamespace(req_id="third"))
    state.store_decode_state(
        state.slots[0], torch.tensor([[9.0, 8.0]]), torch.full((1, 6), 9.0), torch.full((1, 4, 2), 9.0)
    )
    state.remove_request(1)
    state.add_request(1, SimpleNamespace(req_id="first"))
    assert state.slots[1].curr_embed_for_next is None
    assert state.tensor_for(state.slots[1], "curr_embed_for_next").tolist() == [[1.0] * 6]
    assert state.tensor_for(state.slots[1], "curr_prefix_feat_cond")[0, 0].item() == 1.0
    assert state.slots[1].audio_patch_ready is pending_patch
    assert state.slots[1].pending_vae_count == 1
    assert state.pending_latents[1, 0].tolist() == [[3.0, 3.0]] * 4
    assert state.stop_logits[1].tolist() == [1.0, 0.0]


def test_batch_slot_indices_reuse_runner_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _talker(torch.device("cpu"))
    state = _state(model, 2, torch.device("cpu"), monkeypatch)
    state.add_request(0, SimpleNamespace(req_id="first"))
    state.add_request(1, SimpleNamespace(req_id="second"))
    runner_indices = torch.tensor([1, 0], dtype=torch.int32)
    monkeypatch.setattr(OmniModelState, "prepare_inputs", lambda *_args: {})
    model_inputs = state.prepare_inputs(
        SimpleNamespace(
            req_ids=["second", "first"], idx_mapping=runner_indices, idx_mapping_np=np.array([1, 0]), num_reqs=2
        ),
        None,
    )
    monkeypatch.setattr(state, "_device_indices", lambda _values: pytest.fail("unnecessary index H2D"))
    context = VoxCPM2BatchSlots(model_inputs["batch_slots"], model_inputs["batch_slot_rows"])
    selected = state.indices_for([state.slots[1], state.slots[0]], context)
    assert not hasattr(state, "batch_slot_indices")
    assert selected.dtype == torch.long
    assert selected.tolist() == [1, 0]


def test_mrv2_hot_path_uses_bound_slot_without_request_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _talker(torch.device("cpu"))
    state = _state(model, 2, torch.device("cpu"), monkeypatch)
    state.add_request(1, SimpleNamespace(req_id="first"))
    owner = state.slots[1]
    state.intermediate_buffer.req_id_to_index.clear()
    state.store_decode_state(owner, torch.tensor([[0.0, 1.0]]), torch.ones(6), torch.ones(4, 2))
    assert state.tensor_for(owner, "curr_embed_for_next").tolist() == [[1.0] * 6]
    model._audio_queue = [(owner, torch.tensor([3.0]))]
    state.make_audio_output(torch.zeros(1, 6), ["first"])
    assert state.audio_buffer[1, 0].item() == 3.0
    state.remove_request(1)
    assert owner.slot_index is None
    with pytest.raises(RuntimeError, match="no runner slot"):
        state.tensor_for(owner, "curr_embed_for_next")


def test_compute_logits_batches_mixed_slot_outcomes(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _talker(torch.device("cpu"))
    model.config.vocab_size = 5
    state = _state(model, 5, torch.device("cpu"), monkeypatch)
    for slot in range(5):
        state.add_request(slot, SimpleNamespace(req_id=f"req-{slot}"))
    for slot, values in ((4, [0.2, 0.8]), (1, [0.9, 0.1]), (3, [0.3, 0.7])):
        state.store_decode_state(state.slots[slot], torch.tensor([values]), torch.zeros(6), torch.zeros(4, 2))
    state.slots[1].is_stopping = True
    state.slots[2].prefill_completed = True
    state.slots[3].precomputed_is_stopping = True
    model._results_queue = [
        (
            state.slots[slot],
            state.tensor_for(state.slots[slot], "precomputed_stop_logits") if slot in (4, 1, 3) else None,
        )
        for slot in (4, 1, 2, 0, 3)
    ]
    state.intermediate_buffer.req_id_to_index.clear()
    logits = model.compute_logits(torch.zeros(5, 6))
    torch.testing.assert_close(
        logits[:, :2],
        torch.tensor([[0.2, 0.8], [0.0, 1.0], [float("-inf"), 1.0], [1.0, float("-inf")], [0.3, 0.7]]),
    )
    assert torch.isneginf(logits[:, 2:]).all()
    assert state.slots[3].is_stopping
    assert all(not slot.stop_logits_ready for slot in (state.slots[4], state.slots[1], state.slots[3]))
    assert model._results_queue == []
    model._results_queue = [(state.slots[0], None)]
    short = model.compute_logits(torch.zeros(2, 6))
    assert short[0, 0] == 1.0
    assert torch.isneginf(short[1]).all()
    empty = model.compute_logits(torch.zeros(2, 6))
    assert empty[:, 0].tolist() == [1.0, 1.0]


def test_compute_logits_full_decode_batch_preserves_row_order(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _talker(torch.device("cpu"))
    model.config.vocab_size = 5
    state = _state(model, 3, torch.device("cpu"), monkeypatch)
    for index in range(3):
        state.add_request(index, SimpleNamespace(req_id=f"req-{index}"))
        state.store_decode_state(
            state.slots[index], torch.tensor([[float(index), float(index + 1)]]), torch.zeros(6), torch.zeros(4, 2)
        )
    model._results_queue = [
        (state.slots[index], state.tensor_for(state.slots[index], "precomputed_stop_logits")) for index in (2, 0, 1)
    ]
    logits = model.compute_logits(torch.zeros(3, 6))
    torch.testing.assert_close(logits[:, :2], torch.tensor([[2.0, 3.0], [0.0, 1.0], [1.0, 2.0]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("vocab_size", [5, 259])
def test_cuda_stop_logits_match_legacy_path(monkeypatch, dtype, vocab_size) -> None:
    model = _talker(torch.device("cuda"))
    model.config.vocab_size = vocab_size
    state = _state(model, 5, torch.device("cuda"), monkeypatch)
    for slot in range(5):
        state.add_request(slot, SimpleNamespace(req_id=f"req-{slot}"))

    def prepare(order):
        for owner in state.slots:
            owner.is_stopping = False
            owner.prefill_completed = False
            owner.precomputed_is_stopping = None
        for slot, values in ((4, [0.2, 0.8]), (1, [0.9, 0.1]), (3, [0.3, 0.7])):
            state.store_decode_state(
                state.slots[slot],
                torch.tensor([values], device="cuda"),
                torch.zeros(6, device="cuda"),
                torch.zeros(4, 2, device="cuda"),
            )
        state.slots[1].is_stopping = True
        state.slots[2].prefill_completed = True
        state.slots[3].precomputed_is_stopping = True
        model._results_queue = [
            (
                state.slots[slot],
                state.tensor_for(state.slots[slot], "precomputed_stop_logits") if slot in (4, 1, 3) else None,
            )
            for slot in order
        ]

    for order in ((4, 1, 2, 0, 3), (4,), ()):
        hidden = torch.zeros(5, 6, device="cuda", dtype=dtype)
        prepare(order)
        # Run the retained per-row implementation without switching storage ownership.
        expected = model.compute_logits(hidden.cpu()).cuda()
        expected_states = [(s.is_stopping, s.stop_logits_ready, s.precomputed_is_stopping) for s in state.slots]
        prepare(order)
        with monkeypatch.context() as patch:
            patch.setattr(torch, "cat", lambda *_args, **_kwargs: pytest.fail("unexpected logits concatenation"))
            actual = model.compute_logits(hidden)
        torch.testing.assert_close(actual, expected)
        assert [(s.is_stopping, s.stop_logits_ready, s.precomputed_is_stopping) for s in state.slots] == expected_states
        assert model._results_queue == []


def test_noncontiguous_batch_rows_select_runner_slots_in_request_order(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _talker(torch.device("cpu"))
    state = _state(model, 3, torch.device("cpu"), monkeypatch)
    for index, req_id in enumerate(("first", "second", "third")):
        state.add_request(index, SimpleNamespace(req_id=req_id))
    batch_slots = torch.tensor([2, 0, 1], dtype=torch.int32)
    batch_rows = {2: 0, 0: 1, 1: 2}
    selected = state.indices_for([state.slots[0], state.slots[2]], VoxCPM2BatchSlots(batch_slots, batch_rows))
    assert selected.dtype == torch.long
    assert selected.tolist() == [0, 2]


def test_partial_audio_emits_on_sampled_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _talker(torch.device("cpu"))
    state = _state(model, 2, torch.device("cpu"), monkeypatch)
    state.add_request(0, SimpleNamespace(req_id="first"))
    state.add_request(1, SimpleNamespace(req_id="second"))
    model._audio_queue = [("first", torch.arange(17, dtype=torch.float32))]

    model.make_omni_output(torch.zeros(2, 4), request_ids=["second", "first"])
    assert _audio_output(state, [1, 0], [0, 0]) == [None, None]
    output = _audio_output(state, [1, 0], [0, 1])
    assert output[0] is None
    assert output[1]["model_outputs"].tolist() == list(range(17))
    assert model._audio_queue == []


def test_full_chunk_emits_and_then_accumulates_next_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _talker(torch.device("cpu"))
    model._audio_emit_every = 2
    state = _state(model, 2, torch.device("cpu"), monkeypatch)
    state.add_request(0, SimpleNamespace(req_id="first"))
    state.add_request(1, SimpleNamespace(req_id="second"))
    first_patch = torch.arange(32, dtype=torch.bfloat16)
    model._audio_queue = [("first", first_patch)]
    model.make_omni_output(torch.zeros(2, 4), request_ids=["second", "first"])
    assert _audio_output(state, [1, 0], [0, 0]) == [None, None]
    model._audio_queue = [("first", torch.arange(32, 64, dtype=torch.float32))]
    model.make_omni_output(torch.zeros(2, 4), request_ids=["second", "first"])
    output = _audio_output(state, [1, 0], [0, 0])
    assert output[0] is None
    assert output[1]["model_outputs"].tolist() == list(range(64))
    assert state.audio_output_lengths[0] == 0
    model._audio_queue = [("first", torch.tensor([7.0, 8.0]))]
    model.make_omni_output(torch.zeros(2, 4), request_ids=["second", "first"])
    assert _audio_output(state, [1, 0], [0, 1])[1]["model_outputs"].tolist() == [7.0, 8.0]


def test_length_limit_flushes_partial_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _talker(torch.device("cpu"))
    model._audio_emit_every = 2
    state = _state(model, 1, torch.device("cpu"), monkeypatch)
    state.add_request(0, SimpleNamespace(req_id="first"))
    model._audio_queue = [("first", torch.tensor([2.0, 3.0]))]
    model.make_omni_output(torch.zeros(1, 4), request_ids=["first"])
    assert _audio_output(state, [0], [0], max_seq_len=2)[0]["model_outputs"].tolist() == [2.0, 3.0]


@pytest.mark.parametrize(
    "tail_patches,batched,audio_emit_every,limit_source",
    [
        (1, False, 1, "request"),
        (2, False, 2, "model"),
        (1, True, 1, "request"),
        (2, True, 1, "model"),
        (1, True, 2, "request"),
        (2, True, 2, "model"),
    ],
)
def test_length_limit_decodes_pending_latent_tail(
    monkeypatch: pytest.MonkeyPatch, tail_patches: int, batched: bool, audio_emit_every: int, limit_source: str
) -> None:
    model = _talker(torch.device("cpu"))
    model._vae_decode_every = 3
    model._audio_emit_every = audio_emit_every
    model._enable_batched_vae_decode = batched
    model._enable_delayed_audio_copy = False
    model._enable_vae_cuda_graph = False
    model._coalesce_audio_d2h = True
    model._perf = SimpleNamespace(start=lambda *_: None, stop=lambda *_: None)
    state = _state(model, 2, torch.device("cpu"), monkeypatch)
    for slot, req_id in enumerate(("running", "ending")):
        state.add_request(slot, SimpleNamespace(req_id=req_id))
        for _ in range(tail_patches - 1):
            state.append_pending_latent(state.slots[slot], torch.ones(4, 2))
        # The model predicts continuation even on the last allowed step.
        state.store_decode_state(state.slots[slot], torch.tensor([[1.0, 0.0]]), torch.ones(1, 6), torch.ones(1, 4, 2))

    batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping_np=np.array([1, 0]),
        idx_mapping=torch.tensor([1, 0]),
        num_computed_tokens_np=np.array([8, 7]),
        num_scheduled_tokens=[1, 1],
    )
    req_states = SimpleNamespace(max_seq_len=np.array([100, 10 if limit_source == "request" else 100]))
    model.vllm_config.model_config.max_model_len = 10 if limit_source == "model" else 100
    calls = []

    def decode(feat: torch.Tensor) -> torch.Tensor:
        calls.append(feat.shape)
        return feat.sum(1).repeat_interleave(8, dim=1).unsqueeze(1)

    model._run_vae_decode = decode
    state.prepare_audio_length_limits(batch, req_states)
    audio = model._collect_audio_batch([state.slots[1], state.slots[0]])
    model._audio_queue = [(state.slots[slot], chunk) for slot, chunk in audio.items()]
    state.make_audio_output(torch.zeros(2, 6), ["ending", "running"])
    output = state.prepare_streaming_audio_output(batch, req_states, {"voxcpm2_audio_pending": True}).get_output()

    assert calls == [(1, 2, tail_patches * 4)]
    assert output[0]["model_outputs"].tolist() == [2.0] * (tail_patches * 32)
    assert output[1] is None
    assert state.slots[1].pending_vae_count == 0
    assert state.slots[0].pending_vae_count == tail_patches
    assert not state.slots[1].is_stopping
    assert state.slots[1].pending_audio_chunks_gpu == []


def test_chunk_output_copies_audio_and_valid_lengths_together() -> None:
    copied = []

    def copy_tensor(value: torch.Tensor) -> torch.Tensor:
        copied.append(value)
        return value.clone()

    output = VoxCPM2AudioOutput(torch.arange(3), torch.tensor([3]), torch.tensor(48000), (0,), 2)
    host = output.to_cpu(None, copy_tensor)
    assert copied[0] is output.wav
    assert len(copied) == 1
    assert host.valid_samples is output.valid_samples
    assert host.get_output()[0]["model_outputs"].tolist() == [0, 1, 2]
    assert host.get_output()[1] is None


def test_audio_free_step_does_not_gather_or_copy_pcm(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _talker(torch.device("cpu"))
    model._audio_emit_every = 2
    state = _state(model, 2, torch.device("cpu"), monkeypatch)
    for slot in range(2):
        state.add_request(slot, SimpleNamespace(req_id=f"req-{slot}"))
    # Even buffered PCM stays on the device until its emission boundary.
    state.audio_output_lengths[0] = 32
    monkeypatch.setattr(torch.Tensor, "index_select", lambda *_: pytest.fail("unexpected PCM gather"))
    monkeypatch.setattr(torch, "cat", lambda *_: pytest.fail("unexpected PCM packing"))
    batch = SimpleNamespace(
        num_reqs=2, idx_mapping_np=np.array([1, 0]), num_computed_tokens_np=np.zeros(2), num_scheduled_tokens=[1, 1]
    )
    output = state.prepare_streaming_audio_output(batch, SimpleNamespace(max_seq_len=np.array([100, 100])), {})
    host = output.to_cpu(None, lambda *_: pytest.fail("unexpected D2H copy"))
    assert host.get_output() == [None, None]
    assert state.audio_output_lengths.tolist() == [32, 0]


def test_audio_copy_packs_only_ready_samples_in_batch_order(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _talker(torch.device("cpu"))
    model._audio_emit_every = 2
    state = _state(model, 3, torch.device("cpu"), monkeypatch)
    for slot in range(3):
        state.add_request(slot, SimpleNamespace(req_id=f"req-{slot}"))
    state.audio_buffer[0, :64].copy_(torch.arange(64))
    state.audio_buffer[1, :17].fill_(9)
    state.audio_buffer[2, :5].fill_(7)
    state.audio_output_lengths[:] = [64, 17, 5]
    state.slots[2].is_stopping = True
    batch = SimpleNamespace(
        num_reqs=3, idx_mapping_np=np.array([2, 1, 0]), num_computed_tokens_np=np.zeros(3), num_scheduled_tokens=[1] * 3
    )
    output = state.prepare_streaming_audio_output(batch, SimpleNamespace(max_seq_len=np.full(3, 100)), {})
    copied = []

    def copy(value):
        copied.append(value.numel())
        return value.clone()

    host = output.to_cpu(None, copy)
    # Packed PCM owns its storage across later slot reuse.
    state.audio_buffer.fill_(-1)
    payload = host.get_output()
    assert copied == [69]
    assert payload[0]["model_outputs"].tolist() == [7.0] * 5
    assert payload[1] is None
    assert payload[2]["model_outputs"].tolist() == list(range(64))
    assert state.audio_output_lengths.tolist() == [0, 17, 0]


def test_partial_chunk_follows_request_to_new_slot_after_preemption(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _talker(torch.device("cpu"))
    model._audio_emit_every = 2
    state = _state(model, 2, torch.device("cpu"), monkeypatch)
    state.add_request(0, SimpleNamespace(req_id="first"))
    model._audio_queue = [("first", torch.arange(32, dtype=torch.float32))]
    model.make_omni_output(torch.zeros(1, 4), request_ids=["first"])
    assert _audio_output(state, [0], [0]) == [None]
    state.on_request_preempted("first", 0)
    state.remove_request(0)
    state.add_request(1, SimpleNamespace(req_id="first"))
    model._audio_queue = [("first", torch.arange(32, 64, dtype=torch.float32))]
    model.make_omni_output(torch.zeros(1, 4), request_ids=["first"])
    assert _audio_output(state, [1], [0])[0]["model_outputs"].tolist() == list(range(64))


@pytest.mark.parametrize(
    "device_name",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"))],
)
def test_ready_slots_same_total_frames_keep_different_boundaries(
    monkeypatch: pytest.MonkeyPatch, device_name: str
) -> None:
    device = torch.device(device_name)
    model = _talker(device)
    model._vae_decode_every = 3
    model._audio_emit_every = 2
    state = _state(model, 2, device, monkeypatch)
    for index in range(2):
        state.add_request(index, SimpleNamespace(req_id=f"req-{index}"))
    state.decode_pad[0, :12].fill_(9)
    state.decode_pad[1, :4].fill_(7)
    state.pending_latents[0, :1].fill_(1)
    state.pending_latents[1, :3].fill_(2)
    calls = []

    def decode(feat):
        calls.append(feat.clone())
        return feat.sum(1).repeat_interleave(8, dim=1).unsqueeze(1)

    model._run_vae_decode = decode
    state.audio_buffer.fill_(-1)
    state.audio_output_lengths[:] = [3, 5]
    state.decode_ready_slots([(state.slots[1], 4, 12), (state.slots[0], 12, 4)])
    assert len(calls) == 1
    assert calls[0].shape == (2, 2, 16)
    assert calls[0][0, 0].tolist() == [7.0] * 4 + [2.0] * 12
    assert calls[0][1, 0].tolist() == [9.0] * 12 + [1.0] * 4
    assert state.audio_output_lengths.tolist() == [35, 101]
    assert state.audio_buffer[0, :35].tolist() == [-1.0] * 3 + [2.0] * 32
    assert state.audio_buffer[1, :101].tolist() == [-1.0] * 5 + [4.0] * 96
    assert state.audio_buffer[0, 35:].eq(-1).all()
    assert state.audio_buffer[1, 101:].eq(-1).all()
    tail = min(16, state.decode_pad.shape[1])
    torch.testing.assert_close(state.decode_pad[1, :tail], calls[0][0, :, -tail:].T)
    torch.testing.assert_close(state.decode_pad[0, :tail], calls[0][1, :, -tail:].T)


def test_ready_slot_tail_appends_pcm_and_updates_context(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _talker(torch.device("cpu"))
    model._vae_decode_every = 3
    state = _state(model, 2, torch.device("cpu"), monkeypatch)
    state.add_request(1, SimpleNamespace(req_id="tail"))
    owner = state.slots[1]
    state.decode_pad[1, :2].fill_(9)
    owner.decode_pad_len = 2
    state.append_pending_latent(owner, torch.ones(4, 2))
    state.audio_buffer[1, :3].fill_(7)
    state.audio_output_lengths[1] = 3
    captured = []

    def decode(feat):
        captured.append(feat.clone())
        return feat.sum(1).repeat_interleave(8, dim=1).unsqueeze(1)

    model._run_vae_decode = decode
    state.decode_ready_slots([(owner, 2, 4)])
    assert captured[0][0].tolist() == [[9.0, 9.0, 1.0, 1.0, 1.0, 1.0]] * 2
    assert state.audio_buffer[1, :35].tolist() == [7.0] * 3 + [2.0] * 32
    assert owner.pending_vae_count == 0
    assert owner.decode_pad_len == 6
    assert state.decode_pad[1, :6].tolist() == [[9.0] * 2] * 2 + [[1.0] * 2] * 4


def test_per_patch_decode_does_not_require_stop_check(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _talker(torch.device("cpu"))
    model._enable_batched_vae_decode = True
    model._enable_delayed_audio_copy = False
    model._enable_vae_cuda_graph = False
    model._coalesce_audio_d2h = True
    state = _state(model, 3, torch.device("cpu"), monkeypatch)
    for index, req_id in enumerate(("running", "stopping", "other")):
        state.add_request(index, SimpleNamespace(req_id=req_id))
    patch = torch.ones(1, 4, 2)
    for slot in state.slots:
        state.store_decode_state(slot, torch.tensor([[1.0, 0.0]]), torch.ones(1, 6), patch)
    state.store_decode_state(state.slots[1], torch.tensor([[0.0, 1.0]]), torch.ones(1, 6), patch)
    calls = []

    def decode(feat: torch.Tensor) -> torch.Tensor:
        calls.append(feat.shape)
        return feat.sum(1).repeat_interleave(8, dim=1).unsqueeze(1)

    model._run_vae_decode = decode
    monkeypatch.setattr(torch.Tensor, "cpu", lambda *_args, **_kwargs: pytest.fail("unexpected stop-mask D2H"))
    model._precompute_stop_flags_for_audio_collect(state.slots)
    output = model._collect_audio_batch(state.slots)
    assert calls == [(3, 2, 4)]
    assert all(value is None for value in output.values())
    assert state.audio_output_lengths.tolist() == [32, 32, 32]
    assert state.audio_buffer.tolist() == [[2.0] * 32] * 3
    assert all(slot.pending_vae_count == 0 for slot in state.slots)


def test_batched_audio_uses_slots_for_reordered_cohort(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _talker(torch.device("cpu"))
    model._enable_batched_vae_decode = True
    model._enable_delayed_audio_copy = False
    model._enable_vae_cuda_graph = False
    model._coalesce_audio_d2h = True
    state = _state(model, 3, torch.device("cpu"), monkeypatch)
    for slot_index, req_id in enumerate(("first", "second", "third")):
        state.add_request(slot_index, SimpleNamespace(req_id=req_id))
    for index in (2, 0):
        state.store_decode_state(
            state.slots[index], torch.tensor([[1.0, 0.0]]), torch.ones(6), torch.full((4, 2), index + 1.0)
        )
    model._run_vae_decode = lambda feat: feat.sum(1).repeat_interleave(8, dim=1).unsqueeze(1)
    state.intermediate_buffer.req_id_to_index.clear()
    cohort = [state.slots[2], state.slots[0]]
    batch_context = VoxCPM2BatchSlots(torch.tensor([2, 1, 0]), {2: 0, 1: 1, 0: 2})
    output = model._collect_audio_batch(cohort, batch_context=batch_context)
    assert list(output) == [2, 0]
    assert output[2] is None and output[0] is None
    assert state.audio_buffer[2].tolist() == [6.0] * 32
    assert state.audio_buffer[0].tolist() == [2.0] * 32
    assert state.slots[2].decode_pad_len == state.slots[0].decode_pad_len == 4
    assert state.decode_pad[2, :4].tolist() == [[3.0, 3.0]] * 4
    assert state.decode_pad[0, :4].tolist() == [[1.0, 1.0]] * 4


def test_decode_batch_restores_mixed_prefill_batch_order(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _talker(torch.device("cpu"))
    model._pending_requests = []
    state = _state(model, 3, torch.device("cpu"), monkeypatch)
    for slot_index, req_id in enumerate(("prefill", "unready", "ready")):
        state.add_request(slot_index, SimpleNamespace(req_id=req_id))
        state.intermediate_buffer.buffers[slot_index]["req_id"] = req_id
    state.next_embed[2].fill_(7)
    state.slots[2].decode_state_ready = True

    def run_generic(_owner, _batch, _inputs, _req_states, _dispatcher):
        infos = [state.intermediate_buffer.buffers[slot] for slot in (2, 1)]
        _ids, embeds, _hidden, _text_step, _updates = model.preprocess_decode_batch_mrv2(
            input_ids=torch.zeros(2, dtype=torch.long), input_embeds=torch.zeros(2, 6), req_infos=infos
        )
        assert embeds.tolist() == [[7.0] * 6, [0.0] * 6]
        model._pending_requests.append((state.slots[0], True, None, 3))
        model._pending_batch_rows.append(state.intermediate_buffer.buffers[0]["batch_row"])

    monkeypatch.setattr(OmniModelState, "run_preprocess", run_generic)
    state.run_preprocess(
        SimpleNamespace(idx_mapping_np=np.array([2, 0, 1]), idx_mapping=torch.tensor([2, 0, 1]), num_reqs=3), {}
    )
    assert state.preprocess_batch_slots is None
    assert [request.request_id for request, *_ in model._pending_requests] == ["ready", "prefill", "unready"]
    assert model._pending_batch_rows is None
    assert all("batch_row" not in info for info in state.intermediate_buffer.buffers)


def test_prefill_batch_embeds_zero_shot_once_and_cleans_scratch(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _talker(torch.device("cpu"))
    model._pending_requests = []
    state = _state(model, 3, torch.device("cpu"), monkeypatch)
    model.config.bos_token_id = 1
    model._tts.audio_start_token = 9
    monkeypatch.setattr(model, "_get_multichar_zh_split", lambda: {})
    calls = []

    def embed(ids):
        calls.append(ids.clone())
        return ids.float().unsqueeze(-1).expand(-1, 6)

    model.model = SimpleNamespace(embed_input_ids=embed)
    for index, (req_id, tokens) in enumerate((("first", [1, 2]), ("voice", [1, 3]), ("third", [1, 4, 5]))):
        state.add_request(index, SimpleNamespace(req_id=req_id))
        state.intermediate_buffer.buffers[index].update(req_id=req_id, text_token_ids=[tokens])
    state.intermediate_buffer.buffers[1]["voice_name"] = "cached-voice"

    def run_generic(owner, _batch, _inputs, _req_states, _dispatcher):
        infos = owner.intermediate_buffer.buffers
        model.preprocess_batch_mrv2(req_infos=infos, device=torch.device("cpu"))
        assert infos[0]["prefill_text_embed"][:, 0].tolist() == [2.0, 9.0]
        assert infos[2]["prefill_text_embed"][:, 0].tolist() == [4.0, 5.0, 9.0]
        assert "prefill_text_embed" not in infos[1]

    monkeypatch.setattr(OmniModelState, "run_preprocess", run_generic)
    state.run_preprocess(
        SimpleNamespace(idx_mapping_np=np.array([0, 1, 2]), idx_mapping=torch.tensor([0, 1, 2]), num_reqs=3), {}
    )
    assert [ids.tolist() for ids in calls] == [[2, 9, 4, 5, 9]]
    assert all("prefill_text_embed" not in info for info in state.intermediate_buffer.buffers)


def test_prefill_consumes_batched_text_without_reencoding(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _talker(torch.device("cpu"))
    model._pending_requests = []
    model._pending_batch_rows = []
    model.config.bos_token_id = 1
    state = _state(model, 1, torch.device("cpu"), monkeypatch)
    state.add_request(0, SimpleNamespace(req_id="first"))
    monkeypatch.setattr(model, "_get_multichar_zh_split", lambda: {})
    model.model = SimpleNamespace(embed_input_ids=lambda *_: pytest.fail("duplicate text embedding"))
    model._tts = SimpleNamespace(
        feat_encoder=lambda *_: pytest.fail("masked audio should not be encoded"),
        enc_to_lm_proj=lambda *_: pytest.fail("masked audio should not be projected"),
    )
    monkeypatch.setattr(
        model,
        "_build_prefill_inputs",
        lambda *_args, **_kwargs: _PrefillInputs(
            text_token=torch.tensor([[2, 9]], dtype=torch.int32),
            audio_feat=torch.zeros(1, 2, 4, 2),
            text_mask=torch.ones(1, 2, dtype=torch.int32),
            audio_mask=torch.zeros(1, 2, dtype=torch.int32),
        ),
    )
    _ids, embeds, _updates = model.preprocess(
        torch.tensor([2, 9]),
        None,
        req_id="first",
        slot_index=0,
        batch_row=0,
        _omni_is_prefill=True,
        text_token_ids=[[1, 2]],
        prefill_text_embed=torch.full((2, 6), 5.0),
    )
    assert embeds.tolist() == [[5.0] * 6] * 2
    assert model._pending_batch_rows == [0]


def test_voice_clone_batched_encode_and_embedding_match_individual(monkeypatch: pytest.MonkeyPatch) -> None:
    class AudioVAE(nn.Module):
        latent_dim = 2

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))
            self.batch_sizes = []

        def encode(self, audio, sr):
            assert sr == 8000
            self.batch_sizes.append(audio.shape[0])
            return audio.unsqueeze(1).expand(-1, 2, -1).contiguous()

    vae = AudioVAE()
    tts = SimpleNamespace(audio_vae=vae, _encode_sample_rate=8000, patch_size=4, chunk_size=1, audio_start_token=9)
    wavs = [([1.0, 2.0, 3.0], 8000, "right"), ([4.0, 5.0, 6.0, 7.0], 8000, "left"), ([8.0] * 5, 8000, "right")]
    batched = _encode_raw_audio_batch(tts, wavs)
    assert vae.batch_sizes == [2, 1]
    for (samples, sr, mode), feat in zip(wavs, batched, strict=True):
        torch.testing.assert_close(feat, _encode_raw_audio(tts, samples, sr, padding_mode=mode, keep_on_device=True))

    model = _talker(torch.device("cpu"))
    state = _state(model, 3, torch.device("cpu"), monkeypatch)
    model._pending_requests = []
    model._tts = tts
    model.config.bos_token_id = 1
    speaker_cache = {}
    model._speaker_cache = SimpleNamespace(
        make_cache_key=lambda name, *, model_type, created_at: (name, model_type, created_at),
        get=speaker_cache.get,
        put=speaker_cache.__setitem__,
    )
    monkeypatch.setattr(model, "_get_multichar_zh_split", lambda: {})
    model.model = SimpleNamespace(embed_input_ids=lambda ids: ids.float().unsqueeze(-1).expand(*ids.shape, 6))
    tts.feat_encoder = lambda feat: feat.sum(2)
    tts.enc_to_lm_proj = lambda feat: feat.repeat_interleave(3, dim=-1)
    tts._make_ref_prefix = lambda ref, _device: (
        torch.zeros(ref.shape[0], dtype=torch.int32),
        ref,
        torch.zeros(ref.shape[0], dtype=torch.int32),
        torch.ones(ref.shape[0], dtype=torch.int32),
    )
    tts.text_tokenizer = lambda _text: [6]
    infos = []
    for index, samples in enumerate(([1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0])):
        req_id = f"voice-{index}"
        state.add_request(index, SimpleNamespace(req_id=req_id))
        info = state.intermediate_buffer.buffers[index]
        ref_bytes = np.asarray(samples, dtype=np.float32).tobytes()
        info.update(req_id=req_id, text_token_ids=[[1, index + 2]], ref_audio=[[ref_bytes, 8000]])
        if index == 0:
            info.update(voice_name="named", voice_created_at=42)
        if index == 2:
            prompt_bytes = np.asarray([11.0, 12.0, 13.0], dtype=np.float32).tobytes()
            info.update(prompt_audio=[[prompt_bytes, 8000]], prompt_text=["hello"])
        infos.append(info)
    model.preprocess_batch_mrv2(req_infos=infos, device=torch.device("cpu"))
    assert vae.batch_sizes[-1] == 4
    for index, info in enumerate(infos):
        cache, embeds, text_mask, audio_mask, audio_feat, feat_embed = info["prepared_prefill"]
        individual_cache = model._build_prompt_cache(
            ref_audio=info["ref_audio"][0],
            prompt_audio=info["prompt_audio"][0] if info.get("prompt_audio") else None,
            prompt_text=info["prompt_text"][0] if info.get("prompt_text") else None,
        )
        state.slots[index].prompt_cache = individual_cache
        inputs = model._build_prefill_inputs([index + 2], torch.device("cpu"), state=state.slots[index])
        text = model.model.embed_input_ids(inputs.text_token)
        feat = tts.enc_to_lm_proj(tts.feat_encoder(inputs.audio_feat))
        expected = (inputs.text_mask.unsqueeze(-1) * text + inputs.audio_mask.unsqueeze(-1) * feat).squeeze(0)
        torch.testing.assert_close(cache["ref_audio_feat"], individual_cache["ref_audio_feat"])
        torch.testing.assert_close(embeds, expected)
        for actual, reference in zip(
            (text_mask, audio_mask, audio_feat, feat_embed),
            (inputs.text_mask, inputs.audio_mask, inputs.audio_feat, feat),
            strict=True,
        ):
            torch.testing.assert_close(actual, reference)
        model._pending_batch_rows = []
        _ids, consumed, _updates = model.preprocess(
            torch.zeros(embeds.shape[0], dtype=torch.long),
            None,
            req_id=info["req_id"],
            slot_index=index,
            batch_row=index,
            _omni_is_prefill=True,
            text_token_ids=info["text_token_ids"],
            voice_name=info.get("voice_name"),
            voice_created_at=info.get("voice_created_at"),
            prepared_prefill=info["prepared_prefill"],
        )
        torch.testing.assert_close(consumed, expected)
    torch.testing.assert_close(
        speaker_cache[("named", "voxcpm2", 42)]["ref_audio_feat"],
        infos[0]["prepared_prefill"][0]["ref_audio_feat"],
    )
    infos[0].pop("prepared_prefill")
    state.slots[0].prefill_embeds = None
    state.slots[0].prompt_cache = None
    before = list(vae.batch_sizes)
    model.preprocess_batch_mrv2(req_infos=infos, device=torch.device("cpu"))
    assert "prepared_prefill" not in infos[0]
    assert vae.batch_sizes == before


def test_raw_audio_encode_can_keep_features_on_model_device(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeVAE(nn.Module):
        latent_dim = 2

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def encode(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
            return torch.ones(1, 2, 4, device=audio.device)

    tts = SimpleNamespace(audio_vae=FakeVAE(), _encode_sample_rate=16000, patch_size=2, chunk_size=2)
    monkeypatch.setattr(torch.Tensor, "cpu", lambda *_args, **_kwargs: pytest.fail("unexpected feature D2H"))
    feat = _encode_raw_audio(tts, [0.0] * 4, 16000, keep_on_device=True)
    assert feat.shape == (2, 2, 2)


def test_raw_audio_resampling_avoids_numpy_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    class EchoVAE(nn.Module):
        latent_dim = 1

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def encode(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
            return audio.unsqueeze(1)

    tts = SimpleNamespace(audio_vae=EchoVAE(), _encode_sample_rate=16000, patch_size=2, chunk_size=2)
    samples = [float(i) / 16 for i in range(16)]
    expected = _encode_raw_audio(tts, samples, 8000)
    monkeypatch.setattr(
        torch.Tensor, "numpy", lambda *_args, **_kwargs: pytest.fail("unexpected audio NumPy round trip")
    )
    actual = _encode_raw_audio(tts, samples, 8000, keep_on_device=True)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_reference_resampling_matches_legacy_cpu_path() -> None:
    class EchoVAE(nn.Module):
        latent_dim = 1

        def __init__(self, device: torch.device) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, device=device))

        def encode(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
            return audio.unsqueeze(1)

    samples = np.linspace(-1, 1, 24, dtype=np.float32)
    cpu_tts = SimpleNamespace(
        audio_vae=EchoVAE(torch.device("cpu")), _encode_sample_rate=16000, patch_size=2, chunk_size=2
    )
    gpu_tts = SimpleNamespace(
        audio_vae=EchoVAE(torch.device("cuda")), _encode_sample_rate=16000, patch_size=2, chunk_size=2
    )
    expected = _encode_raw_audio(cpu_tts, samples.tolist(), 8000)
    actual = _encode_raw_audio(gpu_tts, samples.tobytes(), 8000, keep_on_device=True)
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mrv2_output_does_not_copy_audio_to_host(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cuda")
    model = _talker(device)
    state = _state(model, 1, device, monkeypatch)
    state.add_request(0, SimpleNamespace(req_id="first"))
    model._audio_queue = [("first", torch.ones(8, device=device))]
    state.intermediate_buffer.req_id_to_index = {"first": 0}
    monkeypatch.setattr(torch.Tensor, "cpu", lambda *_args, **_kwargs: pytest.fail("model-side audio D2H"))
    output = model.make_omni_output(torch.zeros(1, 4, device=device), request_ids=["first"])
    assert state.audio_buffer.device.type == "cuda"
    assert output.multimodal_outputs["voxcpm2_audio_pending"] is True
