# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for OmniARModelRunner v2: async output staging, snapshot ownership, payload slicing."""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.v1.outputs import RoutedExpertsTensors
from vllm.v1.worker.gpu.sample.output import SamplerOutput, SamplingMaskTensors

import vllm_omni.worker_v2.omni_ar_model_runner as omni_ar_model_runner
from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner, OmniAsyncOutput
from vllm_omni.worker_v2.output_snapshot import pack_output_snapshot

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeStream:
    def wait_stream(self, _stream) -> None:
        pass


class _FakeEvent:
    def record(self, _stream) -> None:
        pass

    def synchronize(self) -> None:
        pass


def _async_output(req_ids=("req-0",), **overrides) -> OmniAsyncOutput:
    rid2idx = {rid: i for i, rid in enumerate(req_ids)}
    mro = omni_ar_model_runner.OmniModelRunnerOutput(list(req_ids), rid2idx, None, prompt_logprobs_dict={})
    sampler_output = SamplerOutput(torch.tensor([[123]]), None, None, None)
    kwargs = dict(model_runner_output=mro, sampler_output=sampler_output)
    kwargs.update(num_sampled_tokens=torch.tensor([1] * len(req_ids)), copy_event=_FakeEvent())
    kwargs.update(main_stream=_FakeStream(), copy_stream=_FakeStream())
    return OmniAsyncOutput(**(kwargs | overrides))


def test_async_output_blocking_event_and_routing_masks(monkeypatch) -> None:
    event_kwargs = []
    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)

    def make_event(**kwargs):
        event_kwargs.append(kwargs)
        return _FakeEvent()

    monkeypatch.setattr(torch.cuda, "Event", make_event)
    masks = SamplingMaskTensors(torch.tensor([[5], [0]], dtype=torch.uint8), torch.tensor([2, 0]), 4)
    sampler_output = SamplerOutput(torch.tensor([[2], [0]]), None, None, torch.tensor([1, 0]), None, masks)
    routed = RoutedExpertsTensors(torch.tensor([[[2, 3]], [[4, 5]]]), torch.tensor([7, 9]))

    output = _async_output(
        req_ids=["decode", "prefill"],
        sampler_output=sampler_output,
        num_sampled_tokens=torch.tensor([1, 0]),
        copy_event=None,  # None → constructor builds the default blocking event
        routed_experts=routed,
    ).get_output()
    assert event_kwargs == [{"blocking": True}]  # blocking event by default
    assert output.sampled_token_ids == [[2], []] and output.sampling_masks.cu_num_generated_tokens == [0, 1, 1]
    np.testing.assert_array_equal(output.routed_experts.routing_data, [[[2, 3]], [[4, 5]]])
    np.testing.assert_array_equal(output.routed_experts.slot_mapping, [7, 9])
    np.testing.assert_array_equal(output.sampling_masks.token_ids, [0, 2])


@pytest.mark.parametrize("needs_history", [False, True])
def test_last_pp_rank_orchestration_and_kv_resolver(monkeypatch, needs_history) -> None:
    runner = OmniARModelRunner.__new__(OmniARModelRunner)
    input_batch = SimpleNamespace(req_ids=["req"], num_reqs=1, seq_lens=torch.tensor([3]))
    input_batch.idx_mapping, input_batch.query_start_loc = torch.tensor([0]), torch.tensor([0, 1])
    state = SimpleNamespace(input_batch=input_batch, hidden_states=torch.zeros(1, 2))
    state.finished_req_ids, state.ec_connector_output, state.routed_experts = {"finished"}, None, None
    runner.execute_model_state = state
    runner._kv_extracted_req_ids = runner._last_aux_output = runner._last_multimodal_outputs = None
    runner.is_last_pp_rank, runner.pp_handler, runner.check_ep_fault = True, None, False
    runner.model_config = SimpleNamespace(async_chunk=False)
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(engine_output_type="text"))
    runner.model_state = SimpleNamespace(postprocess_model_output=MagicMock(return_value=(torch.zeros(1, 2), None)))
    runner.model_state.intermediate_buffer = SimpleNamespace(buffers={0: {"global_request_id": "global-req"}})
    runner.req_states = SimpleNamespace(req_id_to_index={"req": 0})
    runner.req_states.all_token_ids = SimpleNamespace(gpu=torch.tensor([[1]]))
    runner.req_states.num_computed_tokens = SimpleNamespace(gpu=torch.tensor([0]))
    runner.req_states.prompt_len = SimpleNamespace(np=np.array([1]), gpu=torch.tensor([1]))
    runner.main_stream = runner.output_copy_stream = MagicMock()
    runner.eplb = runner._finalize_native_data_plane_output = runner._reserve_native_data_plane_outputs = MagicMock()
    sampler_out = (SimpleNamespace(sampled_token_ids=torch.tensor([[2]])), MagicMock(), MagicMock())
    sampling_active = False

    @contextmanager
    def sampling_context(*, req_ids, num_output_tokens):
        nonlocal sampling_active
        assert needs_history and req_ids == ["req"]
        sampling_active = True
        yield
        sampling_active = False

    runner.model = SimpleNamespace(compute_logits=None, logitsprocs_need_output_token_ids=needs_history)
    runner.model.mrv2_sampling_context = sampling_context
    runner.sample = MagicMock(return_value=sampler_out)
    runner.sample.side_effect = lambda *_: sampler_out if sampling_active is needs_history else pytest.fail()
    logprobs_mock = MagicMock(side_effect=lambda *_: pytest.fail("inside ctx") if sampling_active else {})
    runner.prompt_logprobs_worker = SimpleNamespace(compute_prompt_logprobs=logprobs_mock)
    runner.postprocess_sampled, connector_output = MagicMock(), object()

    def post_forward(finished_req_ids):
        runner.postprocess_sampled.assert_called_once()  # postprocess precedes kv_connector.post_forward
        assert finished_req_ids == {"finished"}
        return connector_output

    runner.kv_connector = SimpleNamespace(post_forward=MagicMock(side_effect=post_forward))
    mock_out = SimpleNamespace(copy_event=None)
    monkeypatch.setattr(omni_ar_model_runner, "OmniAsyncOutput", MagicMock(return_value=mock_out))

    assert runner.sample_tokens(None) is mock_out
    built = omni_ar_model_runner.OmniAsyncOutput.call_args.kwargs["model_runner_output"]
    assert built.kv_connector_output is connector_output
    assert runner._resolve_global_request_id("req") == "global-req"  # from the intermediate buffer
    assert runner._resolve_global_request_id("unknown") == "unknown"  # fallback to the local id


def test_async_mm_snapshot_owns_output_until_copy_finishes() -> None:
    runner = OmniARModelRunner.__new__(OmniARModelRunner)
    runner.model_config = SimpleNamespace(async_chunk=True)
    runner._async_mm_snapshot_slots, runner._async_mm_snapshot_events = [{}], [None]
    runner._async_mm_snapshot_pending, runner._async_mm_snapshot_cursor = [False], 0
    waited: list[object] = []
    runner.main_stream = SimpleNamespace(wait_event=waited.append)
    source = torch.tensor([[7, 8]], dtype=torch.long)

    snapshot = runner._retain_multimodal_outputs({"codes": {"audio": source}})
    source.fill_(99)

    # Snapshot owns the data (graph replay cannot overwrite it)...
    snap = snapshot["codes"]["audio"]
    assert snap.tolist() == [[7, 8]] and snap.data_ptr() != source.data_ptr()
    assert runner._last_multimodal_snapshot_slot == 0
    # ...and slot reuse waits for the previous D2H copy event.
    runner._release_multimodal_snapshot(0, copy_event := object())
    runner._retain_multimodal_outputs({"codes": {"audio": torch.zeros(1, 2)}})
    assert waited == [copy_event]


def test_snapshot_slots_bounded_by_shape_and_packed_grouping_isolation(monkeypatch) -> None:
    slot: dict[tuple[Any, ...], torch.Tensor] = {}
    omni_ar_model_runner._copy_mm_to_snapshot_slot(torch.ones(1, 2), slot)
    omni_ar_model_runner._copy_mm_to_snapshot_slot(torch.ones(4, 2), slot)
    assert len(slot) == 2  # separate shapes do not share a buffer
    monkeypatch.setattr(omni_ar_model_runner, "_ASYNC_MM_SNAPSHOT_MAX_BUCKETS_PER_SLOT", 1)
    bounded: dict[tuple[Any, ...], torch.Tensor] = {}
    kept = omni_ar_model_runner._copy_mm_to_snapshot_slot(torch.ones(2, 2), bounded)
    overflow = omni_ar_model_runner._copy_mm_to_snapshot_slot(torch.ones(5, 2), bounded)
    assert len(bounded) == 1 and overflow.data_ptr() != kept.data_ptr()  # overflow clones without evicting

    source = torch.arange(12, dtype=torch.int64).view(3, 4).t()
    payload = {"noncontiguous": source, "nested": [torch.tensor(7), (torch.tensor([1.5]),)], "meta": "ok"}
    pack_slot: dict[tuple[Any, ...], torch.Tensor] = {}
    snapshot = pack_output_snapshot(payload, pack_slot, max_buckets=4)
    source.fill_(99)  # ownership: snapshot keeps pre-mutation values
    copies = []

    def copy(tensor):
        copies.append(tensor.numel())
        return tensor.clone()

    host = snapshot.copy_to_cpu(copy)
    assert len(copies) == 2  # grouped by dtype (int64, float32), not tensor count
    assert host["noncontiguous"].tolist() == torch.arange(12).view(3, 4).t().tolist()
    assert host["nested"][0].item() == 7 and isinstance(host["nested"][1], tuple) and host["meta"] == "ok"
    pack_output_snapshot(payload, pack_slot, max_buckets=4)
    assert host["noncontiguous"][0, 0].item() == 0  # repacking must not corrupt published host data


@pytest.mark.parametrize("need_pooler,async_chunk", [(False, False), (False, True), (True, False), (True, True)])
def test_guard_graph_replay_for_pooler_copy(need_pooler, async_chunk) -> None:
    main_stream = MagicMock()
    omni_ar_model_runner._guard_graph_replay_for_pooler_copy(
        main_stream, object(), need_pooler=need_pooler, async_chunk=async_chunk
    )
    # Non-async pooler copies must gate the next graph replay on the copy event.
    assert main_stream.wait_event.call_count == (1 if need_pooler and not async_chunk else 0)


def test_build_async_chunk_outputs_slices_padded_axis_and_splits_channels() -> None:
    # Graph-padded batch: padded_total_tokens > total_tokens; slice by the real token axis.
    padded_codes = torch.arange(16, dtype=torch.long).reshape(8, 2)
    build = OmniARModelRunner._build_async_chunk_outputs_from_mm
    inter_stage, client = build({"codes": {"audio": padded_codes}}, np.array([0, 1, 2]), np.array([1, 1]), 2, 2, 8)
    assert client is None
    assert torch.equal(inter_stage[0]["codes.audio"], padded_codes[0:1])
    assert torch.equal(inter_stage[1]["codes.audio"], padded_codes[1:2])

    codes, audio = torch.arange(16, dtype=torch.long).reshape(4, 4), torch.randn(4, 8)
    req_codes = [torch.arange(16 * i, 16 * (i + 1), dtype=torch.long).reshape(1, 16) for i in range(2)]
    inter_stage, client = build(
        {"codes": {"audio": codes}, "audio": audio}, np.array([0, 2, 4]), np.array([2, 2]), 2, 4
    )
    assert "hidden" not in inter_stage[0] and torch.equal(client[1]["audio"], audio[2:])  # channel split
    assert torch.equal(inter_stage[1]["codes.audio"], codes[2:])  # inter-stage channel
    # Per-request code lists pass through unsliced.
    inter_stage, client = build({"codes": {"audio": req_codes}}, np.array([0, 1, 2]), np.array([1, 1]), 2, 2)
    assert client is None and torch.equal(inter_stage[0]["codes.audio"], req_codes[0])


def test_async_chunk_output_stages_mm_on_copy_stream_before_get_output(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    calls = []

    def copy_mm(mm_outputs, total_tokens, **ctx):
        calls.append((total_tokens, ctx))
        return {"codes": {"audio": mm_outputs["codes"]["audio"].clone()}}

    monkeypatch.setattr(omni_ar_model_runner, "_async_copy_mm", copy_mm)
    source_codes = torch.tensor([[7, 8]], dtype=torch.long)
    input_batch = SimpleNamespace(query_start_loc_np=np.array([0, 1]), num_scheduled_tokens=[1], num_reqs=1)
    output = _async_output(
        multimodal_outputs={"codes": {"audio": source_codes}},
        input_batch=input_batch,
        copy_stream=(copy_stream := _FakeStream()),
        async_chunk=True,
    )
    # Staged once on the copy stream during construction, with one resolved pin-memory context for all D2H helpers.
    [(total_tokens, ctx)] = calls
    assert total_tokens == 1 and ctx["copy_stream"] is copy_stream and ctx["pin_memory"] is not None
    assert output._mm_snapshot["codes"]["audio"].device.type == "cpu"
    source_codes.fill_(99)  # a later graph replay cannot leak into the snapshot
    assert torch.equal(output.get_output().inter_stage_outputs[0]["codes.audio"], torch.tensor([[7, 8]]))


@pytest.mark.parametrize("prefill_first", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_request_reference_codes_preserve_local_axis(prefill_first, padded):
    lengths = np.array([255, 1] if prefill_first else [1, 255])
    offsets = np.array([0, lengths[0], 256])
    size = 512 if padded else 256
    ref = torch.arange(size * 16).reshape(size, 16)
    refs = [ref, torch.empty(0)] if prefill_first else [torch.empty(0), ref]
    codes = torch.arange(size * 16).reshape(size, 16)
    outputs, _ = OmniARModelRunner._build_async_chunk_outputs_from_mm(
        {"codes": {"audio": codes, "ref": refs}}, offsets, lengths, 2, 256, size
    )
    index = 0 if prefill_first else 1
    assert torch.equal(outputs[index]["codes.ref"], ref)
    assert outputs[index]["codes.ref"].data_ptr() != ref.data_ptr()
    for i in range(2):
        assert torch.equal(outputs[i]["codes.audio"], codes[offsets[i] : offsets[i + 1]])
