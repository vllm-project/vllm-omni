# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native Lychee snapshot leases and publication ownership."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.sampling_params import RequestOutputKind

from vllm_omni.model_executor.models.lychee_fd.duplex.codec import output_payload
from vllm_omni.model_executor.models.lychee_fd.output_ring import LYCHEE_OUTPUT_COLUMNS, LycheeOutputRing
from vllm_omni.model_executor.output_snapshot import OutputCopyLifetimeError
from vllm_omni.outputs.output_processor import OmniRequestState
from vllm_omni.worker_v2.model_states.lychee_sampler import LycheeSampler
from vllm_omni.worker_v2.native_output_worker import NativeOutputWorker
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner
from vllm_omni.worker_v2.omni_sampler import OmniSamplingContext

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _outputs(rows=5, tick=10):
    values = {key: torch.full((rows,), index, dtype=torch.int32) for index, key in enumerate(LYCHEE_OUTPUT_COLUMNS)}
    values["lychee_tick"] = torch.arange(tick, tick + rows, dtype=torch.int32)
    return values


def _copy_and_bind(snapshot):
    snapshot.mark_copy_started()
    cpu = snapshot.copy_to_cpu(lambda slab: slab.clone())
    snapshot.bind_copy_event(object())
    return cpu


def _output(payload):
    return SimpleNamespace(request_id="owner", outputs=[SimpleNamespace(multimodal_output=payload)])


def _state():
    return OmniRequestState(
        request_id="owner",
        external_req_id="owner",
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
        output_kind=RequestOutputKind.DELTA,
    )


def test_compact_snapshot_has_one_group_and_retained_cpu_survives_slot_reuse():
    ring = LycheeOutputRing(max_num_reqs=2, capacity=2, device=torch.device("cpu"))
    saved = []
    for index in range(20):
        source = _outputs(tick=index * 10)
        rows = [4] if index % 2 else [1, 4]
        snapshot = ring.pack(source, rows)
        copied = []
        snapshot.mark_copy_started()

        def copy_slab(slab: torch.Tensor) -> torch.Tensor:
            copied.append(slab.numel())
            return slab.clone()

        cpu = snapshot.copy_to_cpu(copy_slab)
        snapshot.bind_copy_event(object())
        source["lychee_tick"].fill_(999)
        saved.append((cpu, [index * 10 + row for row in rows]))
        assert copied == [11 * len(rows)]
    assert len(ring.slots) == 2
    assert all(slot.storage.numel() == 22 for slot in ring.slots)
    for cpu, ticks in saved:
        assert cpu["lychee_tick"].tolist() == ticks


def test_pending_started_and_bound_leases_have_distinct_cleanup():
    ring = LycheeOutputRing(max_num_reqs=1, capacity=3, device=torch.device("cpu"))
    unstarted = ring.pack(_outputs(), [4])
    started = ring.pack(_outputs(), [4])
    started.mark_copy_started()
    bound = ring.pack(_outputs(), [4])
    bound_event = object()
    bound.mark_copy_started()
    bound.bind_copy_event(bound_event)
    ring.abort_unbound()
    assert [slot.pending for slot in ring.slots] == [False, True, False]
    assert ring.slots[2].completion is bound_event
    assert ring.slots[0].snapshot is None and ring.slots[1].snapshot is started
    ring.pack(_outputs(), [4])
    with pytest.raises(RuntimeError, match="unbound copy lease"):
        ring.pack(_outputs(), [4])
    assert not unstarted.copy_started


def test_old_callback_cannot_release_a_reused_generation():
    ring = LycheeOutputRing(max_num_reqs=1, capacity=1, device=torch.device("cpu"))
    snapshot = ring.pack(_outputs(), [4])
    callback = snapshot._copy_completion_callback
    _copy_and_bind(snapshot)
    ring.pack(_outputs(), [4])
    callback(object())  # Callback itself is explicitly non-throwing.
    assert ring.slots[0].generation == 2 and ring.slots[0].pending


def test_partial_d2h_failure_and_cpu_finalizer_error_preserve_bound_lifetime():
    ring = LycheeOutputRing(max_num_reqs=1, capacity=1, device=torch.device("cpu"))
    snapshot = ring.pack(_outputs(), [4])
    snapshot.mark_copy_started()
    cpu = snapshot.copy_to_cpu(lambda slab: slab.clone())
    event = object()
    snapshot.bind_copy_event(event)  # Generic constructor's finally fence.
    ring.abort_unbound()
    assert ring.slots[0].completion is event and not ring.slots[0].pending
    batch = SimpleNamespace(num_reqs=1, query_start_loc_np=np.array([0, 5]))
    finalize = LycheeSampler._output_finalizer(batch, compact=True)
    with pytest.raises(ValueError, match="counts"):
        finalize(cpu, [])
    ring.pack(_outputs(tick=500), [4])
    assert cpu["lychee_tick"].tolist() == [14]


def test_preconstructor_failure_releases_only_unstarted_transaction_lease():
    ring = LycheeOutputRing(max_num_reqs=1, capacity=2, device=torch.device("cpu"))
    bound = ring.pack(_outputs(), [4])
    completion = object()
    bound.mark_copy_started()
    bound.bind_copy_event(completion)
    ring.pack(_outputs(), [4])
    marks = []
    sampler = LycheeSampler(
        SimpleNamespace(), SimpleNamespace(mark_primary_continuation_failed=lambda **kw: marks.append(kw))
    )
    sampler._output_ring = ring
    with pytest.raises(RuntimeError, match="prompt logprobs"):
        with sampler.set_sampling_context(OmniSamplingContext(object(), lambda: None)):
            raise RuntimeError("prompt logprobs")
    assert not any(slot.pending for slot in ring.slots)
    assert ring.slots[0].completion is completion
    assert len(marks) == 1 and sampler._sampling_context is None


@pytest.mark.parametrize("started", [False, True])
def test_copy_dependency_fault_leaves_unfenced_slab_unavailable(started):
    ring = LycheeOutputRing(max_num_reqs=1, capacity=1, device=torch.device("cpu"))
    snapshot = ring.pack(_outputs(), [4])
    if started:
        snapshot.mark_copy_started()
    sampler = LycheeSampler(SimpleNamespace(), SimpleNamespace(mark_primary_continuation_failed=lambda **kw: None))
    sampler._output_ring = ring
    with pytest.raises(OutputCopyLifetimeError):
        with sampler.set_sampling_context(OmniSamplingContext(object(), lambda: None)):
            raise OutputCopyLifetimeError("copy dependency event")
    assert ring.slots[0].pending
    with pytest.raises(RuntimeError, match="unbound"):
        ring.pack(_outputs(), [4])


@pytest.mark.parametrize("representation", ["legacy", "flat_chunk", "nested_chunk"])
def test_payload_normalization_is_canonical_idempotent_and_keeps_waveform_owner(representation):
    values = _outputs(rows=1)
    if representation == "flat_chunk":
        payload = {f"chunk.{key}": value for key, value in values.items()}
    elif representation == "nested_chunk":
        payload = {"chunk": dict(values)}
    else:
        payload = dict(values)
    payload["audio"] = torch.arange(3, dtype=torch.float32)
    payload["chunk.lychee_t2w.session_epoch"] = torch.tensor([2])
    _, _, canonical = output_payload(_output(payload))
    assert canonical["lychee_tick"].tolist() == [10]
    assert canonical["audio"].numel() == 3
    assert "chunk.lychee_t2w.session_epoch" in canonical
    _, _, again = output_payload(_output(canonical))
    assert again["lychee_tick"].tolist() == [10]
    assert all(key in values for key in LYCHEE_OUTPUT_COLUMNS)


def test_ambiguous_top_level_and_chunk_decisions_are_rejected():
    with pytest.raises(ValueError, match="Ambiguous Lychee decision"):
        output_payload(_output({"lychee_tick": torch.tensor([1]), "chunk.lychee_tick": torch.tensor([2])}))


def test_real_delta_processor_drains_all_300_step_columns_without_losing_rows():
    state = _state()
    ring = LycheeOutputRing(max_num_reqs=1, capacity=2, device=torch.device("cpu"))
    batch = SimpleNamespace(num_reqs=1, query_start_loc_np=np.array([0, 5]))
    finalize = LycheeSampler._output_finalizer(batch, compact=True, client_delta_rows=(True,))
    observed = []
    for tick in range(300):
        snapshot = ring.pack(_outputs(tick=tick * 10), [4])
        cpu = _copy_and_bind(snapshot)
        partition = finalize(cpu, [1])
        assert "lychee_tick" in partition.inter_stage[0]
        assert "lychee_tick" not in partition.client[0]
        state.add_multimodal_tensor(partition.client[0], "text")
        output = state.make_request_output([], None, None, None)
        _, _, canonical = output_payload(output)
        observed.extend(canonical["lychee_tick"].tolist())
        assert all(canonical[key].numel() == 1 for key in LYCHEE_OUTPUT_COLUMNS)
        assert state.mm_accumulated.is_empty
    assert observed == [tick * 10 + 4 for tick in range(300)]


def test_pending_delta_batch_retains_all_rows_before_publication():
    state = _state()
    for tick in (1, 2, 3):
        state.add_multimodal_tensor(
            {f"chunk.{key}": value for key, value in _outputs(rows=1, tick=tick).items()}, "text"
        )
    output = state.make_request_output([], None, None, None)
    _, _, canonical = output_payload(output)
    ticks = canonical["lychee_tick"]
    assert torch.cat(ticks).tolist() == [1, 2, 3]
    assert state.mm_accumulated.is_empty


def test_sampler_captures_duplex_client_policy_before_slot_reuse():
    buffer = SimpleNamespace(buffers=[{"duplex": {"data_plane": True}}, {}])
    sampler = LycheeSampler(SimpleNamespace(), SimpleNamespace(intermediate_buffer=buffer))
    batch = SimpleNamespace(num_reqs=2, query_start_loc_np=np.array([0, 2, 5]), idx_mapping_np=np.array([0, 1]))
    policy = sampler._client_delta_rows(batch)
    finalize = sampler._output_finalizer(batch, compact=True, client_delta_rows=policy)
    buffer.buffers[:] = [{}, {"duplex": {"data_plane": True}}]
    batch.idx_mapping_np[:] = [1, 0]
    partition = finalize(_outputs(rows=2), [1, 1])
    assert "chunk.lychee_tick" in partition.client[0] and "lychee_tick" in partition.client[1]
    assert all("lychee_tick" in payload for payload in partition.inter_stage)


def test_native_materializer_publishes_inter_stage_once_and_retains_client_namespace():
    records = []
    plane = SimpleNamespace(enqueue_outputs=lambda **kw: records.append(kw), get_omni_connector_output=lambda: [])
    value = torch.tensor([7], dtype=torch.int32)
    output = SimpleNamespace(
        req_ids=["owner"],
        sampled_token_ids=[[5]],
        sampled_token_ids_materialized=True,
        inter_stage_outputs=[{"lychee_tick": value}],
        multimodal_outputs=[{"chunk.lychee_tick": value}],
    )
    worker = NativeOutputWorker(2)
    try:
        async_output = SimpleNamespace(get_output=lambda: output, copy_event=object())
        owned = worker.submit(async_output, plane)
        first = owned.get_output()
        assert owned.get_output() is first
        assert len(records) == 1 and records[0]["inter_stage_outputs"][0]["lychee_tick"].tolist() == [7]
        assert first.inter_stage_outputs is None
        assert first.multimodal_outputs[0]["chunk.lychee_tick"].tolist() == [7]
    finally:
        worker.close()
    # The alternate direct finalizer uses the same two-channel routing.
    direct = SimpleNamespace(
        req_ids=["owner"],
        sampled_token_ids=[[5]],
        sampled_token_ids_materialized=True,
        inter_stage_outputs=[{"lychee_tick": value}],
        multimodal_outputs=[{"chunk.lychee_tick": value}],
    )
    OmniGPUModelRunner._finalize_native_data_plane_output(SimpleNamespace(_omni_data_plane=plane), direct)
    assert len(records) == 2 and direct.inter_stage_outputs is None
    assert "chunk.lychee_tick" in direct.multimodal_outputs[0]


@pytest.mark.parametrize("nested", [False, True])
def test_namespaced_decisions_reach_duplex_control_and_codec_projection_once(nested):
    from vllm_omni.model_executor.models.lychee_fd.duplex.codec import LycheeCodecStreams
    from vllm_omni.model_executor.models.lychee_fd.duplex.data_plane import (
        LycheeDataPlaneContext,
        LycheeDataPlaneSession,
    )

    owner = "duplex-s.cHJvYmU.e.0.r.stage0"
    values = {
        "lychee_tick": torch.tensor([9], dtype=torch.int32),
        "lychee_control_token_ids": torch.tensor([158354], dtype=torch.int32),
        "lychee_audio_window_seq": torch.tensor([1], dtype=torch.int32),
    }
    payload = {"chunk": values} if nested else {f"chunk.{key}": value for key, value in values.items()}
    output = _output(payload)
    output.request_id = owner
    plane = LycheeDataPlaneSession()
    context = LycheeDataPlaneContext(
        epoch=0,
        turn_id=3,
        active_response_turn_id=None,
        active_response_id=None,
        auto_responds=True,
        response_format="wav",
        speed=None,
        modalities=("audio",),
    )
    events = tuple(plane.project({"data_plane_outputs": [output]}, context=context))
    assert len(events) == 1 and events[0]["is_listen"] is True
    assert events[0]["lychee_tick"] == 9 and events[0]["data_plane_request_id"] == owner
    assert tuple(plane.project({"data_plane_outputs": [output]}, context=context)) == ()

    codec_values = {
        "lychee_tick": torch.tensor([10, 11]),
        "lychee_text_token_ids": torch.tensor([158358, 158358]),
        "lychee_speech_token_ids": torch.tensor([151696, 151694]),
        "lychee_control_token_ids": torch.tensor([158352, 158353]),
        "lychee_execution_epoch": torch.tensor([0, 0]),
    }
    codec_payload = (
        {"chunk": codec_values} if nested else {f"chunk.{key}": value for key, value in codec_values.items()}
    )
    _, _, canonical = output_payload(_output(codec_payload))
    streams = LycheeCodecStreams()
    packets = streams.consume_all(owner, canonical, session_epoch=0)
    assert len(packets) == 1 and packets[0]["codec_ids"] == [0] and packets[0]["final"] is True
    assert streams.consume_all(owner, canonical, session_epoch=0) == []
