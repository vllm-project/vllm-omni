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
from vllm.distributed.aux_output_connector.connector import AuxRequestOutput
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.output import SamplerOutput, SamplingMaskTensors
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker

import vllm_omni.worker_v2.omni_ar_model_runner as omni_ar_model_runner
from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.output_snapshot import PackedOutputSnapshot
from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner, OmniAsyncOutput
from vllm_omni.worker_v2.output_snapshot import pack_output_snapshot

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeStream:
    def wait_stream(self, _stream) -> None:
        pass

    def wait_event(self, _event) -> None:
        pass


class _FakeEvent:
    def record(self, _stream) -> None:
        pass

    def synchronize(self) -> None:
        pass


def _async_output(req_ids=("req-0",), **overrides) -> OmniAsyncOutput:
    rid2idx = {rid: i for i, rid in enumerate(req_ids)}
    mro = omni_ar_model_runner.OmniModelRunnerOutput(list(req_ids), rid2idx, None, prompt_logprobs_dict={})
    sampler_output = SamplerOutput(torch.tensor([[123]]), None, None, torch.tensor([1]), torch.tensor([0]))
    kwargs = dict(model_runner_output=mro, sampler_output=sampler_output)
    kwargs.update(num_sampled_tokens=torch.tensor([1] * len(req_ids)), copy_event=_FakeEvent())
    kwargs.update(main_stream=_FakeStream(), copy_stream=_FakeStream())
    return OmniAsyncOutput(**(kwargs | overrides))


@pytest.mark.parametrize("compact_width", [1, 2])
def test_async_output_blocking_event_preserves_masks_and_aux_output(monkeypatch, compact_width) -> None:
    event_kwargs = []
    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)

    def make_event(**kwargs):
        event_kwargs.append(kwargs)
        return _FakeEvent()

    monkeypatch.setattr(torch.cuda, "Event", make_event)
    masks = SamplingMaskTensors(
        token_ids=torch.tensor([[0, 2], [0, 0]], dtype=torch.int32)[:, :compact_width],
        packed_mask=torch.tensor([[5], [0]], dtype=torch.uint8),
        counts=torch.tensor([2, 0]),
        vocab_size=4,
    )
    sampler_output = SamplerOutput(
        torch.tensor([[2], [0]]), None, None, torch.tensor([1, 0]), torch.tensor([2, 0]), masks
    )
    copied: list[tuple[np.ndarray, np.ndarray]] = []

    class _FakePendingAuxOutput:
        def enqueue_cpu_copy(self, *, num_sampled, num_rejected) -> None:
            copied.append((num_sampled, num_rejected))

        def process_output(self) -> dict[str, AuxRequestOutput]:
            return {
                "decode": AuxRequestOutput(token_start=0, rows=np.array([[2, 3]], dtype=np.uint8)),
                "prefill": AuxRequestOutput(token_start=0, rows=np.array([[4, 5]], dtype=np.uint8)),
            }

    pending = _FakePendingAuxOutput()
    output = _async_output(
        req_ids=["decode", "prefill"],
        sampler_output=sampler_output,
        num_sampled_tokens=torch.tensor([1, 0]),
        copy_event=None,  # None → constructor builds the default blocking event
        pending_aux_output=pending,
    ).get_output()
    assert event_kwargs == [{"blocking": True}]  # blocking event by default
    assert output.sampled_token_ids == [[2], []]
    assert output.sampling_masks.to_nested_list() == [[0, 2], []]
    np.testing.assert_array_equal(output.sampling_masks.offsets, [0, 2, 2])
    np.testing.assert_array_equal(output.aux_output_connector_output["decode"].rows, [[2, 3]])
    np.testing.assert_array_equal(output.aux_output_connector_output["prefill"].rows, [[4, 5]])
    assert output.aux_output_connector_output["decode"].token_start == 0
    np.testing.assert_array_equal(output.sampling_masks.token_ids, [0, 2])
    assert len(copied) == 1
    np.testing.assert_array_equal(copied[0][0], [1, 0])
    np.testing.assert_array_equal(copied[0][1], [2, 0])


@pytest.mark.parametrize("needs_history", [False, True])
def test_last_pp_rank_orchestration_and_kv_resolver(monkeypatch, needs_history) -> None:
    # This CPU orchestration test does not exercise pinned host transfers.
    monkeypatch.setattr("vllm.utils.torch_utils.PIN_MEMORY", False)
    runner = OmniARModelRunner.__new__(OmniARModelRunner)
    input_batch = SimpleNamespace(req_ids=["req"], num_reqs=1, seq_lens=torch.tensor([3]))
    input_batch.idx_mapping, input_batch.query_start_loc = torch.tensor([0]), torch.tensor([0, 1])
    input_batch.idx_mapping_np = np.array([0])
    input_batch.num_computed_prefill_tokens_np = np.array([0])
    input_batch.num_scheduled_tokens = np.array([3])
    input_batch.prefill_len_np = np.array([3])
    input_batch.query_start_loc_np = np.array([0, 3])
    state = SimpleNamespace(input_batch=input_batch, hidden_states=torch.zeros(1, 2))
    state.finished_req_ids, state.ec_connector_output = {"finished"}, None
    runner.execute_model_state = state
    runner._kv_extracted_req_ids = runner._last_aux_output = runner._last_multimodal_outputs = None
    runner.is_last_pp_rank, runner.pp_handler, runner.check_ep_fault = True, None, False
    runner.aux_output_connector = None
    runner.model_config = SimpleNamespace(async_chunk=False)
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(engine_output_type="text"))
    text_hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    runner.model_state = SimpleNamespace(postprocess_model_output=MagicMock(return_value=(text_hidden, None)))
    runner.model_state.intermediate_buffer = SimpleNamespace(buffers={0: {"global_request_id": "global-req"}})
    runner.req_states = SimpleNamespace(req_id_to_index={"req": 0})
    runner.req_states.all_token_ids = SimpleNamespace(gpu=torch.tensor([[1]]))
    runner.req_states.num_computed_tokens = SimpleNamespace(gpu=torch.tensor([0]))
    runner.req_states.prompt_len = SimpleNamespace(np=np.array([3]), gpu=torch.tensor([3]))
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

    runner.model = SimpleNamespace(
        compute_logits=lambda hidden: hidden, logitsprocs_need_output_token_ids=needs_history
    )
    runner.model.mrv2_sampling_context = sampling_context
    runner.sampler = None
    runner.sample = MagicMock(return_value=sampler_out)
    runner.sample.side_effect = lambda *_: sampler_out if sampling_active is needs_history else pytest.fail()
    logprobs_mock = MagicMock(side_effect=lambda *_: pytest.fail("inside ctx") if sampling_active else {})
    runner.prompt_logprobs_worker = PromptLogprobsWorker(1, torch.device("cpu"), logprobs_mode="raw_logits")
    runner.prompt_logprobs_worker.add_request("req", 0, SamplingParams(prompt_logprob_token_ids=[1, 0]))
    runner.prompt_logprobs_worker.compute_prompt_logprobs = logprobs_mock
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
    torch.testing.assert_close(built.prompt_token_id_logprobs_dict["req"], torch.tensor([[2.0, 1.0], [4.0, 3.0]]))
    assert runner._resolve_global_request_id("req") == "global-req"  # from the intermediate buffer
    assert runner._resolve_global_request_id("unknown") == "unknown"  # fallback to the local id


def test_async_output_preserves_fixed_token_scores_on_host(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    scores = torch.tensor([[-0.5, -1.5], [-0.2, -2.2]])
    mro = omni_ar_model_runner.OmniModelRunnerOutput(
        req_ids=["req-0"],
        req_id_to_index={"req-0": 0},
        sampled_token_ids=None,
        prompt_logprobs_dict={},
        prompt_token_id_logprobs_dict={"req-0": scores},
    )
    output = _async_output(model_runner_output=mro).get_output()
    assert output.prompt_token_id_logprobs_dict["req-0"].device.type == "cpu"
    torch.testing.assert_close(output.prompt_token_id_logprobs_dict["req-0"], scores)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_chunked_fixed_token_scores_are_copied_from_cuda() -> None:
    device = torch.device("cuda")
    hidden = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0], [4.0, 5.0, 6.0]], device=device)
    worker = PromptLogprobsWorker(1, device)
    worker.add_request("req", 0, SamplingParams(prompt_logprob_token_ids=[2, 0], prompt_logprob_start=1))
    batch = SimpleNamespace(
        req_ids=["req"],
        idx_mapping_np=np.array([0]),
        num_computed_prefill_tokens_np=np.array([0]),
        num_scheduled_tokens=np.array([2]),
        prefill_len_np=np.array([4]),
        query_start_loc_np=np.array([0, 2]),
    )
    assert worker.compute_prompt_token_id_logprobs(lambda hidden: hidden, hidden[:2], batch, np.array([4])) == {}
    batch.num_computed_prefill_tokens_np = np.array([2])
    scores = worker.compute_prompt_token_id_logprobs(lambda hidden: hidden, hidden[2:], batch, np.array([4]))
    mro = omni_ar_model_runner.OmniModelRunnerOutput(
        req_ids=["req"],
        req_id_to_index={"req": 0},
        sampled_token_ids=None,
        prompt_logprobs_dict={},
        prompt_token_id_logprobs_dict=scores,
    )
    num_sampled = torch.tensor([1], device=device, dtype=torch.int32)
    sampler_output = SamplerOutput(
        torch.tensor([[1]], device=device), None, None, num_sampled, torch.zeros_like(num_sampled)
    )
    output = OmniAsyncOutput(
        model_runner_output=mro,
        sampler_output=sampler_output,
        num_sampled_tokens=num_sampled,
        main_stream=torch.cuda.current_stream(),
        copy_stream=torch.cuda.Stream(),
    ).get_output()

    expected = hidden[1:3].log_softmax(dim=-1)[:, [2, 0]].cpu()
    assert output.prompt_token_id_logprobs_dict["req"].device.type == "cpu"
    torch.testing.assert_close(output.prompt_token_id_logprobs_dict["req"], expected)


def test_async_mm_snapshot_owns_output_until_copy_finishes() -> None:
    runner = OmniARModelRunner.__new__(OmniARModelRunner)
    runner.model_config = SimpleNamespace(async_chunk=True)
    runner.model = SimpleNamespace()
    runner._async_mm_snapshot_slots, runner._async_mm_snapshot_events = [{}], [None]
    runner._async_mm_snapshot_pending, runner._async_mm_snapshot_cursor = [False], 0
    runner._last_multimodal_snapshot_slot = None
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


def test_producer_snapshot_is_not_repacked() -> None:
    runner = OmniARModelRunner.__new__(OmniARModelRunner)
    runner.model_config = SimpleNamespace(async_chunk=True)
    runner.model = SimpleNamespace()
    runner._async_mm_snapshot_slots, runner._async_mm_snapshot_events = [{}], [None]
    runner._async_mm_snapshot_pending, runner._async_mm_snapshot_cursor = [False], 0
    runner._last_multimodal_snapshot_slot = None
    source = torch.tensor([[7, 8]], dtype=torch.float32)
    snapshot = pack_output_snapshot({"audio": source}, {}, max_buckets=1)
    assert isinstance(snapshot, PackedOutputSnapshot)

    retained = runner._retain_multimodal_outputs(snapshot)

    assert retained is snapshot
    assert runner._last_multimodal_snapshot_slot is None
    assert runner._async_mm_snapshot_pending == [False]


def test_fresh_per_step_outputs_are_not_repacked() -> None:
    runner = OmniARModelRunner.__new__(OmniARModelRunner)
    runner.model_config = SimpleNamespace(async_chunk=True)
    runner.model = SimpleNamespace(mm_outputs_fresh_per_step=True)
    runner._last_multimodal_snapshot_slot = None
    outputs = {"model_outputs": torch.ones(2, 4)}

    assert runner._retain_multimodal_outputs(outputs) is outputs
    assert runner._last_multimodal_snapshot_slot is None


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


@pytest.mark.parametrize("async_chunk", [False, True])
@pytest.mark.parametrize(
    "hidden_padded,codes_padded",
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["unpadded", "hidden_padded", "codes_padded", "both_padded"],
)
@pytest.mark.parametrize("lengths", [(1, 1, 1), (3, 1, 1)], ids=["decode", "mixed_prefill_decode"])
def test_async_output_slices_request_payloads_with_graph_padding(
    monkeypatch, mocker, async_chunk, hidden_padded, codes_padded, lengths
):
    """Sync and async transfers must exclude other requests and graph padding."""
    from vllm.v1.worker.gpu.input_batch import InputBatch

    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    total = sum(lengths)
    padded_total = 8 if hidden_padded or codes_padded else total
    offsets = np.cumsum([0, *lengths])
    hidden_rows = padded_total if hidden_padded else total
    code_rows = padded_total if codes_padded else total
    hidden = torch.arange(hidden_rows * 4, dtype=torch.float32).reshape(hidden_rows, 4)
    codes = torch.arange(code_rows * 16).reshape(code_rows, 16)
    # Reference frames are request-local, even if their length matches the
    # padded token count. They must not be sliced along the batch token axis.
    refs = [torch.arange(padded_total * 16).reshape(padded_total, 16), torch.empty(0), torch.empty(0)]
    batch = mocker.Mock(spec=InputBatch)
    batch.query_start_loc_np = offsets
    batch.num_scheduled_tokens = np.array(lengths)
    batch.num_reqs = len(lengths)
    batch.num_tokens_after_padding = padded_total
    output = _async_output(
        req_ids=[f"req-{i}" for i in range(len(lengths))],
        sampler_output=SamplerOutput(
            torch.ones(len(lengths), 1, dtype=torch.long),
            None,
            None,
            torch.ones(len(lengths), dtype=torch.long),
            torch.zeros(len(lengths), dtype=torch.long),
        ),
        text_hidden=hidden,
        multimodal_outputs={"codes": {"audio": codes, "ref": refs}},
        input_batch=batch,
        async_chunk=async_chunk,
    ).get_output()

    for i, payload in enumerate(output.inter_stage_outputs):
        torch.testing.assert_close(payload["codes.audio"], codes[offsets[i] : offsets[i + 1]])
        torch.testing.assert_close(payload["codes.ref"], refs[i])
        if not async_chunk:
            torch.testing.assert_close(payload["hidden"], hidden[offsets[i] : offsets[i + 1]])


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


def test_stream_audio_cancel_rejects_inflight_output_and_id_reuse() -> None:
    from vllm_omni.worker_v2.streaming_audio import StreamingAudioBuffer

    buffer = StreamingAudioBuffer(samples_per_frame=4, chunk_frames=3)
    cancelled = buffer.add("request")
    buffer.finish({"request"})
    replacement = buffer.add("request")
    assert cancelled.push(torch.ones(4), False) is None
    assert not cancelled.pending
    assert replacement.push(torch.full((4,), 2.0), False).tolist() == [2.0] * 4
    buffer.finish({"request"})
    assert not buffer.requests


def test_stream_audio_length_end_flushes_first_and_partial_chunks() -> None:
    from vllm_omni.worker_v2.streaming_audio import StreamingAudioBuffer, StreamingAudioOutput

    buffer = StreamingAudioBuffer(samples_per_frame=4, chunk_frames=3)
    state = buffer.add("request")
    output = StreamingAudioOutput(
        torch.ones(1, 4),
        torch.tensor([True]),
        torch.tensor(24000),
        None,
        np.array([0, 1]),
        [state],
        np.array([True]),
    )
    assert output.get_output()[0]["model_outputs"].numel() == 4
    assert not state.active
    state = buffer.add("another")
    assert state.push(torch.ones(4), False).numel() == 4
    assert state.push(torch.ones(4), False) is None
    assert state.push(torch.ones(4), True).numel() == 8


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.bfloat16])
def test_stream_audio_batch_views_retain_pcm_after_output_release(dtype):
    from vllm_omni.worker_v2.streaming_audio import StreamingAudioBuffer, StreamingAudioOutput

    buffer = StreamingAudioBuffer(samples_per_frame=4, chunk_frames=3)
    states = [buffer.add("a"), buffer.add("b")]
    expected: list[list[torch.Tensor]] = [[], []]
    emitted: list[list[torch.Tensor]] = [[], []]
    for step in range(5):
        # Mixed prefill/decode spans, invalid rows, and a final partial chunk.
        wav = (torch.arange(16).reshape(4, 4) + 100 * step).to(dtype)
        valid = torch.tensor([False, True, True, step != 4])
        output = StreamingAudioOutput(
            wav,
            valid,
            torch.tensor(24000),
            None,
            np.array([0, 3, 4]),
            states,
            np.array([step == 4, step == 4]),
        )
        for i, (start, end) in enumerate([(0, 3), (3, 4)]):
            expected[i].append(wav[start:end][valid[start:end]].reshape(-1).clone())
        for i, payload in enumerate(output.get_output()):
            if payload is not None:
                emitted[i].append(payload["model_outputs"])
        del output, wav  # Pending views must keep the old host allocation alive.
    for i in range(2):
        assert torch.equal(torch.cat(emitted[i]), torch.cat(expected[i]))
        assert torch.cat(emitted[i]).dtype == dtype
        assert not states[i].active and not states[i].pending
        assert states[i].pending_samples == 0


def test_stream_audio_numpy_direct_first_cancel_and_mixed_fallback():
    from vllm_omni.worker_v2.streaming_audio import StreamingAudioBuffer

    buffer = StreamingAudioBuffer(samples_per_frame=4, chunk_frames=3)
    state = buffer.add("r")
    state.accept_first_audio()
    assert state.push(np.arange(8, dtype=np.float32), False) is None
    assert state.pending_samples == 4
    # A dtype fallback can arrive while a NumPy frame is pending.
    chunk = state.push(torch.arange(8, 12, dtype=torch.bfloat16), True)
    assert chunk.tolist() == list(range(4, 12))
    assert state.pending_samples == 0
    buffer.finish({"r"})
    replacement = buffer.add("r")
    assert state.push(np.ones(4, dtype=np.float32), False) is None
    assert replacement.push(np.full(4, 99, dtype=np.float32), False).tolist() == [99] * 4
    replacement.push(np.ones(4, dtype=np.float32), False)
    assert replacement.pending_samples == 4
    buffer.finish({"r"})
    assert replacement.pending_samples == 0 and not replacement.pending


def test_stream_audio_prefill_length_end_reaches_output(mocker) -> None:
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.states import RequestState

    from vllm_omni.worker_v2.model_states.eager_mtp import EagerMTPState
    from vllm_omni.worker_v2.streaming_audio import StreamingAudioBuffer

    owner = mocker.Mock()
    owner.vllm_config.model_config.max_model_len = 4096
    owner._stream_decode_event = None
    eager = EagerMTPState(owner)
    eager._audio_buffer = StreamingAudioBuffer(4, 25)
    eager._audio_buffer.add("request")
    eager._frame_requests = {"request"}
    batch = InputBatch.__new__(InputBatch)
    batch.num_reqs, batch.req_ids = 1, ["request"]
    batch.num_computed_tokens_np = np.array([0])
    batch.num_scheduled_tokens = np.array([3])
    batch.idx_mapping_np = np.array([0])
    batch.query_start_loc_np = np.array([0, 3])
    batch.is_prefilling_np = np.array([True])
    states = RequestState.__new__(RequestState)
    states.max_seq_len = np.array([4])
    mm = {
        "model_outputs": torch.ones(3, 4),
        "meta": {"codec_frame_valid": torch.tensor([0, 0, 1])},
        "sr": [torch.tensor(24000)],
    }
    output = eager.prepare_audio_output(batch, states, mm)
    assert output.get_output()[0]["model_outputs"].numel() == 4
    assert "model_outputs" not in mm


def test_stream_audio_preemption_preserves_position_rng_and_pending_pcm(mocker) -> None:
    from vllm_omni.worker_v2.model_states.eager_mtp import EagerMTPState, TalkerInputs
    from vllm_omni.worker_v2.streaming_audio import StreamingAudioBuffer

    owner = mocker.Mock()
    owner._stream_pos = {"request": 37}
    owner._eager_embeds = torch.arange(18).reshape(6, 3).float()
    owner._eager_ready = {}
    owner.intermediate_buffer.buffers = [{} for _ in range(6)]
    generator = torch.Generator().manual_seed(42)
    owner._mtp_generators = {"request": generator}
    saved = [torch.arange(6)]
    owner.model.stream_decoder.save_slot.return_value = saved
    eager = EagerMTPState(owner)
    eager._talker_inputs["request"] = TalkerInputs(torch.ones(20, 3), 10)
    eager._side_stream = mocker.Mock()
    eager._audio_buffer = StreamingAudioBuffer(4, 25)
    request = eager._audio_buffer.add("request")
    request.push(torch.ones(4), False)
    request.push(torch.ones(4), False)
    eager.suspend_audio("request", 2)
    owner._stream_pos.clear()
    owner._mtp_generators.clear()
    eager.resume_audio("request", 5)
    assert owner._stream_pos["request"] == 37
    assert owner._mtp_generators["request"] is generator
    assert eager._restore_audio["request"] is saved
    assert sum(part.numel() for part in request.pending) == 4
    # A second preemption can happen before the restored slot is used.
    eager.suspend_audio("request", 5)
    owner.model.stream_decoder.save_slot.assert_called_once_with(2)
    eager.finish_audio({"request"})
    assert not eager._suspended_audio and not eager._restore_audio
    assert not eager._audio_buffer.requests and not request.active


def test_stream_audio_replay_uses_exact_inputs_and_consumes_last_frame_once(mocker):
    from vllm_omni.worker_v2.model_states.eager_mtp import EagerMTPState, TalkerInputs

    owner = mocker.Mock()
    owner._eager_embeds = torch.tensor([[20.0, 30.0]])
    owner.model.preprocess.return_value = (None, None, {"mtp_inputs": (None, torch.tensor([[2.0, 3.0]]))})
    owner.intermediate_buffer.buffers = [{"req_id": "request"}]
    eager = EagerMTPState(owner)
    eager._restore_audio["request"] = []
    history = torch.arange(8).reshape(4, 2).float()
    eager._talker_inputs["request"] = TalkerInputs(history, 4)
    first = torch.empty(2, 2)
    assert eager.replay_inputs("request", 0, 0, torch.ones(2), first)
    torch.testing.assert_close(first, history[:2])
    owner.model.preprocess.assert_not_called()
    last = torch.empty(3, 2)
    assert eager.replay_inputs("request", 0, 2, torch.ones(3), last)
    torch.testing.assert_close(last, torch.cat((history[2:], torch.tensor([[22.0, 33.0]]))))
    owner.model.preprocess.assert_called_once()
    with pytest.raises(RuntimeError, match="exceeds saved inputs"):
        eager.replay_inputs("request", 0, 4, torch.ones(2), torch.empty(2, 2))


def test_stream_reference_priming_groups_lengths_without_padding_or_slot_aliasing(mocker):
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import Qwen3TTSTalkerForConditionalGeneration

    primes = [(4, torch.full((26, 2), 4)), (1, torch.full((3, 2), 1)), (7, torch.full((26, 2), 7))]
    calls = []

    def stream(codes, slots, pos):
        calls.append((codes.clone(), slots.clone(), pos.clone()))

    decoder = mocker.Mock(side_effect=stream, device=torch.device("cpu"))
    model = mocker.Mock(stream_decoder=decoder, stream_prime_graphs=None, stream_chunk_frames=25)
    Qwen3TTSTalkerForConditionalGeneration.prime_stream_decoder(model, primes)
    assert len(calls) == 3
    for (codes, slots, pos), frames, indices, position in zip(
        calls, [25, 1, 3], [[4, 7], [4, 7], [1]], [0, 25, 0], strict=True
    ):
        assert codes.shape == (len(indices), frames, 2)
        assert codes.dtype == slots.dtype == pos.dtype == torch.int32
        assert slots.tolist() == indices and pos.tolist() == [position] * len(indices)
        for row, idx in enumerate(indices):
            assert torch.all(codes[row] == idx)
    Qwen3TTSTalkerForConditionalGeneration.prime_stream_decoder(model, [])
    assert len(calls) == 3


@pytest.mark.parametrize("context_frames", [25, 72])
def test_model_owned_reference_context_respects_configured_boundary(mocker, context_frames):
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import Qwen3TTSTalkerForConditionalGeneration

    model = mocker.Mock(stream_ref_context_frames=context_frames, talker_config=mocker.Mock(num_code_groups=16))
    reference = torch.arange(100 * 16).reshape(100, 16)
    info = {"codes": {"ref": reference}}
    got = Qwen3TTSTalkerForConditionalGeneration.get_stream_ref_context(model, info)
    torch.testing.assert_close(got, reference[-context_frames:])
    assert info["codes"]["ref"] is reference
    assert Qwen3TTSTalkerForConditionalGeneration.get_stream_ref_context(model, {}) is None
    assert (
        Qwen3TTSTalkerForConditionalGeneration.get_stream_ref_context(model, {"codes": {"ref": torch.empty(0)}}) is None
    )


@pytest.mark.parametrize("streaming", [False, True])
def test_model_owned_audio_finalizer_runs_after_copy_without_hidden(monkeypatch, streaming):
    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    batch = SimpleNamespace(
        num_reqs=2,
        query_start_loc_np=np.array([0, 1, 4]),
        num_scheduled_tokens=np.array([1, 3]),
        num_tokens_after_padding=4,
    )
    snapshot = torch.tensor([[10, 11], [20, 21]])
    seen = []

    def finalize(payload, counts):
        seen.append(counts)
        return {"codes": {"audio": [payload["snapshot"][0:1].clone(), torch.empty(0, dtype=torch.long)]}}

    result = _async_output(
        req_ids=["audio", "partial"],
        sampler_output=SamplerOutput(
            torch.tensor([[99], [100]]), None, None, torch.tensor([1, 1]), torch.tensor([0, 0])
        ),
        num_sampled_tokens=torch.tensor([1, 0]),
        multimodal_outputs={"snapshot": snapshot},
        input_batch=batch,
        async_chunk=streaming,
        finalize_multimodal=finalize,
    ).get_output()
    snapshot.zero_()
    assert seen == [[1, 0]]
    assert result.sampled_token_ids == [[99], []]
    assert result.inter_stage_outputs[0]["codes.audio"].tolist() == [[10, 11]]
    assert result.inter_stage_outputs[1]["codes.audio"].numel() == 0
    assert (result.pooler_output is None) == streaming


def test_stream_audio_history_replays_exact_inputs_after_batch_reordering(mocker):
    from vllm.v1.worker.gpu.input_batch import InputBatch

    from vllm_omni.worker_v2.model_states.eager_mtp import EagerMTPState

    owner = mocker.Mock()
    owner.vllm_config.model_config.max_model_len = 12
    eager = EagerMTPState(owner)
    first = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    second = torch.arange(9, 18, dtype=torch.float32).reshape(3, 3)
    expected = {"a": torch.cat((first[:2], second[2:])), "b": torch.cat((first[2:], second[:2]))}
    batch = InputBatch.__new__(InputBatch)
    batch.req_ids = ["a", "b"]
    batch.query_start_loc_np = np.array([0, 2, 3])
    batch.num_computed_tokens_np = np.array([0, 0])
    eager.record_inputs(batch, first)
    batch.req_ids = ["b", "a"]
    batch.num_computed_tokens_np = np.array([1, 2])
    eager.record_inputs(batch, second)
    # Later input-buffer reuse must not change the conditioned replay history.
    first.fill_(-1)
    second.fill_(-2)
    for request_id, wanted in expected.items():
        eager._restore_audio[request_id] = []
        replayed = torch.empty_like(wanted)
        assert eager.replay_inputs(request_id, 0, 0, torch.zeros(3, dtype=torch.long), replayed)
        torch.testing.assert_close(replayed, wanted)


@pytest.mark.parametrize("valid", [False, True])
def test_stream_audio_direct_first_requires_only_real_pcm_on_terminal(valid):
    from vllm_omni.data_entry_keys import FIRST_AUDIO_REQUIRED_KEY
    from vllm_omni.worker_v2.streaming_audio import StreamingAudioBuffer, StreamingAudioOutput

    buffer = StreamingAudioBuffer(samples_per_frame=4, chunk_frames=3)
    state = buffer.add("r")
    state.accept_first_audio()
    first = StreamingAudioOutput(
        torch.arange(4).reshape(1, 4),
        torch.tensor([valid]),
        torch.tensor(24000),
        None,
        np.array([0, 1]),
        [state],
        np.array([True]),
    )
    payload = first.get_output()[0]
    if valid:
        assert payload == {FIRST_AUDIO_REQUIRED_KEY: torch.tensor(True)}
    else:
        assert payload is None  # EOS has no promised first PCM to wait for.
    assert not state.active and not state.pending
    buffer.finish({"r"})
    replacement = buffer.add("r")
    assert replacement.push(torch.full((4,), 99), False).tolist() == [99] * 4


def test_stream_audio_direct_first_drops_one_frame_and_retains_partial_suffix():
    from vllm_omni.worker_v2.streaming_audio import AudioRequest

    state = AudioRequest(samples_per_frame=4, chunk_frames=3)
    state.accept_first_audio()
    assert state.push(torch.arange(4), False) is None
    assert state.first_audio_required and state.emitted
    assert state.push(torch.arange(4, 8), False) is None
    chunk = state.push(torch.arange(8, 12), True)
    assert chunk.tolist() == list(range(4, 12))
    assert not state.active and not state.pending


@pytest.mark.parametrize("consumer", ["pcm_only", "finalizer", "extra"])
def test_stream_audio_snapshot_skips_discarded_code_partition(monkeypatch, mocker, consumer):
    from vllm.v1.worker.gpu.input_batch import InputBatch

    from vllm_omni.worker_v2.output_snapshot import RequestOutputSnapshot
    from vllm_omni.worker_v2.streaming_audio import StreamingAudioBuffer, StreamingAudioOutput

    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    monkeypatch.setattr(
        OmniARModelRunner,
        "_build_async_chunk_outputs_from_mm",
        lambda *args: pytest.fail("in-stage PCM unnecessarily partitioned code outputs"),
    )
    batch = InputBatch.__new__(InputBatch)
    batch.num_reqs = 3
    batch.query_start_loc_np = np.array([0, 1, 2, 4])
    batch.num_scheduled_tokens = np.array([1, 1, 2])
    batch.num_tokens_after_padding = 4
    buffer = StreamingAudioBuffer(samples_per_frame=4, chunk_frames=3)
    states = [buffer.add(request_id) for request_id in ["frame", "eos", "prefill"]]
    wav = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    streaming_audio = StreamingAudioOutput(
        wav,
        torch.tensor([True, False, True, False]),
        torch.tensor(24000),
        None,
        batch.query_start_loc_np.copy(),
        states,
        np.array([False, False, True]),
    )
    copy_mm = mocker.spy(omni_ar_model_runner, "_async_copy_mm")

    def finalize(payload, counts):
        assert payload["codes"]["audio"].shape == (4, 2)
        return RequestOutputSnapshot([None] * batch.num_reqs)

    finalizer = mocker.Mock(side_effect=finalize) if consumer == "finalizer" else None
    extra = ({"extra": torch.tensor([7])}, _FakeEvent()) if consumer == "extra" else None
    result = _async_output(
        req_ids=["frame", "eos", "prefill"],
        sampler_output=SamplerOutput(torch.tensor([[1], [2], [3]]), None, None, None, None),
        num_sampled_tokens=torch.tensor([1, 1, 1]),
        multimodal_outputs={"codes": {"audio": torch.ones(4, 2)}},
        input_batch=batch,
        async_chunk=True,
        streaming_audio=streaming_audio,
        finalize_multimodal=finalizer,
        extra_multimodal_outputs=extra,
    ).get_output()
    assert copy_mm.call_count == {"pcm_only": 0, "finalizer": 1, "extra": 2}[consumer]
    if finalizer is not None:
        finalizer.assert_called_once()
    wav.fill_(-1)
    assert result.inter_stage_outputs is None
    assert result.multimodal_outputs[0]["model_outputs"].tolist() == [0, 1, 2, 3]
    assert result.multimodal_outputs[1] == {}
    assert result.multimodal_outputs[2]["model_outputs"].tolist() == [8, 9, 10, 11]
    assert not states[1].active and not states[2].active


@pytest.mark.parametrize("streaming", [False, True])
def test_request_owned_snapshot_skips_generic_partition(monkeypatch, streaming):
    from vllm_omni.worker_v2.output_snapshot import RequestOutputSnapshot

    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    monkeypatch.setattr(
        OmniARModelRunner,
        "_build_async_chunk_outputs_from_mm",
        lambda *args: pytest.fail("already partitioned payload was repartitioned"),
    )
    batch = SimpleNamespace(
        num_reqs=2,
        query_start_loc_np=np.array([0, 1, 4]),
        num_scheduled_tokens=np.array([1, 3]),
        num_tokens_after_padding=4,
    )
    snapshot = torch.tensor([[10, 11], [20, 21]])

    def finalize(payload, counts):
        return RequestOutputSnapshot([{"codes.audio": payload["snapshot"][0:1].clone()}, None])

    result = _async_output(
        req_ids=["audio", "partial"],
        sampler_output=SamplerOutput(
            torch.tensor([[99], [100]]), None, None, torch.tensor([1, 1]), torch.tensor([0, 0])
        ),
        num_sampled_tokens=torch.tensor([1, 0]),
        multimodal_outputs={"snapshot": snapshot},
        input_batch=batch,
        async_chunk=streaming,
        finalize_multimodal=finalize,
    ).get_output()
    snapshot.zero_()
    assert result.inter_stage_outputs[0]["codes.audio"].tolist() == [[10, 11]]
    assert result.inter_stage_outputs[1] is None
    assert (result.pooler_output is None) == streaming
