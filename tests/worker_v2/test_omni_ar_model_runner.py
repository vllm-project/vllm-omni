"""Unit tests for OmniARModelRunner v2."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import vllm_omni.worker_v2.omni_ar_model_runner as omni_ar_model_runner
from vllm_omni.data_entry_keys import unflatten_payload
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker_v2.omni_ar_model_runner import (
    OmniARModelRunner,
    OmniAsyncOutput,
    _async_copy_mm,
    _partition_pooler_outputs,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_async_output_preserves_routing_and_sampling_masks(monkeypatch) -> None:
    from vllm.v1.outputs import RoutedExpertsTensors
    from vllm.v1.worker.gpu.sample.output import SamplerOutput, SamplingMaskTensors

    class FakeStream:
        def wait_stream(self, _stream) -> None:
            pass

    class FakeEvent:
        def record(self, _stream) -> None:
            pass

        def synchronize(self) -> None:
            pass

    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    routing = RoutedExpertsTensors(torch.tensor([[[2, 3]], [[4, 5]]]), torch.tensor([7, 9]))
    masks = SamplingMaskTensors(torch.tensor([[5], [0]], dtype=torch.uint8), torch.tensor([2, 0]), 4)
    runner_output = omni_ar_model_runner.OmniModelRunnerOutput(
        req_ids=["decode", "prefill"],
        req_id_to_index={"decode": 0, "prefill": 1},
        sampled_token_ids=None,
        prompt_logprobs_dict={},
    )
    output = OmniAsyncOutput(
        model_runner_output=runner_output,
        sampler_output=SamplerOutput(
            sampled_token_ids=torch.tensor([[2], [0]]),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=torch.tensor([1, 0]),
            sampling_mask_tensors=masks,
        ),
        num_sampled_tokens=torch.tensor([1, 0]),
        main_stream=FakeStream(),
        copy_stream=FakeStream(),
        copy_event=FakeEvent(),
        routed_experts=routing,
    ).get_output()

    assert output.sampled_token_ids == [[2], []]
    np.testing.assert_array_equal(output.routed_experts.routing_data, [[[2, 3]], [[4, 5]]])
    np.testing.assert_array_equal(output.routed_experts.slot_mapping, [7, 9])
    np.testing.assert_array_equal(output.sampling_masks.token_ids, [0, 2])
    np.testing.assert_array_equal(output.sampling_masks.offsets, [0, 2])
    assert output.sampling_masks.cu_num_generated_tokens == [0, 1, 1]


# ---------------------------------------------------------------
# _build_pooler_output_from_cpu (was _build_pooler_output)
# ---------------------------------------------------------------


def test_reconstruct_raw_model_output_preserves_omni_multimodal_outputs():
    hidden = torch.randn(3, 4)
    latent = torch.randn(3, 4)
    raw = OmniARModelRunner._reconstruct_raw_model_output(
        hidden_states=hidden,
        multimodal_outputs={"latent": latent},
        aux=None,
    )

    assert isinstance(raw, OmniOutput)
    assert raw.text_hidden_states is hidden
    assert raw.multimodal_outputs["latent"] is latent


def test_reconstruct_raw_model_output_keeps_aux_tuple_without_multimodal_outputs():
    hidden = torch.randn(3, 4)
    aux = {"layers": torch.randn(3, 2)}
    raw = OmniARModelRunner._reconstruct_raw_model_output(
        hidden_states=hidden,
        multimodal_outputs=None,
        aux=aux,
    )

    assert raw == (hidden, aux)


def test_reconstruct_raw_model_output_ignores_empty_multimodal_outputs():
    hidden = torch.randn(3, 4)
    raw = OmniARModelRunner._reconstruct_raw_model_output(
        hidden_states=hidden,
        multimodal_outputs={},
        aux=None,
    )

    assert raw is hidden


def test_non_last_pp_rank_uses_vllm_027_pp_handler() -> None:
    runner = OmniARModelRunner.__new__(OmniARModelRunner)
    input_batch = SimpleNamespace(idx_mapping=torch.tensor([0]))
    connector_output = MagicMock()
    connector_output.is_empty.return_value = False
    runner._kv_extracted_req_ids = None
    runner.execute_model_state = SimpleNamespace(
        input_batch=input_batch,
        hidden_states=None,
        finished_req_ids={"finished"},
        ec_connector_output=None,
        routed_experts=None,
    )
    runner.kv_connector = SimpleNamespace(
        post_forward=MagicMock(return_value=connector_output),
    )
    runner.is_last_pp_rank = False

    def receive(batch):
        assert batch is input_batch
        assert torch.is_inference_mode_enabled()
        return False

    runner.pp_handler = SimpleNamespace(receive=MagicMock(side_effect=receive))
    runner.postprocess_num_computed_tokens = MagicMock()
    runner.model_state = SimpleNamespace(postprocess_state=MagicMock())
    runner.eplb = SimpleNamespace(step=MagicMock())

    output = runner.sample_tokens(None)

    runner.pp_handler.receive.assert_called_once_with(input_batch)
    runner.postprocess_num_computed_tokens.assert_called_once_with(input_batch)
    runner.model_state.postprocess_state.assert_called_once_with(input_batch.idx_mapping, 0)
    runner.kv_connector.post_forward.assert_called_once_with({"finished"})
    assert output.kv_connector_output is connector_output


def test_last_pp_rank_runs_connector_after_sampling_state_is_finalized(monkeypatch) -> None:
    runner = OmniARModelRunner.__new__(OmniARModelRunner)
    input_batch = SimpleNamespace(
        req_ids=["req"],
        idx_mapping=torch.tensor([0]),
        query_start_loc=torch.tensor([0, 1]),
        num_reqs=1,
        seq_lens=torch.tensor([3]),
    )
    runner._kv_extracted_req_ids = None
    runner.execute_model_state = SimpleNamespace(
        input_batch=input_batch,
        hidden_states=torch.zeros(1, 2),
        finished_req_ids={"finished"},
        ec_connector_output=None,
        routed_experts=None,
    )
    runner._last_aux_output = None
    runner._last_multimodal_outputs = None
    runner._last_multimodal_snapshot_slot = None
    runner.is_last_pp_rank = True
    runner.pp_handler = None
    runner.model_config = SimpleNamespace(async_chunk=False)
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(engine_output_type="text"))
    runner.model_state = SimpleNamespace(postprocess_model_output=MagicMock(return_value=(torch.zeros(1, 2), None)))
    runner.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=torch.tensor([[1]])),
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([0])),
        prompt_len=SimpleNamespace(np=np.array([1]), gpu=torch.tensor([1])),
    )
    runner.sample = MagicMock(
        return_value=(
            SimpleNamespace(sampled_token_ids=torch.tensor([[2]])),
            torch.tensor([1]),
            torch.tensor([0]),
        )
    )
    runner.model = SimpleNamespace(compute_logits=MagicMock())
    runner.prompt_logprobs_worker = SimpleNamespace(compute_prompt_logprobs=MagicMock(return_value={}))
    runner.postprocess_sampled = MagicMock()
    connector_output = object()

    def post_forward(finished_req_ids):
        runner.postprocess_sampled.assert_called_once()
        assert finished_req_ids == {"finished"}
        return connector_output

    runner.kv_connector = SimpleNamespace(post_forward=MagicMock(side_effect=post_forward))
    runner.main_stream = MagicMock()
    runner.output_copy_stream = MagicMock()
    runner._finalize_native_data_plane_output = MagicMock()
    runner.check_ep_fault = False
    runner._reserve_native_data_plane_outputs = MagicMock()
    runner.eplb = SimpleNamespace(step=MagicMock())

    async_output = SimpleNamespace(copy_event=None)
    monkeypatch.setattr(
        omni_ar_model_runner,
        "OmniAsyncOutput",
        MagicMock(return_value=async_output),
    )

    output = runner.sample_tokens(None)

    assert output is async_output
    model_runner_output = omni_ar_model_runner.OmniAsyncOutput.call_args.kwargs["model_runner_output"]
    assert model_runner_output.kv_connector_output is connector_output


def test_async_output_uses_blocking_cuda_event_by_default(monkeypatch) -> None:
    class FakeStream:
        def wait_stream(self, _stream) -> None:
            pass

    class FakeEvent:
        def record(self, _stream) -> None:
            pass

        def synchronize(self) -> None:
            pass

    event_kwargs = []

    def make_event(**kwargs):
        event_kwargs.append(kwargs)
        return FakeEvent()

    monkeypatch.setattr(torch.cuda, "Event", make_event)
    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)

    OmniAsyncOutput(
        model_runner_output=SimpleNamespace(
            req_ids=["req-0"],
            sampled_token_ids=None,
            prompt_logprobs_dict={},
        ),
        sampler_output=SimpleNamespace(
            sampled_token_ids=torch.tensor([[123]], dtype=torch.long),
            logprobs_tensors=None,
            num_nans=None,
            sampling_mask_tensors=None,
        ),
        num_sampled_tokens=torch.tensor([1], dtype=torch.long),
        main_stream=FakeStream(),
        copy_stream=FakeStream(),
    )

    assert event_kwargs == [{"blocking": True}]


def test_async_mm_snapshot_owns_graph_output_until_copy_finishes() -> None:
    runner = OmniARModelRunner.__new__(OmniARModelRunner)
    runner.model_config = SimpleNamespace(async_chunk=True)
    runner._async_mm_snapshot_slots = [{}]
    runner._async_mm_snapshot_events = [None]
    runner._async_mm_snapshot_pending = [False]
    runner._async_mm_snapshot_cursor = 0
    runner._last_multimodal_snapshot_slot = None
    runner.main_stream = SimpleNamespace(wait_event=lambda _event: None)
    source = torch.tensor([[7, 8]], dtype=torch.long)

    snapshot = runner._retain_multimodal_outputs({"codes": {"audio": source}})
    source.fill_(99)

    assert snapshot["codes"]["audio"].tolist() == [[7, 8]]
    assert runner._last_multimodal_snapshot_slot == 0
    assert runner._async_mm_snapshot_pending == [True]


def test_async_mm_snapshot_waits_for_d2h_before_reusing_slot() -> None:
    runner = OmniARModelRunner.__new__(OmniARModelRunner)
    runner.model_config = SimpleNamespace(async_chunk=True)
    runner._async_mm_snapshot_slots = [{}]
    runner._async_mm_snapshot_events = [None]
    runner._async_mm_snapshot_pending = [False]
    runner._async_mm_snapshot_cursor = 0
    runner._last_multimodal_snapshot_slot = None
    waited = []
    runner.main_stream = SimpleNamespace(wait_event=waited.append)
    copy_event = object()

    runner._retain_multimodal_outputs({"codes": {"audio": torch.ones(1, 2)}})
    runner._release_multimodal_snapshot(0, copy_event)
    runner._retain_multimodal_outputs({"codes": {"audio": torch.zeros(1, 2)}})

    assert waited == [copy_event]


def test_async_mm_snapshot_can_retain_postprocessed_payload() -> None:
    runner = OmniARModelRunner.__new__(OmniARModelRunner)
    runner.model_config = SimpleNamespace(async_chunk=True)
    runner._async_mm_snapshot_slots = [{}]
    runner._async_mm_snapshot_events = [None]
    runner._async_mm_snapshot_pending = [False]
    runner._async_mm_snapshot_cursor = 0
    runner._last_multimodal_snapshot_slot = None
    runner.main_stream = SimpleNamespace(wait_event=lambda _event: None)
    source = {"codes": {"audio": torch.tensor([[1, 2, 3]])}}

    retained = runner._retain_multimodal_outputs(source)

    assert retained["codes"]["audio"].data_ptr() != source["codes"]["audio"].data_ptr()


def test_async_mm_snapshot_keeps_separate_shape_buckets() -> None:
    slot = {}

    first = omni_ar_model_runner._copy_mm_to_snapshot_slot(torch.ones(1, 2), slot)
    second = omni_ar_model_runner._copy_mm_to_snapshot_slot(torch.ones(4, 2), slot)

    assert len(slot) == 2
    assert first.shape == (1, 2)
    assert second.shape == (4, 2)


def test_async_mm_snapshot_cache_is_bounded_without_evicting_existing_buffers(monkeypatch) -> None:
    monkeypatch.setattr(omni_ar_model_runner, "_ASYNC_MM_SNAPSHOT_MAX_BUCKETS_PER_SLOT", 1)
    slot = {}
    first = omni_ar_model_runner._copy_mm_to_snapshot_slot(torch.ones(1, 2), slot)
    second = omni_ar_model_runner._copy_mm_to_snapshot_slot(torch.ones(4, 2), slot)

    assert len(slot) == 1
    assert next(iter(slot.values())) is first
    assert second.shape == (4, 2)
    assert second.data_ptr() != first.data_ptr()


def test_non_async_pooler_copy_blocks_next_graph_replay():
    main_stream = MagicMock()
    copy_event = object()

    omni_ar_model_runner._guard_graph_replay_for_pooler_copy(
        main_stream,
        copy_event,
        need_pooler=True,
        async_chunk=False,
    )

    main_stream.wait_event.assert_called_once_with(copy_event)


def test_has_cuda_tensor_recurses_nested_payloads() -> None:
    assert not omni_ar_model_runner._has_cuda_tensor({"codes": {"audio": torch.ones(1)}})


def test_partition_pooler_outputs_splits_async_chunk_payload():
    payload = {
        "hidden": torch.randn(2, 4),
        "codes.audio": torch.ones(2, 3),
        "audio": torch.randn(160),
        "sr": torch.tensor(24000),
    }

    inter_stage, client = _partition_pooler_outputs([payload], async_chunk=True)

    assert inter_stage is not None
    assert client is not None
    assert set(inter_stage[0]) == {"hidden", "codes.audio"}
    assert inter_stage[0]["hidden"] is payload["hidden"]
    assert inter_stage[0]["codes.audio"] is payload["codes.audio"]
    assert set(client[0]) == {"audio", "sr"}
    assert client[0]["audio"] is payload["audio"]
    assert client[0]["sr"] is payload["sr"]


def test_partition_pooler_outputs_keeps_full_payload_without_async_chunk():
    payload = {"hidden": torch.randn(2, 4), "audio": torch.randn(160)}

    inter_stage, client = _partition_pooler_outputs([payload], async_chunk=False)

    assert inter_stage == [payload]
    assert client == [payload]


def test_build_pooler_output_basic():
    """Verify _build_pooler_output_from_cpu slices per-request hidden + mm."""
    hidden = torch.randn(6, 8)
    mm = {"audio": torch.randn(6, 2)}

    pooler = OmniARModelRunner._build_pooler_output_from_cpu(
        hidden,
        mm,
        query_start_loc_np=np.array([0, 3]),
        num_scheduled_tokens=np.array([3, 3], dtype=np.int32),
        num_reqs=2,
    )

    assert len(pooler) == 2
    assert pooler[0]["hidden"].shape == (3, 8)
    assert pooler[1]["hidden"].shape == (3, 8)
    assert pooler[0]["audio"].shape == (3, 2)


def test_build_pooler_output_hidden_slice_has_owned_storage():
    hidden = torch.randn(67, 2048)

    pooler = OmniARModelRunner._build_pooler_output_from_cpu(
        hidden,
        {},
        query_start_loc_np=np.array([66]),
        num_scheduled_tokens=np.array([1], dtype=np.int32),
        num_reqs=1,
    )

    slice_hidden = pooler[0]["hidden"]
    assert slice_hidden.is_contiguous()
    assert slice_hidden.untyped_storage().nbytes() == slice_hidden.numel() * slice_hidden.element_size()


def test_build_async_chunk_outputs_slices_graph_padded_token_axis() -> None:
    padded_codes = torch.arange(16, dtype=torch.long).reshape(8, 2)

    inter_stage, client = OmniARModelRunner._build_async_chunk_outputs_from_mm(
        {"codes": {"audio": padded_codes}},
        query_start_loc_np=np.array([0, 1, 2]),
        num_scheduled_tokens=np.array([1, 1], dtype=np.int32),
        num_reqs=2,
        total_tokens=2,
        padded_total_tokens=8,
    )

    assert client is None
    assert inter_stage is not None
    assert torch.equal(inter_stage[0]["codes.audio"], padded_codes[0:1])
    assert torch.equal(inter_stage[1]["codes.audio"], padded_codes[1:2])


def test_async_chunk_output_snapshots_mm_before_deferred_get_output(monkeypatch) -> None:
    class FakeStream:
        def wait_stream(self, _stream) -> None:
            pass

    class FakeEvent:
        def record(self, _stream) -> None:
            pass

        def synchronize(self) -> None:
            pass

    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)

    source_codes = torch.tensor([[7, 8]], dtype=torch.long)
    model_runner_output = SimpleNamespace(
        req_ids=["req-0"],
        sampled_token_ids=None,
        prompt_logprobs_dict={},
    )
    sampler_output = SimpleNamespace(
        sampled_token_ids=torch.tensor([[123]], dtype=torch.long),
        logprobs_tensors=None,
        num_nans=None,
        sampling_mask_tensors=None,
    )
    input_batch = SimpleNamespace(
        query_start_loc_np=np.array([0, 1], dtype=np.int32),
        num_scheduled_tokens=np.array([1], dtype=np.int32),
        num_reqs=1,
        num_tokens_after_padding=1,
    )

    output = OmniAsyncOutput(
        model_runner_output=model_runner_output,
        sampler_output=sampler_output,
        num_sampled_tokens=torch.tensor([1], dtype=torch.long),
        main_stream=FakeStream(),
        copy_stream=FakeStream(),
        copy_event=FakeEvent(),
        multimodal_outputs={"codes": {"audio": source_codes}},
        input_batch=input_batch,
        async_chunk=True,
    )

    # Simulate the next CUDA-graph replay overwriting the static output buffer
    # before vLLM drains this deferred output object.
    source_codes.fill_(99)

    finalized = output.get_output()

    assert finalized.inter_stage_outputs is not None
    assert torch.equal(
        finalized.inter_stage_outputs[0]["codes.audio"],
        torch.tensor([[7, 8]], dtype=torch.long),
    )


def test_async_chunk_output_stages_mm_on_output_copy_stream(monkeypatch) -> None:
    class FakeStream:
        def wait_stream(self, _stream) -> None:
            pass

    class FakeEvent:
        def record(self, _stream) -> None:
            pass

        def synchronize(self) -> None:
            pass

    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    copied = []

    def copy_mm(mm_outputs, total_tokens, **_copy_context):
        copied.append((mm_outputs, total_tokens))
        return {"codes": {"audio": mm_outputs["codes"]["audio"].clone()}}

    monkeypatch.setattr(omni_ar_model_runner, "_async_copy_mm", copy_mm)

    output = OmniAsyncOutput(
        model_runner_output=SimpleNamespace(
            req_ids=["req-0"],
            sampled_token_ids=None,
            prompt_logprobs_dict={},
        ),
        sampler_output=SimpleNamespace(
            sampled_token_ids=torch.tensor([[123]], dtype=torch.long),
            logprobs_tensors=None,
            num_nans=None,
            sampling_mask_tensors=None,
        ),
        num_sampled_tokens=torch.tensor([1], dtype=torch.long),
        main_stream=FakeStream(),
        copy_stream=FakeStream(),
        copy_event=FakeEvent(),
        multimodal_outputs={"codes": {"audio": torch.tensor([[7, 8]])}},
        input_batch=SimpleNamespace(
            query_start_loc_np=np.array([0, 1], dtype=np.int32),
            num_scheduled_tokens=np.array([1], dtype=np.int32),
            num_reqs=1,
            num_tokens_after_padding=1,
        ),
        async_chunk=True,
    )

    assert len(copied) == 1
    assert copied[0][1] == 1
    assert output._mm_snapshot["codes"]["audio"].device.type == "cpu"


def test_async_output_reuses_copy_context_for_all_d2h_helpers(monkeypatch) -> None:
    class FakeStream:
        def wait_stream(self, _stream) -> None:
            pass

    class FakeEvent:
        def record(self, _stream) -> None:
            pass

        def synchronize(self) -> None:
            pass

    copy_stream = FakeStream()
    copy_contexts = []

    def copy_to_np(value, *, copy_stream, pin_memory):
        copy_contexts.append((copy_stream, pin_memory))
        return value.numpy().copy()

    def copy_mm(mm_outputs, total_tokens, *, copy_stream, pin_memory):
        copy_contexts.append((copy_stream, pin_memory))
        return mm_outputs

    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    monkeypatch.setattr(omni_ar_model_runner, "PIN_MEMORY", True)
    monkeypatch.setattr(omni_ar_model_runner, "_async_copy_to_np", copy_to_np)
    monkeypatch.setattr(omni_ar_model_runner, "_async_copy_mm", copy_mm)

    OmniAsyncOutput(
        model_runner_output=SimpleNamespace(
            req_ids=["req-0"],
            sampled_token_ids=None,
            prompt_logprobs_dict={},
        ),
        sampler_output=SimpleNamespace(
            sampled_token_ids=torch.tensor([[123]], dtype=torch.long),
            logprobs_tensors=None,
            num_nans=None,
            sampling_mask_tensors=None,
        ),
        num_sampled_tokens=torch.tensor([1], dtype=torch.long),
        main_stream=FakeStream(),
        copy_stream=copy_stream,
        copy_event=FakeEvent(),
        multimodal_outputs={"codes": {"audio": torch.tensor([[7, 8]])}},
        input_batch=SimpleNamespace(
            query_start_loc_np=np.array([0, 1], dtype=np.int32),
            num_scheduled_tokens=np.array([1], dtype=np.int32),
            num_reqs=1,
            num_tokens_after_padding=1,
        ),
        async_chunk=True,
    )

    assert copy_contexts == [(copy_stream, True)] * 3


def test_async_output_does_not_probe_pin_memory_at_runtime(monkeypatch) -> None:
    class FakeStream:
        def wait_stream(self, _stream) -> None:
            pass

    class FakeEvent:
        def record(self, _stream) -> None:
            pass

    def fail_runtime_probe() -> bool:
        raise AssertionError("pin-memory support must be resolved outside the output hot path")

    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    monkeypatch.setattr(
        omni_ar_model_runner,
        "is_pin_memory_available",
        fail_runtime_probe,
        raising=False,
    )

    OmniAsyncOutput(
        model_runner_output=SimpleNamespace(
            req_ids=["req-0"],
            sampled_token_ids=None,
            prompt_logprobs_dict={},
        ),
        sampler_output=SimpleNamespace(
            sampled_token_ids=torch.tensor([[123]], dtype=torch.long),
            logprobs_tensors=None,
            num_nans=None,
            sampling_mask_tensors=None,
        ),
        num_sampled_tokens=torch.tensor([1], dtype=torch.long),
        main_stream=FakeStream(),
        copy_stream=FakeStream(),
        copy_event=FakeEvent(),
    )


def test_build_pooler_output_empty_mm():
    hidden = torch.randn(4, 8)

    pooler = OmniARModelRunner._build_pooler_output_from_cpu(
        hidden,
        {},
        query_start_loc_np=np.array([0]),
        num_scheduled_tokens=np.array([4], dtype=np.int32),
        num_reqs=1,
    )
    assert len(pooler) == 1
    assert "hidden" in pooler[0]
    assert len(pooler[0]) == 1


# ---------------------------------------------------------------
# _async_copy_mm (was _copy_mm_to_cpu)
# ---------------------------------------------------------------


def test_copy_mm_to_cpu_tensor():
    total = 10
    t = torch.randn(10, 4)
    result = _async_copy_mm({"feat": t}, total)
    assert "feat" in result
    assert result["feat"].shape == (10, 4)
    assert result["feat"].device == torch.device("cpu")


def test_copy_mm_to_cpu_dict():
    total = 10
    d = {"inner": torch.randn(10, 2)}
    result = _async_copy_mm({"nested": d}, total)
    assert "nested" in result
    assert "inner" in result["nested"]


def test_copy_mm_to_cpu_list():
    result = _async_copy_mm({"items": [torch.randn(3), "text"]}, 10)
    assert "items" in result
    assert isinstance(result["items"][0], torch.Tensor)
    assert result["items"][1] == "text"


def test_copy_mm_to_cpu_empty():
    assert _async_copy_mm({}, 10) == {}


def test_copy_mm_to_cpu_fails_entire_payload_when_one_leaf_copy_fails(monkeypatch):
    good = torch.randn(2)
    bad = torch.randn(2)
    original = omni_ar_model_runner._async_copy_mm_value

    def copy_or_fail(value, **kwargs):
        if value is bad:
            raise RuntimeError("injected D2H failure")
        return original(value, **kwargs)

    monkeypatch.setattr(omni_ar_model_runner, "_async_copy_mm_value", copy_or_fail)

    with pytest.raises(RuntimeError, match="injected D2H failure"):
        _async_copy_mm({"good": good, "bad": bad}, 10)


# ---------------------------------------------------------------
# Slicing via _build_pooler_output_from_cpu (was _slice_mm_payload)
# ---------------------------------------------------------------


def test_slice_mm_payload_tensor():
    hidden = torch.randn(6, 4)
    mm_cpu = {"feat": torch.randn(6, 2)}

    pooler = OmniARModelRunner._build_pooler_output_from_cpu(
        hidden,
        mm_cpu,
        query_start_loc_np=np.array([0, 3]),
        num_scheduled_tokens=np.array([3, 3], dtype=np.int32),
        num_reqs=2,
    )
    assert pooler[0]["feat"].shape == (3, 2)
    assert pooler[1]["feat"].shape == (3, 2)


def test_slice_mm_payload_list():
    hidden = torch.randn(6, 4)
    mm_cpu = {"items": [torch.randn(2), torch.randn(3)]}

    pooler = OmniARModelRunner._build_pooler_output_from_cpu(
        hidden,
        mm_cpu,
        query_start_loc_np=np.array([0, 3]),
        num_scheduled_tokens=np.array([3, 3], dtype=np.int32),
        num_reqs=2,
    )
    assert isinstance(pooler[0]["items"], torch.Tensor)
    assert isinstance(pooler[1]["items"], torch.Tensor)


def test_slice_mm_payload_dict():
    hidden = torch.randn(6, 4)
    mm_cpu = {"nested": {"a": torch.randn(6, 2)}}

    pooler = OmniARModelRunner._build_pooler_output_from_cpu(
        hidden,
        mm_cpu,
        query_start_loc_np=np.array([0, 3]),
        num_scheduled_tokens=np.array([3, 3], dtype=np.int32),
        num_reqs=2,
    )
    assert pooler[1]["nested"]["a"].shape == (3, 2)


def test_build_pooler_output_flattens_nested_payload_for_msgspec():
    hidden = torch.randn(4, 4)
    mm_cpu = {"codes": {"audio": torch.randn(4, 16)}}

    pooler = OmniARModelRunner._build_pooler_output_from_cpu(
        hidden,
        mm_cpu,
        query_start_loc_np=np.array([0, 2]),
        num_scheduled_tokens=np.array([2, 2], dtype=np.int32),
        num_reqs=2,
    )

    assert "codes" not in pooler[0]
    assert pooler[0]["codes.audio"].shape == (2, 16)
    assert pooler[1]["codes.audio"].shape == (2, 16)


def test_build_async_chunk_outputs_from_mm_omits_hidden_and_splits_channels():
    codes = torch.arange(4 * 4, dtype=torch.long).reshape(4, 4)
    audio = torch.randn(4, 8)
    mm = {
        "codes": {"audio": codes},
        "audio": audio,
    }

    inter_stage, client = OmniARModelRunner._build_async_chunk_outputs_from_mm(
        mm,
        query_start_loc_np=np.array([0, 2, 4]),
        num_scheduled_tokens=np.array([2, 2], dtype=np.int32),
        num_reqs=2,
        total_tokens=4,
    )

    assert inter_stage is not None
    assert client is not None
    assert "hidden" not in inter_stage[0]
    assert torch.equal(inter_stage[0]["codes.audio"], codes[:2])
    assert torch.equal(inter_stage[1]["codes.audio"], codes[2:])
    assert torch.equal(client[0]["audio"], audio[:2])
    assert torch.equal(client[1]["audio"], audio[2:])


def test_build_async_chunk_outputs_from_mm_keeps_per_request_code_list():
    req0 = torch.arange(16, dtype=torch.long).reshape(1, 16)
    req1 = torch.arange(16, 32, dtype=torch.long).reshape(1, 16)
    mm = {"codes": {"audio": [req0, req1]}}

    inter_stage, client = OmniARModelRunner._build_async_chunk_outputs_from_mm(
        mm,
        query_start_loc_np=np.array([0, 1, 2]),
        num_scheduled_tokens=np.array([1, 1], dtype=np.int32),
        num_reqs=2,
        total_tokens=2,
    )

    assert inter_stage is not None
    assert client is None
    assert torch.equal(inter_stage[0]["codes.audio"], req0)
    assert torch.equal(inter_stage[1]["codes.audio"], req1)


def test_build_pooler_output_preserves_qwen3_nested_payload():
    hidden = torch.randn(4, 4)
    mm = {
        "hidden_states": {
            "layers": {
                0: torch.randn(4, 4),
                24: torch.randn(4, 4),
            },
        },
        "embed": {
            "tts_bos": [torch.randn(1, 1, 4)],
            "tts_eos": [torch.randn(1, 1, 4)],
            "tts_pad": [torch.randn(1, 1, 4)],
        },
    }

    mm_cpu = _async_copy_mm(mm, total_tokens=4)
    pooler = OmniARModelRunner._build_pooler_output_from_cpu(
        hidden,
        mm_cpu,
        query_start_loc_np=np.array([0, 2]),
        num_scheduled_tokens=np.array([2, 2], dtype=np.int32),
        num_reqs=2,
    )

    payload = unflatten_payload(pooler[0])
    assert payload["hidden_states"]["layers"][0].shape == (2, 4)
    assert payload["hidden_states"]["layers"][24].shape == (2, 4)
    assert payload["embed"]["tts_bos"].shape == (1, 1, 4)


def test_kv_transfer_uses_global_request_id_from_intermediate_buffer():
    runner = object.__new__(OmniARModelRunner)
    runner.req_states = SimpleNamespace(req_id_to_index={"local": 1})
    runner.model_state = SimpleNamespace(
        intermediate_buffer=SimpleNamespace(
            buffers=[{}, {"global_request_id": "global"}],
        ),
    )
    runner.model = object()
    runner.kv_caches = [object()]
    runner.cache_config = SimpleNamespace(block_size=16, cache_dtype="auto")
    manager = MagicMock()
    manager.handle_finished_requests_kv_transfer.return_value = ["local"]
    runner._ensure_kv_transfer_manager = MagicMock(return_value=manager)

    runner._handle_kv_transfer_pre(
        SimpleNamespace(
            finished_requests_needing_kv_transfer={
                "local": {"block_ids": [1], "seq_len": 3},
            }
        )
    )

    resolver = manager.handle_finished_requests_kv_transfer.call_args.kwargs["request_id_resolver"]
    assert resolver("local") == "global"
