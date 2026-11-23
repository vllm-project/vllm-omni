# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full-payload completion, failure, preemption replay, and prefix provenance.

The shared accumulator is the lifecycle owner. MRv2's data plane is the AR
caller that supplies token ranges and the prefix-cache keys actually rewritten
on a step. One-shot generation omits that range and keeps append/replace.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# isort: off
import vllm_omni  # noqa: F401 - install Request patches before importing vLLM
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler
from vllm.v1.request import RequestStatus
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
from vllm_omni.model_executor.stage_input_processors.qwen3_omni import thinker2talker_full_payload
from vllm_omni.model_executor.stage_input_processors.qwen3_tts import talker2code2wav_full_payload
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin
from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner
from vllm_omni.worker_v2.omni_data_plane import OmniRunnerDataPlane
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner
from vllm_omni.worker_v2.prefix_cache import PrefixCacheRewrite

# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _host(replace_keys: frozenset[str] = frozenset()) -> OmniRunnerDataPlane:
    host = OmniRunnerDataPlane.__new__(OmniRunnerDataPlane)
    host._native_requests = {}
    host._native_output_lock = threading.RLock()
    host._pending_full_payload_send = {}
    host._full_payload_token_ends = {}
    host._full_payload_replace_keys_cached = replace_keys
    host._stage_id = 0
    host.send_full_payload_outputs = MagicMock()
    host._request_ids_mapping = {}
    host._lock = threading.Lock()
    host._pending_save_counts = {}
    host._deferred_send_cleanup = set()
    host._put_req_chunk = {}
    host._send_side_request_payload = {}
    host._code_prompt_token_ids = {}
    host._cached_ic = {}
    host._ramp_chunk_count = {}
    host._adaptive_states = {}
    host._kv_pending_transfers = {}
    host._kv_active_transfers = set()
    host._kv_completed_transfers = set()
    host._kv_triggered_requests = set()
    host._get_req_chunk = {}
    host._pending_load_reqs = {}
    host._finished_load_reqs = set()
    host._chunk_ready_req_ids = set()
    host._chunk_finished_req_ids = set()
    host._chunk_stream_completed = set()
    host._stage_recv_req_ids = set()
    host._full_payload_pending_broadcast_req_ids = set()
    host._async_chunk_updated_req_ids = set()
    host._local_stage_payload_cache = {}
    host._local_request_metadata = {}
    return host


def _payload(host: OmniConnectorModelRunnerMixin, req_id: str) -> dict:
    entry = host._pending_full_payload_send[req_id]
    output, _request = host._materialize_full_payload_entry(entry)
    return output


def _accumulate(host, req_id, payload, token_range=None, prefix_cache_keys=frozenset()):
    host.accumulate_full_payload_output(
        req_id,
        payload,
        None,
        token_range=token_range,
        prefix_cache_keys=prefix_cache_keys,
    )


def test_one_shot_generation_keeps_append_policy():
    host = _host()
    _accumulate(host, "g", {"audio": torch.ones(4, 2)})
    _accumulate(host, "g", {"audio": torch.full((4, 2), 2)})
    assert torch.equal(_payload(host, "g")["audio"], torch.cat([torch.ones(4, 2), torch.full((4, 2), 2)]))


def test_replay_keeps_existing_rows_and_appends_only_the_new_suffix():
    host = _host()
    original = torch.arange(10).reshape(5, 2)
    _accumulate(host, "preempt", {"hidden": original}, token_range=(0, 5))
    _accumulate(host, "other", {"hidden": torch.full((3, 2), 99)}, token_range=(0, 3))
    for start, end in ((0, 2), (2, 4), (4, 7)):
        replay = torch.full((end - start, 2), -1)
        if end == 7:
            replay[-2:] = torch.tensor([[10, 11], [12, 13]])
        _accumulate(host, "preempt", {"hidden": replay}, token_range=(start, end))
    assert torch.equal(_payload(host, "preempt")["hidden"], torch.arange(14).reshape(7, 2))
    assert torch.equal(_payload(host, "other")["hidden"], torch.full((3, 2), 99))

    host.flush_full_payload_outputs({"preempt"})
    assert "preempt" not in host._full_payload_token_ends
    host.cleanup_finished_request("preempt")
    _accumulate(host, "preempt", {"hidden": torch.ones(1, 2)}, token_range=(0, 1))
    assert _payload(host, "preempt")["hidden"].shape == (1, 2)


def test_qwen3_omni_replay_keeps_request_scoped_tts_embeddings():
    host = _host()
    del host._full_payload_replace_keys_cached
    host._custom_process_func = thinker2talker_full_payload
    original_hidden = torch.arange(10).reshape(5, 2)
    replay_hidden = torch.cat([torch.full_like(original_hidden, -1), torch.tensor([[10, 11]])])
    fields = ("tts_bos", "tts_eos", "tts_pad")
    for step, hidden in enumerate((original_hidden, replay_hidden)):
        rows = len(hidden)
        embeddings = {key: [torch.full((1, 1, 2), 3 * step + index)] for index, key in enumerate(fields)}
        [payload] = OmniARModelRunner._build_pooler_output_from_cpu(
            hidden_cpu=hidden,
            mm_cpu={"embed": embeddings},
            query_start_loc_np=np.array([0, rows]),
            num_scheduled_tokens=np.array([rows]),
            num_reqs=1,
        )
        _accumulate(host, "thinker", payload, token_range=(0, rows))

    result = _payload(host, "thinker")
    assert torch.equal(result["hidden"], torch.arange(12).reshape(6, 2))
    for key in fields:
        assert torch.equal(result[f"embed.{key}"], payload[f"embed.{key}"])
        assert result[f"embed.{key}"].shape == (1, 1, 2)


def test_replay_preserves_snapshot_metadata_and_recovered_prefix():
    host = _host(frozenset({"snapshot"}))
    _accumulate(
        host,
        "r",
        {"hidden": torch.ones(4, 2), "snapshot": torch.ones(2, 3), "meta": 4},
        token_range=(0, 4),
    )
    _accumulate(
        host,
        "r",
        {"hidden": torch.zeros(2, 2), "snapshot": torch.zeros(1, 3), "meta": 2},
        token_range=(0, 2),
    )
    _accumulate(
        host,
        "r",
        {"hidden": torch.full((6, 2), 2), "snapshot": torch.full((3, 3), 2), "meta": 6},
        token_range=(4, 6),
        prefix_cache_keys=frozenset({"hidden"}),
    )
    payload = _payload(host, "r")
    assert torch.equal(payload["hidden"], torch.cat([torch.ones(4, 2), torch.full((2, 2), 2)]))
    assert torch.equal(payload["snapshot"], torch.full((3, 3), 2))
    assert payload["meta"] == 6


def test_ambiguous_partial_replay_fails_without_advancing_positions():
    host = _host()
    _accumulate(host, "r", {"audio": torch.ones(20, 2)}, token_range=(0, 4))
    _accumulate(host, "r", {"audio": torch.zeros(10, 2)}, token_range=(0, 2))
    with pytest.raises(ValueError, match="do not align with token range"):
        _accumulate(host, "r", {"audio": torch.zeros(30, 2)}, token_range=(2, 6))
    assert host._full_payload_token_ends["r"] == {"audio": 4}
    assert torch.equal(_payload(host, "r")["audio"], torch.ones(20, 2))


@pytest.mark.parametrize("previous_end", [2, 3], ids=["new-delta", "partial-replay"])
def test_scaled_delta_is_not_inferred_as_cached_prefix(previous_end):
    host = _host()
    original = torch.ones(previous_end * 2, 2)
    _accumulate(host, "r", {"audio": original}, token_range=(0, previous_end))
    delta = torch.full((4, 2), 2)
    if previous_end == 3:
        with pytest.raises(ValueError, match="do not align with token range"):
            _accumulate(host, "r", {"audio": delta}, token_range=(2, 4))
        expected = original
        expected_end = 3
    else:
        _accumulate(host, "r", {"audio": delta}, token_range=(2, 4))
        expected = torch.cat([original, delta])
        expected_end = 4
    assert torch.equal(_payload(host, "r")["audio"], expected)
    assert host._full_payload_token_ends["r"]["audio"] == expected_end


def test_codec_validity_mask_is_reconciled_with_its_rows():
    host = _host()
    _accumulate(
        host,
        "r",
        {
            "codes.audio": torch.ones(4, 2, dtype=torch.long),
            "meta.codec_frame_valid": torch.ones(4, dtype=torch.long),
        },
        token_range=(0, 4),
    )
    _accumulate(
        host,
        "r",
        {"codes.audio": torch.arange(6).reshape(3, 2), "meta.codec_frame_valid": torch.tensor([0, 0, 1])},
        token_range=(2, 5),
    )
    payload = _payload(host, "r")
    assert torch.equal(payload["codes.audio"], torch.cat([torch.ones(4, 2, dtype=torch.long), torch.tensor([[4, 5]])]))
    assert torch.equal(payload["meta.codec_frame_valid"], torch.tensor([1, 1, 1, 1, 1]))


def test_data_plane_abort_discards_pending_payload_without_affecting_completion():
    host = _host()
    _accumulate(host, "cancel", {"codes": torch.tensor([[1], [2]])})
    _accumulate(host, "done", {"codes": torch.tensor([[3], [4]])})
    _accumulate(host, "active", {"codes": torch.tensor([[5]])})
    host.abort_requests({"cancel"})
    host.flush_full_payload_outputs({"done"})
    sent = host.send_full_payload_outputs.call_args.kwargs["outputs"]
    assert set(sent) == {"done"}
    assert torch.equal(sent["done"][0]["codes"], torch.tensor([[3], [4]]))
    assert set(host._pending_full_payload_send) == {"active"}
    assert "cancel" not in host._full_payload_token_ends


def test_cleanup_fallback_flushes_normal_completion():
    host = _host()
    _accumulate(host, "cancel", {"codes": torch.tensor([[1]])}, token_range=(0, 1))
    host.cleanup_finished_request("cancel")
    assert not host._full_payload_token_ends
    assert not host._pending_full_payload_send
    host.send_full_payload_outputs.assert_called_once()


def test_request_id_reuse_starts_empty_after_discard():
    host = _host()
    _accumulate(host, "failed", {"hidden": torch.ones(3, 2)}, token_range=(0, 3))
    host.abort_requests({"failed"})
    _accumulate(host, "failed", {"hidden": torch.zeros(1, 2)}, token_range=(0, 1))
    assert torch.equal(_payload(host, "failed")["hidden"], torch.zeros(1, 2))


class _FinishedRequest:
    def __init__(self, request_id: str, status: RequestStatus) -> None:
        self.request_id = request_id
        self.status = status
        self.client_index = 0

    def is_finished(self) -> bool:
        return True


def _ar_scheduler() -> OmniARScheduler:
    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched._omits_kv_transfer_cache = {}
    sched._connector_finished = lambda request: (False, None)
    sched.encoder_cache_manager = MagicMock()
    sched.finished_req_ids = set()
    sched._new_prompt_len_snapshot = {}
    sched.finished_req_ids_dict = None
    sched._should_transfer_kv_for_request = lambda req_id: False
    sched._free_blocks = MagicMock()
    sched._free_input_coordinator_request = MagicMock()
    sched.chunk_transfer_adapter = None
    sched.requests = {}
    sched._discarded_req_ids = set()
    sched._pending_data_plane_terminal_req_ids = set()
    sched._pending_request_prewarms = {}
    sched._finished_req_output_token_ids = {}
    sched.input_coordinator = None
    return sched


@pytest.mark.parametrize(
    "status",
    [RequestStatus.FINISHED_ABORTED, RequestStatus.FINISHED_ERROR, RequestStatus.FINISHED_IGNORED],
)
def test_free_request_publishes_discarded_ids_once(status):
    sched = _ar_scheduler()
    sched._free_request(_FinishedRequest("failed", status))
    sched._free_request(_FinishedRequest("done", RequestStatus.FINISHED_STOPPED))
    assert sched._discarded_req_ids == {"failed"}
    base = SchedulerOutput.make_empty()
    base.finished_req_ids = set(sched.finished_req_ids)
    output = sched._wrap_omni_scheduler_output(base)
    assert output.discarded_req_ids == {"failed"}
    assert set(output.finished_req_ids) == {"failed", "done"}
    again = SchedulerOutput.make_empty()
    again.finished_req_ids = set(sched.finished_req_ids)
    assert sched._wrap_omni_scheduler_output(again).discarded_req_ids == set()


def test_generation_free_request_records_discard_before_upstream_free():
    sched = OmniGenerationScheduler.__new__(OmniGenerationScheduler)
    sched.input_coordinator = None
    request = _FinishedRequest("failed", RequestStatus.FINISHED_ERROR)
    with patch.object(VLLMScheduler, "_free_request", return_value=(None, None)) as upstream:
        sched._free_request(request)
    upstream.assert_called_once_with(request, False)
    assert sched._discarded_req_ids == {"failed"}


def test_data_plane_replay_preserves_token_history_and_codec_payload():
    plane = OmniRunnerDataPlane.__new__(OmniRunnerDataPlane)
    plane._native_requests = {}
    plane._native_output_lock = threading.RLock()
    plane._native_send_lock = threading.Lock()
    plane._native_output_error = None
    plane._native_output_error_lock = threading.Lock()
    plane._native_outputs_in_flight = {}
    plane._native_terminal_pending = set()
    plane._async_chunk = False
    plane._can_send = True
    plane._pending_full_payload_send = {}
    plane._full_payload_replace_keys_cached = frozenset()
    plane._full_payload_token_ends = {}
    plane.recv_full_payload_inputs = MagicMock()
    plane.send_chunks = MagicMock(return_value=1)
    plane.cleanup_finished_request = MagicMock()
    plane._deferred_shm_releases = set()
    runner = object.__new__(OmniGPUModelRunner)
    runner.model_config = SimpleNamespace(async_chunk=False)
    runner._omni_data_plane = plane
    initial = SchedulerOutput.make_empty()
    initial.scheduled_new_reqs = [SimpleNamespace(req_id="preempt", prompt_token_ids=[1])]
    runner._prepare_native_data_plane(initial)
    original = torch.tensor([[10, 11], [20, 21]])
    appended = torch.tensor([[30, 31]])
    plane.complete_outputs(
        req_ids=["preempt"],
        inter_stage_outputs=[{"codes.audio": original}],
        sampled_token_ids=[[4, 5]],
        full_payload_token_ranges={"preempt": (0, 2)},
    )
    resumed = SchedulerOutput.make_empty()
    resumed.scheduled_new_reqs = [SimpleNamespace(req_id="preempt", prompt_token_ids=[1], prefill_token_ids=[1, 4, 5])]
    runner._prepare_native_data_plane(resumed)
    plane.complete_outputs(
        req_ids=["preempt"],
        inter_stage_outputs=[{"codes.audio": torch.cat([torch.full_like(original, 99), appended])}],
        sampled_token_ids=[[6, 7]],
        full_payload_token_ranges={"preempt": (1, 3)},
        full_payload_prefix_cache_keys={"preempt": frozenset({"codes.audio"})},
    )
    plane.send_chunks.assert_not_called()
    plane.request_terminal({"preempt"})
    plane.send_chunks.assert_called_once()
    [(request, payload)] = plane.send_chunks.call_args.args[0]
    assert request.is_finished()
    assert request.prompt_token_ids == [1]
    assert request.output_token_ids == [4, 5, 6, 7]
    codec = talker2code2wav_full_payload(None, payload, request)
    assert torch.equal(codec["codes"]["audio"], torch.tensor([10, 20, 30, 11, 21, 31]))


def test_prefix_cache_rewrite_records_only_replaced_keys():
    hidden = torch.full((6, 2), 2)
    audio = torch.full((6, 1), 2)

    class _Cache:
        def materialize(self, step_id, req_ids, *, include_uncached=True):
            assert step_id == 7
            assert req_ids == ["r"]
            assert include_uncached is False
            return SimpleNamespace(hidden_states={"r": hidden}, mm_outputs={"codes.audio": {"r": audio}})

        def discard_step(self, step_id):
            raise AssertionError(step_id)

    output = OmniModelRunnerOutput(
        req_ids=["r"],
        sampled_token_ids=[[1]],
        prompt_logprobs_dict={},
        inter_stage_outputs=[{"hidden": torch.ones(2, 2), "codes.audio": torch.ones(2, 1), "meta": torch.tensor(1)}],
        pooler_output=None,
        multimodal_outputs=None,
    )
    PrefixCacheRewrite(_Cache(), 7, ((0, "r"),)).apply(output)
    assert output.full_payload_prefix_cache_keys == {"r": frozenset({"hidden", "codes.audio"})}
    assert torch.equal(output.inter_stage_outputs[0]["hidden"], hidden)
    assert torch.equal(output.inter_stage_outputs[0]["meta"], torch.tensor(1))


def test_discarded_terminal_is_aborted_instead_of_flushed():
    runner = object.__new__(OmniGPUModelRunner)
    runner.model_config = SimpleNamespace(async_chunk=False)
    plane = SimpleNamespace(
        register_request=MagicMock(),
        register_receivers=MagicMock(),
        recv_full_payload_inputs=MagicMock(),
        request_terminal=MagicMock(),
        abort_requests=MagicMock(),
    )
    runner._omni_data_plane = plane
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        pending_input_registrations=[],
        data_plane_terminal_req_ids={"done", "failed"},
        finished_req_ids={"done", "failed"},
        discarded_req_ids={"failed"},
    )
    runner._prepare_native_data_plane(scheduler_output)
    plane.request_terminal.assert_called_once_with({"done"})
    plane.abort_requests.assert_called_once_with({"failed"})
