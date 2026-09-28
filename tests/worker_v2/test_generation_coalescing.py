# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import threading

import pytest

from vllm_omni.worker_v2.omni_data_plane import OmniRunnerDataPlane


def _plane():
    p = object.__new__(OmniRunnerDataPlane)
    p._stage_id = 1
    p._lock = threading.Lock()
    p._omni_connector_output_drain_lock = threading.Lock()
    p._generation_batch_wait_s = 0.05
    p._generation_batch_min_size = 3
    p._generation_batch_deadline = None
    p._finished_load_reqs = {"a"}
    p._chunk_ready_req_ids = set()
    p._chunk_finished_req_ids = set()
    p._get_req_chunk = {"a": 2, "b": 2, "c": 2}
    p._get_local_tp_group = lambda: None
    p._async_chunk = True
    p._local_request_metadata = {"a": {"next_stage_prompt_len": 180}}
    p._kv_sent_req_ids = set()
    p._stage_recv_req_ids = set()
    p._local_stage_payload_cache = {}
    p.has_pending_kv_work = lambda: False
    return p


def test_deadline_flushes_original_payload_without_further_arrivals(mocker):
    clock = mocker.patch("vllm_omni.worker_v2.omni_data_plane.time.monotonic", return_value=1.0)
    p = _plane()
    output = p.get_omni_connector_output()
    assert output.has_pending_kv_work and not output.chunk_ready_req_ids
    assert p._finished_load_reqs == {"a"}
    assert "a" in p._local_request_metadata
    clock.return_value = 1.051
    output = p.get_omni_connector_output()
    assert output.chunk_ready_req_ids == {"a"}
    assert output.request_metadata == {"a": {"next_stage_prompt_len": 180}}
    assert not p._local_request_metadata
    assert not p._finished_load_reqs
    assert not p._chunk_ready_req_ids
    assert not output.has_pending_kv_work


def test_urgent_request_bypasses_without_flushing_steady_or_extending_deadline(mocker):
    clock = mocker.patch("vllm_omni.worker_v2.omni_data_plane.time.monotonic", return_value=1.0)
    p = _plane()
    p.get_omni_connector_output()
    p._finished_load_reqs.add("b")
    p._get_req_chunk["b"] = 1
    p._local_request_metadata["b"] = {"next_stage_prompt_len": 12}
    clock.return_value = 1.04
    output = p.get_omni_connector_output()
    assert output.chunk_ready_req_ids == {"b"}
    assert set(output.request_metadata) == {"b"}
    assert output.has_pending_kv_work
    assert p._generation_batch_deadline == 1.05
    clock.return_value = 1.051
    assert p.get_omni_connector_output().chunk_ready_req_ids == {"a"}


def test_target_batch_flushes_before_deadline(mocker):
    mocker.patch("vllm_omni.worker_v2.omni_data_plane.time.monotonic", return_value=1.0)
    p = _plane()
    p.get_omni_connector_output()
    p._finished_load_reqs.update({"b", "c"})
    output = p.get_omni_connector_output()
    assert output.chunk_ready_req_ids == {"a", "b", "c"}
    assert p._generation_batch_deadline is None


def test_terminal_is_not_held_with_unrelated_steady_chunk():
    p = _plane()
    p._chunk_finished_req_ids.add("b")
    p._finished_load_reqs.add("b")
    p._local_request_metadata["b"] = {"input_terminal": True}
    output = p.get_omni_connector_output()
    assert output.chunk_finished_req_ids == {"b"}
    assert output.chunk_ready_req_ids == {"b"}
    assert output.request_metadata == {"b": {"input_terminal": True}}
    assert p._finished_load_reqs == {"a"}
    assert not p._chunk_finished_req_ids


def test_cancelled_group_clears_deadline(mocker):
    mocker.patch("vllm_omni.worker_v2.omni_data_plane.time.monotonic", return_value=1.0)
    p = _plane()
    p.get_omni_connector_output()
    p._finished_load_reqs.clear()
    p._local_request_metadata.clear()
    output = p.get_omni_connector_output()
    assert not output.chunk_ready_req_ids
    assert p._generation_batch_deadline is None


def test_disabled_policy_drains_immediately():
    p = _plane()
    p._generation_batch_wait_s = 0
    output = p.get_omni_connector_output()
    assert output.chunk_ready_req_ids == {"a"}
    assert set(output.request_metadata) == {"a"}


def test_flattened_real_connector_config_enables_policy(mocker):
    from types import SimpleNamespace

    from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector

    def init_connectors(p, model_config):
        p._omni_connector = SharedMemoryConnector({"generation_batch_wait_ms": 50, "generation_batch_min_size": 16})
        p._model_mode = "generation"
        p._async_chunk = True
        p._custom_process_func = None

    mocker.patch.object(OmniRunnerDataPlane, "init_omni_connectors", init_connectors)
    mocker.patch.object(OmniRunnerDataPlane, "_start_output_worker")
    p = OmniRunnerDataPlane(SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=1)), SimpleNamespace())
    assert p._generation_batch_wait_s == 0.05
    assert p._generation_batch_min_size == 16


def test_optional_terminal_audio_retains_finish_signal_until_payload_release(mocker):
    clock = mocker.patch("vllm_omni.worker_v2.omni_data_plane.time.monotonic", return_value=1.0)
    p = _plane()
    p._generation_batch_hold_terminal_audio = True
    p._chunk_finished_req_ids.add("a")
    output = p.get_omni_connector_output()
    assert not output.chunk_finished_req_ids
    assert not output.chunk_ready_req_ids
    assert p._chunk_finished_req_ids == {"a"}
    assert "a" in p._local_request_metadata
    clock.return_value = 1.051
    output = p.get_omni_connector_output()
    assert output.chunk_ready_req_ids == output.chunk_finished_req_ids == {"a"}
    assert output.request_metadata["a"]["next_stage_prompt_len"] == 180
    assert not p._chunk_finished_req_ids


def test_finish_only_bypasses_optional_terminal_audio_coalescing():
    p = _plane()
    p._generation_batch_hold_terminal_audio = True
    p._chunk_finished_req_ids.add("a")
    p._local_request_metadata["a"]["next_stage_prompt_len"] = 0
    output = p.get_omni_connector_output()
    assert output.chunk_ready_req_ids == output.chunk_finished_req_ids == {"a"}


def test_optional_first_audio_deadline_and_terminal_sentinel(mocker):
    clock = mocker.patch("vllm_omni.worker_v2.omni_data_plane.time.monotonic", return_value=1.0)
    p = _plane()
    p._generation_batch_hold_first_audio = True
    p._get_req_chunk["a"] = 1
    output = p.get_omni_connector_output()
    assert not output.chunk_ready_req_ids
    assert output.has_pending_kv_work
    p._finished_load_reqs.add("b")
    p._get_req_chunk["b"] = 1
    p._chunk_finished_req_ids.add("b")
    p._local_request_metadata["b"] = {"next_stage_prompt_len": 0}
    clock.return_value = 1.04
    output = p.get_omni_connector_output()
    assert output.chunk_ready_req_ids == output.chunk_finished_req_ids == {"b"}
    assert p._generation_batch_deadline == 1.05
    clock.return_value = 1.051
    assert p.get_omni_connector_output().chunk_ready_req_ids == {"a"}


def test_optional_first_audio_joins_target_batch(mocker):
    mocker.patch("vllm_omni.worker_v2.omni_data_plane.time.monotonic", return_value=1.0)
    p = _plane()
    p._generation_batch_hold_first_audio = True
    p._get_req_chunk["a"] = 1
    p.get_omni_connector_output()
    p._finished_load_reqs.update({"b", "c"})
    output = p.get_omni_connector_output()
    assert output.chunk_ready_req_ids == {"a", "b", "c"}
    assert p._generation_batch_deadline is None


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
