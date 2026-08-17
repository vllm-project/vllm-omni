# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

HANDLE_KEY = DiffusionModelRunner._STAGE_PAYLOAD_HANDLE_KEY


class _FakeConnector:
    def __init__(self, payload=None, *, raises=False):
        self._payload = payload
        self._raises = raises
        self.calls: list[tuple[str, str, str, object]] = []

    def get(self, from_stage, to_stage, get_key, metadata=None):
        self.calls.append((from_stage, to_stage, get_key, metadata))
        if self._raises:
            raise RuntimeError("transfer failed")
        if self._payload is None:
            return None
        return self._payload, 0


class _FakeKVTransferManager:
    def __init__(self, connector, recv_stages=("0", "1")):
        self.connector = connector
        self.recv_stages = recv_stages
        self.sender_info_calls: list[tuple[dict, str | None]] = []

    def update_sender_info(self, sender_info, sender_stage_id=None):
        self.sender_info_calls.append((sender_info, sender_stage_id))


def _make_runner(connector, *, payload_keys=("text_encoder_output",), recv_stages=("0", "1")):
    runner = object.__new__(DiffusionModelRunner)
    runner.od_config = SimpleNamespace(stage_input_payload_keys=payload_keys, stage_id=1)
    runner.device = torch.device("cpu")
    runner._local_rank = 0
    runner.pipeline = None
    runner.kv_transfer_manager = _FakeKVTransferManager(connector, recv_stages=recv_stages)
    return runner


def _make_request(prompt, *, request_id="req-7", kv_sender_info=None):
    return SimpleNamespace(prompt=prompt, request_id=request_id, kv_sender_info=kv_sender_info)


def _conditioning():
    return {"text_encoder_output": {"hidden_states": torch.zeros(4, 8), "token_tags": torch.zeros(4)}}


def test_key_convention_fetch_merges_into_additional_information():
    connector = _FakeConnector(_conditioning())
    runner = _make_runner(connector)
    req = _make_request({"prompt": "a cat"})

    runner._maybe_recv_stage_payload(req)

    assert connector.calls == [("0", "1", "req-7_0_0", None)]
    assert set(req.prompt["additional_information"]) == {"text_encoder_output"}
    assert req.prompt["additional_information"]["text_encoder_output"]["hidden_states"].shape == (4, 8)


def test_handle_path_uses_its_own_key_and_metadata():
    connector = _FakeConnector(_conditioning())
    runner = _make_runner(connector, payload_keys=())
    handle = {
        "key": "custom-key",
        "from_stage": "3",
        "to_stage": "4",
        "metadata": {"schema_version": 1},
    }
    req = _make_request({"prompt": "a cat", HANDLE_KEY: handle})

    runner._maybe_recv_stage_payload(req)

    assert connector.calls == [("3", "4", "custom-key", {"schema_version": 1})]
    # The handle is transport bookkeeping and must not reach the pipeline.
    assert HANDLE_KEY not in req.prompt
    assert "text_encoder_output" in req.prompt["additional_information"]


def test_stage_without_declared_keys_never_touches_the_connector():
    connector = _FakeConnector(_conditioning())
    runner = _make_runner(connector, payload_keys=())
    req = _make_request({"prompt": "a cat"})

    runner._maybe_recv_stage_payload(req)

    assert connector.calls == []
    assert "additional_information" not in req.prompt


def test_undeclared_payload_keys_are_dropped():
    connector = _FakeConnector({**_conditioning(), "debug_blob": torch.zeros(2)})
    runner = _make_runner(connector)
    req = _make_request({"prompt": "a cat"})

    runner._maybe_recv_stage_payload(req)

    assert set(req.prompt["additional_information"]) == {"text_encoder_output"}


@pytest.mark.parametrize(
    "connector",
    [
        _FakeConnector(None),
        _FakeConnector(raises=True),
        _FakeConnector("not-a-dict"),
    ],
    ids=["missing", "raises", "wrong_type"],
)
def test_failed_transfer_falls_back_to_the_inline_prompt(connector):
    runner = _make_runner(connector)
    inline = {"text_encoder_output": {"hidden_states": torch.ones(2, 8)}}
    req = _make_request({"prompt": "a cat", "additional_information": dict(inline)})

    runner._maybe_recv_stage_payload(req)

    assert torch.equal(
        req.prompt["additional_information"]["text_encoder_output"]["hidden_states"],
        inline["text_encoder_output"]["hidden_states"],
    )


def test_sender_info_is_applied_before_the_connector_is_used():
    connector = _FakeConnector(_conditioning())
    runner = _make_runner(connector)
    req = _make_request({"prompt": "a cat"}, kv_sender_info={0: {"host": "10.0.0.1", "zmq_port": 50171}})

    runner._maybe_recv_stage_payload(req)

    assert runner.kv_transfer_manager.sender_info_calls == [({0: {"host": "10.0.0.1", "zmq_port": 50171}}, "0")]


def test_tp_payload_is_fetched_once_by_leader_and_broadcast_to_followers():
    connector = _FakeConnector(_conditioning())
    state = {}

    class _FakeTPGroup:
        world_size = 4

        def __init__(self, rank):
            self.rank_in_group = rank

        def broadcast_object(self, value, src=0):
            if self.rank_in_group == src:
                state["delivered"] = value
            return state["delivered"]

        def broadcast_tensor_dict(self, value, src=0):
            if self.rank_in_group == src:
                state["payload"] = value
            return state["payload"]

    leader = _make_runner(connector)
    leader._local_rank = 0
    leader._get_local_tp_group = lambda: _FakeTPGroup(0)
    follower = _make_runner(connector)
    follower._local_rank = 1
    follower._get_local_tp_group = lambda: _FakeTPGroup(1)

    leader_req = _make_request({"prompt": "a cat"})
    follower_req = _make_request({"prompt": "a cat"})
    leader._maybe_recv_stage_payload(leader_req)
    follower._maybe_recv_stage_payload(follower_req)

    assert len(connector.calls) == 1
    leader_output = leader_req.prompt["additional_information"]["text_encoder_output"]["hidden_states"]
    follower_output = follower_req.prompt["additional_information"]["text_encoder_output"]["hidden_states"]
    assert torch.equal(leader_output, follower_output)


def test_tp_payload_miss_is_broadcast_without_follower_connector_access():
    connector = _FakeConnector(None)
    state = {}

    class _FakeTPGroup:
        world_size = 2

        def __init__(self, rank):
            self.rank_in_group = rank

        def broadcast_object(self, value, src=0):
            if self.rank_in_group == src:
                state["delivered"] = value
            return state["delivered"]

        def broadcast_tensor_dict(self, value, src=0):
            raise AssertionError("missing payload must not be broadcast")

    leader = _make_runner(connector)
    leader._get_local_tp_group = lambda: _FakeTPGroup(0)
    follower = _make_runner(connector)
    follower._local_rank = 1
    follower._get_local_tp_group = lambda: _FakeTPGroup(1)

    leader._maybe_recv_stage_payload(_make_request({"prompt": "a cat"}))
    follower._maybe_recv_stage_payload(_make_request({"prompt": "a cat"}))

    assert len(connector.calls) == 1


def test_missing_incoming_edge_is_reported_not_fetched():
    connector = _FakeConnector(_conditioning())
    runner = _make_runner(connector, recv_stages=(None, None))
    req = _make_request({"prompt": "a cat"})

    runner._maybe_recv_stage_payload(req)

    assert connector.calls == []
    assert "additional_information" not in req.prompt


def test_non_dict_prompt_is_left_alone():
    connector = _FakeConnector(_conditioning())
    runner = _make_runner(connector)
    req = _make_request("a cat")

    runner._maybe_recv_stage_payload(req)

    assert connector.calls == []
    assert req.prompt == "a cat"
