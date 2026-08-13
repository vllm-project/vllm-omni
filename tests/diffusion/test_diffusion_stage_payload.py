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
