# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

HANDLE_KEY = DiffusionModelRunner._STAGE_PAYLOAD_HANDLE_KEY


class _FakeConnector:
    def abandon_get(self, get_key):
        pass

    def get_with_deadline(self, from_stage, to_stage, get_key, metadata=None, *, deadline):
        return self.get(from_stage, to_stage, get_key, metadata)

    def __init__(self, payload=None, *, raises=False, put_result=(True, 128, {"schema_version": 1})):
        self._payload = payload
        self._raises = raises
        self._put_result = put_result
        self.calls: list[tuple[str, str, str, object]] = []
        self.put_calls: list[tuple[str, str, str, object]] = []

    def get(self, from_stage, to_stage, get_key, metadata=None):
        self.calls.append((from_stage, to_stage, get_key, metadata))
        if self._raises:
            raise RuntimeError("transfer failed")
        if self._payload is None:
            return None
        return self._payload, 0

    def put(self, from_stage, to_stage, put_key, data):
        self.put_calls.append((from_stage, to_stage, put_key, data))
        if self._raises:
            raise RuntimeError("transfer failed")
        return self._put_result


class _FakeKVTransferManager:
    from vllm_omni.distributed.omni_connectors.kv_transfer_manager import OmniKVTransferManager

    _resolve_sender_info = OmniKVTransferManager._resolve_sender_info

    def __init__(self, connector, recv_stages=("0", "1"), send_stages=("0", "1")):
        self.connector = connector
        self.recv_stages = recv_stages
        self.send_stages = send_stages
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
    runner.init_omni_connectors(runner.od_config, runner.kv_transfer_manager, synchronous=True)
    return runner


def _make_sender(connector, *, payload_keys=("prompt_embeds",), send_stages=("0", "1")):
    runner = object.__new__(DiffusionModelRunner)
    runner.od_config = SimpleNamespace(stage_output_payload_keys=payload_keys, stage_id=0)
    runner.device = torch.device("cpu")
    runner.pipeline = None
    runner.kv_transfer_manager = _FakeKVTransferManager(connector, send_stages=send_stages)
    runner.init_omni_connectors(runner.od_config, runner.kv_transfer_manager, synchronous=True)
    return runner


def _make_output(**custom):
    return SimpleNamespace(custom_output=dict(custom))


def _make_request(prompt, *, request_id="req-7", kv_sender_info=None, payload_sender_info=None):
    return SimpleNamespace(
        prompt=prompt,
        request_id=request_id,
        kv_sender_info=kv_sender_info,
        payload_sender_info=payload_sender_info,
    )


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


def test_receive_retries_until_producer_publishes(monkeypatch):
    connector = _FakeConnector(_conditioning())
    get = connector.get
    attempts = []

    def delayed_get(*args, **kwargs):
        attempts.append(args)
        return get(*args, **kwargs) if len(attempts) == 3 else None

    monkeypatch.setattr(connector, "get", delayed_get)
    runner = _make_runner(connector)
    request = _make_request({"prompt": "a cat"})

    runner._maybe_recv_stage_payload(request)

    assert len(attempts) == 3
    assert "text_encoder_output" in request.prompt["additional_information"]


def test_receive_retry_budget_preserves_inline_payload(monkeypatch):
    from vllm_omni.distributed.omni_connectors.model_runner import omni_connector_payload_transport

    clock = [0.0]

    def advance(duration):
        clock[0] += duration

    monkeypatch.setattr(
        omni_connector_payload_transport, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=advance)
    )
    connector = _FakeConnector(None)
    runner = _make_runner(connector)
    inline = _conditioning()
    request = _make_request({"additional_information": inline})

    runner._maybe_recv_stage_payload(request)

    assert 1 < len(connector.calls) <= 41
    assert clock[0] == pytest.approx(2.0)
    assert request.prompt["additional_information"] is inline


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


def test_legacy_sender_info_is_passed_as_request_metadata():
    connector = _FakeConnector(_conditioning())
    runner = _make_runner(connector)
    req = _make_request({"prompt": "a cat"}, kv_sender_info={0: {"host": "10.0.0.1", "zmq_port": 50171}})

    runner._maybe_recv_stage_payload(req)

    assert runner.kv_transfer_manager.sender_info_calls == []
    assert connector.calls[0][3] == {"source_host": "10.0.0.1", "source_port": 50171}


def test_payload_sender_info_overrides_kv_sender_info_for_full_payloads():
    connector = _FakeConnector(_conditioning())
    runner = _make_runner(connector)
    req = _make_request(
        {"prompt": "a cat"},
        kv_sender_info={0: {"host": "10.0.0.1", "zmq_port": 50171}},
        payload_sender_info={"host": "10.0.0.1", "zmq_port": 50071},
    )

    runner._maybe_recv_stage_payload(req)

    assert runner.kv_transfer_manager.sender_info_calls == []
    assert connector.calls[0][3] == {"source_host": "10.0.0.1", "source_port": 50071}


def test_synchronous_receive_keeps_endpoints_request_scoped_and_uses_external_ids():
    connector = _FakeConnector(_conditioning())
    runner = _make_runner(connector, recv_stages=("2", "5"))
    for index in range(2):
        request = _make_request(
            {},
            request_id=f"internal-{index}",
            payload_sender_info={"host": f"10.0.0.{index + 1}", "zmq_port": 50071 + index},
        )
        request.external_req_id = f"external-{index}"
        runner._maybe_recv_stage_payload(request)
        assert "text_encoder_output" in request.prompt["additional_information"]
    assert connector.calls == [
        ("2", "5", "external-0_2_0", {"source_host": "10.0.0.1", "source_port": 50071}),
        ("2", "5", "external-1_2_0", {"source_host": "10.0.0.2", "source_port": 50072}),
    ]
    assert runner.kv_transfer_manager.sender_info_calls == []
    assert runner._pending_load_reqs == {}
    assert runner._get_req_chunk == {}


@pytest.mark.parametrize("stage_key", [2, "2"])
@pytest.mark.parametrize("use_handle", [False, True])
def test_sender_map_selects_actual_source_stage(stage_key, use_handle):
    connector = _FakeConnector(_conditioning())
    runner = _make_runner(connector, recv_stages=("0" if use_handle else "2", "5"))
    prompt = {}
    if use_handle:
        prompt[HANDLE_KEY] = {"key": "explicit", "from_stage": "2", "to_stage": "5"}
    request = _make_request(
        prompt,
        payload_sender_info={
            "0": {"host": "wrong", "zmq_port": 50000},
            stage_key: {"host": "right", "zmq_port": 50002},
        },
    )
    runner._maybe_recv_stage_payload(request)
    assert connector.calls[0][0:2] == ("2", "5")
    assert connector.calls[0][3] == {"source_host": "right", "source_port": 50002}


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

    with pytest.raises(RuntimeError, match="Stage payload unavailable"):
        leader._maybe_recv_stage_payload(_make_request({"prompt": "a cat"}))
    with pytest.raises(RuntimeError, match="Stage payload unavailable"):
        follower._maybe_recv_stage_payload(_make_request({"prompt": "a cat"}))

    assert 1 < len(connector.calls) <= 41


def test_sp_payload_is_fetched_once_by_leader_and_broadcast_to_followers():
    connector = _FakeConnector(_conditioning())
    state = {}

    class _FakeSPGroup:
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
    leader._get_local_tp_group = lambda: None
    leader._stage_payload_broadcast_groups = lambda: (_FakeSPGroup(0),)
    follower = _make_runner(connector)
    follower._local_rank = 1
    follower._get_local_tp_group = lambda: None
    follower._stage_payload_broadcast_groups = lambda: (_FakeSPGroup(1),)

    leader_req = _make_request({"prompt": "a cat"})
    follower_req = _make_request({"prompt": "a cat"})
    leader._maybe_recv_stage_payload(leader_req)
    follower._maybe_recv_stage_payload(follower_req)

    assert len(connector.calls) == 1
    leader_output = leader_req.prompt["additional_information"]["text_encoder_output"]["hidden_states"]
    follower_output = follower_req.prompt["additional_information"]["text_encoder_output"]["hidden_states"]
    assert torch.equal(leader_output, follower_output)


def test_missing_incoming_edge_is_reported_not_fetched():
    connector = _FakeConnector(_conditioning())
    runner = _make_runner(connector, recv_stages=(None, None))
    req = _make_request({"prompt": "a cat"})

    with pytest.raises(RuntimeError, match="no incoming edge"):
        runner._maybe_recv_stage_payload(req)

    assert connector.calls == []
    assert "additional_information" not in req.prompt


@pytest.mark.parametrize("delivered", [True, False])
def test_payload_reaches_full_tp_sp_grid_once(monkeypatch, delivered):
    from vllm_omni.diffusion.distributed import parallel_state

    connector = _FakeConnector(_conditioning() if delivered else None)
    packets = {}

    class GridGroup:
        world_size = 2

        def __init__(self, dimension, peer_rank, rank):
            self.key = (dimension, peer_rank)
            self.rank_in_group = rank

        def broadcast_object(self, value, src=0):
            if self.rank_in_group == src:
                packets[self.key, "delivered"] = value
            return packets[self.key, "delivered"]

        def broadcast_tensor_dict(self, value, src=0):
            if self.rank_in_group == src:
                packets[self.key, "payload"] = value
            return packets[self.key, "payload"]

    for sp_rank in range(2):
        for tp_rank in range(2):
            runner = _make_runner(connector)
            tp_group = GridGroup("tp", sp_rank, tp_rank)
            sp_group = GridGroup("sp", tp_rank, sp_rank)
            monkeypatch.setattr(runner, "_get_local_tp_group", lambda: tp_group)
            monkeypatch.setattr(parallel_state, "get_sp_group", lambda: sp_group)
            request = _make_request({"prompt": "a cat"})

            if delivered:
                runner._maybe_recv_stage_payload(request)
                output = request.prompt["additional_information"]["text_encoder_output"]
                assert torch.equal(output["hidden_states"], connector._payload["text_encoder_output"]["hidden_states"])
            else:
                with pytest.raises(RuntimeError, match="Stage payload unavailable"):
                    runner._maybe_recv_stage_payload(request)
    if delivered:
        assert len(connector.calls) == 1
    else:
        assert 1 < len(connector.calls) <= 41


def test_native_kv_runner_keeps_synchronous_payload_transport(monkeypatch):
    from unittest.mock import Mock

    from vllm_omni.diffusion.worker import diffusion_model_runner as runner_module

    connector = _FakeConnector(_conditioning())
    manager = _FakeKVTransferManager(connector)
    config = SimpleNamespace(stage_input_payload_keys=("text_encoder_output",), stage_id=1, kv_transfer_config=object())
    monkeypatch.setattr(runner_module.OmniKVTransferManager, "from_od_config", lambda config: manager)
    monkeypatch.setattr(runner_module, "DiffusionKVModelRunnerBackend", Mock())
    runner = DiffusionModelRunner(SimpleNamespace(), config, torch.device("cpu"))
    runner._initialize_generator = Mock()
    request = _make_request({"prompt": "a cat"})
    request.sampling_params = SimpleNamespace()

    runner._prepare_request_for_forward(request, od_config=config)

    assert runner.kv_transfer_manager is None
    assert runner._kv_transfer_manager is manager
    assert runner._kv_connector is None
    assert not runner._kv_prefetch_enabled
    assert len(connector.calls) == 1
    assert "text_encoder_output" in request.prompt["additional_information"]
    runner._initialize_generator.assert_called_once_with(request.sampling_params)


def test_step_mode_receives_payload_before_kv_and_reuses_cached_state():
    connector = _FakeConnector(_conditioning())
    runner = _make_runner(connector)
    runner.state_cache = {}
    request = _make_request({"prompt": "a cat"})
    request.sampling_params = SimpleNamespace()
    kv_calls = []

    def receive_kv(state_request, **kwargs):
        assert "text_encoder_output" in state_request.prompt["additional_information"]
        kv_calls.append(state_request.request_id)

    runner.kv_transfer_manager.receive_multi_kv_cache_distributed = receive_kv
    scheduled = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(request_id=request.request_id, req=request, diffusion_kv_metadata=None)],
        scheduled_cached_reqs=SimpleNamespace(request_ids=[]),
    )
    states, new_ids = runner._update_states(scheduled)
    assert new_ids == [request.request_id]
    assert states[0].prompt is request.prompt
    assert len(connector.calls) == 1
    assert kv_calls == [request.request_id]

    scheduled.scheduled_new_reqs = []
    scheduled.scheduled_cached_reqs.request_ids = [request.request_id]
    cached_states, new_ids = runner._update_states(scheduled)
    assert cached_states == states
    assert new_ids == []
    assert len(connector.calls) == 1
    assert kv_calls == [request.request_id]


def test_non_dict_prompt_is_left_alone():
    connector = _FakeConnector(_conditioning())
    runner = _make_runner(connector)
    req = _make_request("a cat")

    runner._maybe_recv_stage_payload(req)

    assert connector.calls == []
    assert req.prompt == "a cat"


@pytest.mark.parametrize("request_id", ["dummy_req_id", "dummy_req_id/profile"])
def test_warmup_never_receives_or_publishes_payload(request_id):
    connector = _FakeConnector(raises=True)
    request = _make_request({}, request_id=request_id)
    receiver = _make_runner(connector)
    receiver._maybe_recv_stage_payload(request)
    sender = _make_sender(connector)
    sender._maybe_send_stage_payload([request], [_make_output(prompt_embeds=torch.ones(1))])
    assert connector.calls == []
    assert connector.put_calls == []


@pytest.mark.parametrize("inline", [{}, {"text_encoder_output": None}])
def test_failed_transfer_requires_complete_inline_payload(inline):
    runner = _make_runner(_FakeConnector(raises=True))
    with pytest.raises(RuntimeError, match="missing keys.*text_encoder_output"):
        runner._maybe_recv_stage_payload(_make_request({"additional_information": inline}))


def test_send_puts_declared_keys_and_attaches_a_handle():
    connector = _FakeConnector()
    runner = _make_sender(connector)
    req = _make_request({"prompt": "a cat"})
    output = _make_output(prompt_embeds=torch.zeros(2, 8), latents=torch.zeros(1, 4))

    runner._maybe_send_stage_payload([req], [output])

    from_stage, to_stage, put_key, data = connector.put_calls[0]
    assert (from_stage, to_stage, put_key) == ("0", "1", "req-7_0_0")
    # Only declared keys travel over the connector.
    assert isinstance(data, dict)
    assert set(data) == {"prompt_embeds"}
    handle = output.custom_output[HANDLE_KEY]
    assert handle["key"] == "req-7_0_0"
    assert handle["metadata"] == {"schema_version": 1}
    assert handle["payload_keys"] == ["prompt_embeds"]
    assert handle["size_bytes"] == 128
    # Transferred keys must not also travel inline through the orchestrator.
    assert "prompt_embeds" not in output.custom_output
    # Undeclared keys are untouched.
    assert "latents" in output.custom_output


def test_send_key_matches_the_receive_key_convention():
    sender_connector = _FakeConnector()
    sender = _make_sender(sender_connector)
    req = _make_request({"prompt": "a cat"})
    sender._maybe_send_stage_payload([req], [_make_output(prompt_embeds=torch.zeros(2, 8))])

    recv_connector = _FakeConnector(_conditioning())
    receiver = _make_runner(recv_connector)
    receiver._maybe_recv_stage_payload(_make_request({"prompt": "a cat"}))

    assert sender_connector.put_calls[0][2] == recv_connector.calls[0][2]


def test_stage_without_declared_output_keys_never_puts():
    connector = _FakeConnector()
    runner = _make_sender(connector, payload_keys=())
    output = _make_output(prompt_embeds=torch.zeros(2, 8))

    runner._maybe_send_stage_payload([_make_request({"prompt": "a cat"})], [output])

    assert connector.put_calls == []
    assert HANDLE_KEY not in output.custom_output


def test_missing_outgoing_edge_is_reported_not_sent():
    connector = _FakeConnector()
    runner = _make_sender(connector, send_stages=(None, None))
    output = _make_output(prompt_embeds=torch.zeros(2, 8))

    runner._maybe_send_stage_payload([_make_request({"prompt": "a cat"})], [output])

    assert connector.put_calls == []
    assert HANDLE_KEY not in output.custom_output


def test_output_without_the_declared_keys_is_skipped():
    connector = _FakeConnector()
    runner = _make_sender(connector)
    output = _make_output(latents=torch.zeros(1, 4))

    runner._maybe_send_stage_payload([_make_request({"prompt": "a cat"})], [output])

    assert connector.put_calls == []
    assert HANDLE_KEY not in output.custom_output


@pytest.mark.parametrize(
    "connector",
    [
        _FakeConnector(raises=True),
        _FakeConnector(put_result=(False, 0, None)),
    ],
    ids=["raises", "rejected"],
)
def test_failed_send_leaves_the_inline_payload_unannotated(connector):
    runner = _make_sender(connector)
    output = _make_output(prompt_embeds=torch.zeros(2, 8))

    runner._maybe_send_stage_payload([_make_request({"prompt": "a cat"})], [output])

    assert HANDLE_KEY not in output.custom_output
    assert "prompt_embeds" in output.custom_output


class _FakeTPGroup:
    def __init__(self, rank_in_group, world_size=2):
        self.rank_in_group = rank_in_group
        self.world_size = world_size
        self.broadcast_calls: list[object] = []
        self.leader_packet: object = None

    def broadcast_object(self, obj, src=0):
        self.broadcast_calls.append(obj)
        return obj if self.rank_in_group == src else self.leader_packet


@pytest.mark.parametrize("rank_in_group", [0, 1], ids=["leader", "follower"])
def test_send_puts_only_on_the_tp_leader(monkeypatch, rank_in_group):
    connector = _FakeConnector()
    runner = _make_sender(connector)
    tp_group = _FakeTPGroup(rank_in_group)
    expected_handle = {
        "key": "req-7_0_0",
        "from_stage": "0",
        "to_stage": "1",
        "size_bytes": 128,
        "metadata": {"schema_version": 1},
        "payload_keys": ["prompt_embeds"],
    }
    tp_group.leader_packet = {"req-7": expected_handle}
    monkeypatch.setattr(DiffusionModelRunner, "_get_local_tp_group", staticmethod(lambda: tp_group))
    output = _make_output(prompt_embeds=torch.zeros(2, 8))

    runner._maybe_send_stage_payload([_make_request({"prompt": "a cat"})], [output])

    assert len(connector.put_calls) == (1 if rank_in_group == 0 else 0)
    # Every rank ends up reporting the same handle, leader or not.
    assert output.custom_output[HANDLE_KEY] == expected_handle
    # ...and every rank drops the inline copy, not just the sender.
    assert "prompt_embeds" not in output.custom_output
    assert len(tp_group.broadcast_calls) == 1


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("accepted", [True, False])
def test_send_elects_one_tp_sp_leader(monkeypatch, tp_size, accepted):
    from vllm_omni.diffusion.distributed import parallel_state

    connector = _FakeConnector(put_result=(accepted, 128, {"schema_version": 1}))
    packets = {}
    outputs = []

    class GridGroup:
        def __init__(self, dimension, peer_rank, rank, world_size):
            self.key = (dimension, peer_rank)
            self.rank_in_group = rank
            self.world_size = world_size

        def broadcast_object(self, value, src=0):
            if self.rank_in_group == src:
                packets[self.key] = value
            return packets[self.key]

    for sp_rank in range(2):
        for tp_rank in range(tp_size):
            runner = _make_sender(connector)
            tp_group = GridGroup("tp", sp_rank, tp_rank, tp_size)
            sp_group = GridGroup("sp", tp_rank, sp_rank, 2)
            monkeypatch.setattr(runner, "_get_local_tp_group", lambda: tp_group)
            monkeypatch.setattr(parallel_state, "get_sp_group", lambda: sp_group)
            output = _make_output(prompt_embeds=torch.zeros(2, 8))

            runner._maybe_send_stage_payload([_make_request({})], [output])

            outputs.append(output.custom_output)
            assert (HANDLE_KEY in output.custom_output) == accepted
            assert ("prompt_embeds" in output.custom_output) != accepted

    assert len(connector.put_calls) == 1
    if accepted:
        assert all(output[HANDLE_KEY] == outputs[0][HANDLE_KEY] for output in outputs)
