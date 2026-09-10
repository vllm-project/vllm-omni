# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU generator/wire tests; actual NPU sampler validation runs on the server."""

from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
import torch
from vllm import SamplingParams
from vllm.distributed import parallel_state
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler, OmniARScheduler
from vllm_omni.core.sched.output import OmniNewRequestData
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.pd_continuation import PD_PREFILL_KEY, PD_RESUME_KEY, PD_RNG_STATE_KEY, PDContinuation
from vllm_omni.entrypoints.pd_utils import PDDisaggregationMixin
from vllm_omni.request import OmniRequest
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner
from vllm_omni.worker.pd_rng import PDRNGHandoff, PDRNGState, capture_pd_rng_states, restore_pd_rng_state

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _params(seed=42, max_tokens=32):
    return SamplingParams(
        seed=seed,
        temperature=1,
        max_tokens=max_tokens,
        extra_args={PD_RESUME_KEY: True, "kv_transfer_params": {"do_remote_prefill": True}},
    )


def _sample(generator, history):
    # Full-vocabulary exponential-race sampling, with history-dependent logits.
    logits = torch.arange(31, dtype=torch.float32).roll(sum(history) % 31) / 20
    noise = torch.empty_like(logits).exponential_(generator=generator)
    return int((logits.softmax(-1) / noise).argmax())


def _producer(generator, tokens, params=None):
    sp = PDDisaggregationMixin._prepare_prefill_sampling_params("req", params or _params())
    return SimpleNamespace(sampling_params=sp, generator=generator, output_token_ids=list(tokens))


def _capture(request, tokens):
    return capture_pd_rng_states(
        {"req": request},
        ["req"],
        [tokens],
        tp_group=lambda: SimpleNamespace(world_size=1, rank_in_group=0),
        async_scheduling=False,
    )


@pytest.mark.parametrize("seed", [42, 43, -2])
@pytest.mark.parametrize("limit", [1, 2, 32])
def test_random_sequence_matches_through_handoff_and_request_wire(seed, limit):
    reference = torch.Generator().manual_seed(seed)
    expected = []
    for _ in range(limit):
        expected.append(_sample(reference, expected))

    producer = torch.Generator().manual_seed(seed)
    first = _sample(producer, [])
    payload = _capture(_producer(producer, [first], _params(seed, limit)), [first])["req"]
    output = SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=[first], finish_reason="length")],
        kv_transfer_params={PD_RNG_STATE_KEY: payload},
    )
    continuation = PDContinuation.from_output(output, [1, 2, 3])
    core = OmniEngineCoreRequest(
        request_id="req",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=_params(seed, limit),
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        pd_continuation=continuation,
    )
    core = msgspec.msgpack.decode(msgspec.msgpack.encode(core), type=OmniEngineCoreRequest)
    request = OmniRequest.from_engine_core_request(core, block_hasher=None)
    worker = OmniNewRequestData.from_request(request, ([0],))
    wire = msgspec.msgpack.decode(msgspec.msgpack.encode(worker))
    assert wire["pd_rng_state"] == payload
    assert worker.pd_rng_state == payload
    actual = list(worker.initial_output_token_ids)
    consumer = torch.Generator().manual_seed(seed)
    if limit > 1:  # Terminal y1 is handled by the scheduler without a D forward.
        restore_pd_rng_state(consumer, worker.pd_rng_state, seed=seed, tp_rank=0, tp_size=1, req_id="req")
        assert torch.equal(consumer.get_state(), producer.get_state())
        for _ in range(limit - 1):
            actual.append(_sample(consumer, actual))
        assert torch.equal(consumer.get_state(), reference.get_state())
    assert actual == expected


def test_does_not_export_discarded_partial_prefill_or_touch_normal_requests():
    generator = torch.Generator().manual_seed(42)
    request = _producer(generator, [])
    untouched = generator.get_state().clone()
    group = Mock(side_effect=AssertionError("must not use a TP collective"))
    assert (
        capture_pd_rng_states(
            {"req": request},
            ["req"],
            [[]],
            tp_group=group,
            async_scheduling=False,
        )
        is None
    )
    assert torch.equal(generator.get_state(), untouched)
    for extra, temperature in [
        ({}, 1),
        ({PD_PREFILL_KEY: True}, 0),
    ]:
        request.sampling_params = SamplingParams(temperature=temperature, seed=42, extra_args=extra)
        assert (
            capture_pd_rng_states(
                {"req": request},
                ["req"],
                [[1]],
                tp_group=group,
                async_scheduling=True,
            )
            is None
        )
    group.assert_not_called()


def test_capture_preserves_each_tp_rank_stream(monkeypatch):
    local = torch.Generator().manual_seed(42)
    peer = torch.Generator().manual_seed(42)
    _sample(local, [])
    for _ in range(3):
        _sample(peer, [])
    peer_state = PDRNGState(42, "cpu", peer.get_state().numpy().tobytes())
    cpu_group = object()

    def gather(results, packet, group):
        assert group is cpu_group
        results[:] = [packet, ({"req": peer_state}, None)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    payload = capture_pd_rng_states(
        {"req": _producer(local, [1])},
        ["req"],
        [[1]],
        tp_group=lambda: SimpleNamespace(world_size=2, rank_in_group=0, cpu_group=cpu_group),
        async_scheduling=False,
    )["req"]
    for rank, expected in enumerate([local, peer]):
        consumer = torch.Generator().manual_seed(42)
        restore_pd_rng_state(consumer, payload, seed=42, tp_rank=rank, tp_size=2, req_id="req")
        assert torch.equal(consumer.get_state(), expected.get_state())


@pytest.mark.parametrize("failure", ["seed", "device", "empty", "tp_size", "version", "missing"])
def test_incompatible_state_fails_before_sampling(failure):
    generator = torch.Generator().manual_seed(42)
    state = PDRNGState(
        43 if failure == "seed" else 42,
        "npu" if failure == "device" else "cpu",
        b"" if failure == "empty" else generator.get_state().numpy().tobytes(),
    )
    payload = msgspec.msgpack.encode(PDRNGHandoff([state], version=2 if failure == "version" else 1))
    before = generator.get_state().clone()
    with pytest.raises(ValueError):
        restore_pd_rng_state(
            generator,
            None if failure == "missing" else payload,
            seed=42,
            tp_rank=0,
            tp_size=2 if failure == "tp_size" else 1,
            req_id="req",
        )
    assert torch.equal(generator.get_state(), before)


def test_rank_export_error_is_reported_on_all_ranks(monkeypatch):
    def gather(results, packet, group):
        results[:] = [packet, ({}, "get_state failed")]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="TP rank 1.*get_state failed"):
        capture_pd_rng_states(
            {"req": _producer(torch.Generator().manual_seed(42), [1])},
            ["req"],
            [[1]],
            tp_group=lambda: SimpleNamespace(world_size=2, rank_in_group=0, cpu_group=object()),
            async_scheduling=False,
        )


def test_logical_defaults_and_request_overrides_have_one_source():
    p, d = _params(999), _params(42)
    defaults = PDDisaggregationMixin._resolve_pd_sampling_params([p, d], (0, 1))
    assert defaults[0].seed == defaults[1].seed == 42
    defaults[1] = defaults[1].clone()
    defaults[1].seed = 43
    defaults[1].temperature = 0.8
    resolved = PDDisaggregationMixin._resolve_pd_sampling_params(defaults, (0, 1))
    producer = PDDisaggregationMixin._prepare_prefill_sampling_params("req", resolved[0])
    assert producer.seed == resolved[1].seed == 43
    assert producer.temperature == resolved[1].temperature == 0.8
    assert d.seed == 42 and p.seed == 999
    assert PD_PREFILL_KEY not in resolved[1].extra_args


def test_rng_envelope_does_not_reach_mooncake_parameters():
    mixin = PDDisaggregationMixin()
    mixin._pd_connector_info = None
    original = {PD_RNG_STATE_KEY: b"state", "remote_engine_id": "P"}
    params = mixin._build_decode_kv_params("req", _params(), original)
    assert PD_RNG_STATE_KEY not in params
    assert params["remote_engine_id"] == "P" and params["do_remote_prefill"]
    assert original[PD_RNG_STATE_KEY] == b"state"


def test_worker_generator_creation_restores_before_batch_admission(monkeypatch):
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: SimpleNamespace(world_size=1, rank_in_group=0))
    runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
    runner.device = torch.device("cpu")
    p = torch.Generator().manual_seed(42)
    first = _sample(p, [])
    data = SimpleNamespace(
        req_id="req",
        sampling_params=_params(),
        initial_output_token_ids=[first],
        pd_rng_state=_capture(_producer(p, [first]), [first])["req"],
    )
    d = runner._create_request_generator(data)
    assert torch.equal(d.get_state(), p.get_state())
    assert _sample(d, [first]) == _sample(p, [first])
    data.pd_rng_state = None
    with pytest.raises(ValueError, match="missing"):
        runner._create_request_generator(data)
    data.initial_output_token_ids = []  # Ordinary seeded request.
    fresh = runner._create_request_generator(data)
    assert torch.equal(fresh.get_state(), torch.Generator().manual_seed(42).get_state())
    data.initial_output_token_ids = [first]
    data.sampling_params = SamplingParams(temperature=0, seed=42)
    assert runner._create_request_generator(data) is None


@pytest.mark.parametrize("scheduler_cls", [OmniARScheduler, OmniARAsyncScheduler])
def test_schedule_wrapper_transfers_rng_to_worker(monkeypatch, scheduler_cls):
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: SimpleNamespace(world_size=1, rank_in_group=0))
    producer = torch.Generator().manual_seed(42)
    first = _sample(producer, [])
    payload = _capture(_producer(producer, [first]), [first])["req"]
    request = OmniRequest(
        "req",
        [1, 2, 3],
        _params(),
        None,
        pd_continuation=PDContinuation(3, [first], rng_state=payload),
    )
    # Upstream schedule produces BASE NewRequestData. Test the actual Omni
    # schedule wrapper, which constructs its own OmniNewRequestData instance.
    upstream = SimpleNamespace(scheduled_new_reqs=[NewRequestData.from_request(request, ([0],))])
    monkeypatch.setattr(Scheduler, "schedule", lambda self, throttle_prefills=False: upstream, raising=False)
    scheduler = scheduler_cls.__new__(scheduler_cls)
    scheduler.requests = {"req": request}
    scheduler.waiting, scheduler.running = [], []
    scheduler.chunk_transfer_adapter = scheduler.input_coordinator = None
    scheduler._finish_pd_terminal_receives = Mock()
    scheduler._restore_omni_wait_queues = Mock()
    scheduler._drop_aborted_queued_requests = Mock()
    scheduler._process_pending_omni_inputs = Mock()
    scheduler._resync_streaming_input_counter = Mock()
    scheduler._should_defer_waiting_admission = Mock(return_value=False)
    scheduler.get_finished_requests_needing_kv_transfer = Mock(return_value={})
    scheduler._wrap_omni_scheduler_output = lambda output, **kwargs: output
    data = scheduler.schedule().scheduled_new_reqs[0]
    assert isinstance(data, OmniNewRequestData)
    assert data.pd_rng_state == payload
    assert data.initial_output_token_ids == [first]
    assert data.prompt_token_ids == [1, 2, 3]
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(data))["pd_rng_state"] == payload
    runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
    runner.device = torch.device("cpu")
    consumer = runner._create_request_generator(data)
    assert torch.equal(consumer.get_state(), producer.get_state())
    assert _sample(consumer, [first]) == _sample(producer, [first])
