# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from argparse import Namespace

import pytest

from tests.entrypoints.test_async_omni import get_async_omni_instance
from tests.entrypoints.test_async_omni_pause_sleep_routing import _make_omni
from vllm_omni.entrypoints import async_omni as async_omni_module
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.omni_base import OmniEngineDeadError
from vllm_omni.errors import OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def make_omni(stage_types=("llm",)):
    omni = _make_omni(stage_types=list(stage_types))
    omni.request_states = {}
    omni._pending_submits = {}
    omni.final_output_task = None
    omni._shutdown_base = lambda: None
    omni._idle_timeout_s = 300.0
    return omni


async def expire(omni):
    """Run the real timer callback without waiting 300 wall-clock seconds."""
    timer = omni._idle_timer
    assert timer is not None
    timer._run()
    timer.cancel()
    await omni._idle_transition


async def test_default_disabled_preserves_manual_admission():
    omni = make_omni()
    omni._idle_timeout_s = None
    await omni.sleep(level=1)
    async with omni._stage_request("manual", stage_id=0):
        assert omni._paused  # Disabled stage calls keep their original ungated path.
    waiter = asyncio.create_task(omni._wait_for_admission())
    await asyncio.sleep(0)
    assert not waiter.done()
    await omni.wake_up()
    assert not waiter.done()
    await omni.resume_generation()
    await waiter
    assert omni._idle_timer is None
    assert not omni._sleeping_tags


@pytest.mark.parametrize("stage_types", [("llm",), ("diffusion",), ("llm", "diffusion")])
async def test_sleep_wake_resume_routes_and_rearms(stage_types):
    omni = make_omni(stage_types)
    omni._arm_idle_sleep()
    await expire(omni)
    assert omni._sleeping_tags
    methods = [call.kwargs["method"] for call in omni.collective_rpc.await_args_list]
    assert "pause_scheduler" not in methods
    async with omni._stage_request("one", stage_id=0):
        assert not omni._paused
        assert not omni._sleeping_tags
        assert omni._idle_timer is None
    methods = [call.kwargs["method"] for call in omni.collective_rpc.await_args_list]
    assert ("resume_scheduler" in methods) == ("llm" in stage_types)
    assert omni._idle_timer is not None
    omni.shutdown()


@pytest.mark.parametrize(
    "method", ["sleep", "wake_up", "pause_generation", "resume_generation", "release_kv_cache_memory"]
)
async def test_manual_lifecycle_rejected_before_rpc(method):
    omni = make_omni()
    with pytest.raises(OmniClientError) as raised:
        await getattr(omni, method)()
    assert raised.value.status_code == 409
    omni.collective_rpc.assert_not_awaited()
    assert not omni._paused


@pytest.mark.parametrize("busy_field", ["_admitting", "_pending_submits"])
async def test_submit_tail_defers_sleep_for_another_full_timeout(busy_field):
    omni = make_omni()
    setattr(omni, busy_field, 1)
    omni._arm_idle_sleep()
    await expire(omni)
    assert not omni._sleeping_tags
    assert omni._idle_timer.when() >= asyncio.get_running_loop().time() + 299
    setattr(omni, busy_field, 0)
    await expire(omni)
    assert omni._sleeping_tags
    omni.shutdown()


@pytest.mark.parametrize("expiry_first", [False, True])
async def test_expiry_racing_request_at_admission_lock(expiry_first):
    omni = make_omni()
    omni._arm_idle_sleep()
    timer = omni._idle_timer
    callback = timer._callback
    omni._arm_idle_sleep()
    assert omni._idle_timer is timer  # Repeated cleanup must not extend the deadline.
    entered, finish = asyncio.Event(), asyncio.Event()

    async def request():
        async with omni._stage_request("one", stage_id=0):
            callback()
            assert omni._idle_transition is None  # A stale callback cannot sleep admitted work.
            assert not omni._paused
            assert not omni._sleeping_tags
            assert omni._idle_timer is None
            entered.set()
            await finish.wait()

    async with omni._pause_cond:
        if expiry_first:
            timer._run()
            await asyncio.sleep(0)
        waiter = asyncio.create_task(request())
        await asyncio.sleep(0)
        if not expiry_first:
            timer._run()
            await asyncio.sleep(0)
        timer.cancel()
    await entered.wait()
    methods = [call.kwargs["method"] for call in omni.collective_rpc.await_args_list]
    assert methods.count("sleep") == int(expiry_first)
    assert methods.count("wake_up") == int(expiry_first)
    finish.set()
    await waiter
    assert omni._idle_timer is not None
    assert omni._idle_timer is not timer
    omni.shutdown()


@pytest.mark.parametrize("cancel_all", [False, True])
async def test_cancelled_wake_waiters_do_not_cancel_shared_transition(cancel_all, monkeypatch):
    omni = make_omni()
    omni._arm_idle_sleep()
    await expire(omni)
    started, finish = asyncio.Event(), asyncio.Event()
    original = omni.wake_up

    async def wake():
        started.set()
        await finish.wait()
        return await original()

    monkeypatch.setattr(omni, "wake_up", wake)

    async def request():
        async with omni._stage_request("one", stage_id=0):
            assert not omni._paused

    first = asyncio.create_task(request())
    await started.wait()
    second = None if cancel_all else asyncio.create_task(request())
    await asyncio.sleep(0)
    transition = omni._idle_transition
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert not transition.done()
    finish.set()
    await transition
    if second is not None:
        await second
    assert not omni.request_states
    assert omni._idle_timer is not None
    await expire(omni)
    assert omni._sleeping_tags
    omni.shutdown()


async def test_cancelled_request_during_sleep_does_not_wake(monkeypatch):
    omni = make_omni()
    started, finish = asyncio.Event(), asyncio.Event()
    original = omni.sleep

    async def sleep(**kwargs):
        started.set()
        await finish.wait()
        return await original(**kwargs)

    monkeypatch.setattr(omni, "sleep", sleep)
    omni._arm_idle_sleep()
    sleeper = asyncio.create_task(expire(omni))
    await started.wait()
    waiter = asyncio.create_task(omni._wait_for_admission())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    finish.set()
    await sleeper
    assert omni._sleeping_tags
    assert omni._idle_timer is None
    assert omni._idle_error is None
    omni.shutdown()


@pytest.mark.parametrize("operation", ["sleep", "wake_up", "resume_generation"])
@pytest.mark.parametrize("timeout", [False, True])
async def test_transition_failure_rejects_health_and_future_work(operation, timeout, monkeypatch):
    omni = make_omni()
    if operation != "sleep":
        omni._arm_idle_sleep()
        await expire(omni)

    async def fail(**kwargs):
        if timeout:
            await asyncio.Event().wait()
        raise RuntimeError("device state unknown")

    monkeypatch.setattr(omni, operation, fail)
    if timeout:
        monkeypatch.setattr(async_omni_module, "IDLE_CONTROL_TIMEOUT_S", 0)
    if operation == "sleep":
        omni._arm_idle_sleep()
        await expire(omni)
    else:
        with pytest.raises(OmniEngineDeadError):
            await omni._wait_for_admission()
    assert omni.errored
    assert not omni.is_running
    with pytest.raises(OmniEngineDeadError):
        await omni.check_health()
    with pytest.raises(OmniEngineDeadError):
        await omni._wait_for_admission()
    assert omni._idle_timer is None
    omni.shutdown()


@pytest.mark.parametrize("off_loop", [False, True])
async def test_shutdown_blocks_late_resume_and_is_idempotent(off_loop, monkeypatch):
    omni = make_omni()
    omni._arm_idle_sleep()
    await expire(omni)
    started, finish = asyncio.Event(), asyncio.Event()
    original = omni.wake_up

    async def wake():
        await original()
        started.set()
        try:
            await finish.wait()
        except asyncio.CancelledError:
            # An executor may complete even after local cancellation.
            await finish.wait()

    monkeypatch.setattr(omni, "wake_up", wake)
    waiter = asyncio.create_task(omni._wait_for_admission())
    await started.wait()
    transition = omni._idle_transition
    if off_loop:
        await asyncio.to_thread(omni.shutdown)
    else:
        omni.shutdown()
    assert omni._idle_closed
    finish.set()
    await transition
    with pytest.raises(OmniEngineDeadError):
        await waiter
    omni.shutdown()
    assert omni._idle_error is None
    assert omni._paused
    assert omni._idle_timer is None
    assert omni._idle_transition is None


async def test_shutdown_before_transition_starts_raises_engine_dead():
    omni = make_omni()
    omni._arm_idle_sleep()
    await expire(omni)
    waiter = asyncio.create_task(omni._wait_for_admission())
    await asyncio.sleep(0)
    omni.shutdown()
    with pytest.raises(OmniEngineDeadError):
        await waiter


async def test_enable_idle_sleep_validates_engine_state(monkeypatch):
    from vllm_omni.config.omni_config import OmniStageModelConfig

    omni = make_omni()
    omni._idle_timeout_s = None
    omni.engine.stage_configs[0].stage_id = 0
    omni.engine.stage_configs[0].model_config = OmniStageModelConfig(enable_sleep_mode=False)
    monkeypatch.setattr(omni, "_get_pd_separation_pair", lambda: None)
    monkeypatch.setattr(AsyncOmni, "weight_transfer_enabled", property(lambda _: False))

    with pytest.raises(ValueError, match="finite and positive"):
        omni.enable_idle_sleep(0)
    with pytest.raises(ValueError, match="enable_sleep_mode"):
        omni.enable_idle_sleep(300)

    omni.engine.stage_configs[0].model_config = OmniStageModelConfig(enable_sleep_mode=True)
    monkeypatch.setattr(omni, "_get_pd_separation_pair", lambda: (0, 1))
    with pytest.raises(ValueError, match="PD or weight transfer"):
        omni.enable_idle_sleep(300)

    monkeypatch.setattr(omni, "_get_pd_separation_pair", lambda: None)
    omni.enable_idle_sleep(300)
    assert omni._idle_timeout_s == 300.0
    assert omni._idle_timer is not None
    omni.shutdown()


@pytest.mark.parametrize("enabled,arrival,expected", [(True, None, 10.0), (True, 2.0, 2.0), (False, None, 20.0)])
@pytest.mark.parametrize("streaming", [False, True])
async def test_generate_registers_before_submit_and_preserves_arrival(
    enabled, arrival, expected, streaming, monkeypatch
):
    submitted = []

    async def add(**kwargs):
        assert kwargs["request_id"] in omni.request_states
        assert getattr(omni, "_admitting", 0) == (0 if streaming else 1)
        omni._arm_idle_sleep()
        assert omni._idle_timer is None
        submitted.append(kwargs["arrival_time"])

    async def stream():
        yield "hello"

    async def add_streaming(*, first_chunk_submitted, **kwargs):
        await add(**kwargs)
        first_chunk_submitted.set_result(None)
        return asyncio.create_task(asyncio.sleep(0))

    omni = get_async_omni_instance(fake_add_request=add)
    omni.final_output_task = None
    omni._shutdown_base = lambda: None
    e2e_starts = []
    original_process = omni._process_orchestrator_results

    async def process_results(request_id, metrics, final_stage_id_for_e2e, req_start_ts, wall_start_ts):
        e2e_starts.append(req_start_ts[request_id])
        async for out in original_process(request_id, metrics, final_stage_id_for_e2e, req_start_ts, wall_start_ts):
            yield out

    monkeypatch.setattr(omni, "_process_orchestrator_results", process_results)
    monkeypatch.setattr(omni, "_add_streaming_input_request", add_streaming)
    omni._idle_timeout_s = 300.0 if enabled else None
    omni._sleeping_tags = set()
    clock = [10.0]
    monkeypatch.setattr(async_omni_module.time, "time", lambda: clock[0])
    ready = asyncio.Event()

    async def transition():
        await ready.wait()
        omni._idle_transition = None

    if enabled:
        omni._idle_transition = asyncio.create_task(transition())
    else:
        omni._paused = True
    generator = omni.generate(
        stream() if streaming else "hello", "one", sampling_params_list=[None], arrival_time=arrival
    )
    result = asyncio.create_task(anext(generator))
    await asyncio.sleep(0)
    assert submitted == []
    clock[0] = 20.0
    ready.set()
    async with omni._pause_cond:
        omni._paused = False
        omni._pause_cond.notify_all()
    await result
    with pytest.raises(StopAsyncIteration):
        await anext(generator)
    assert submitted == [expected]
    assert e2e_starts == [expected]
    assert not omni.request_states
    assert (omni._idle_timer is not None) == enabled
    omni.shutdown()


async def test_generate_failure_before_registration_rearms_without_abort(mocker):
    omni = get_async_omni_instance()
    omni.final_output_task = None
    omni._shutdown_base = lambda: None
    omni.engine.stage_configs = make_omni(("diffusion",)).engine.stage_configs
    omni._idle_timeout_s = 300.0
    omni._sleeping_tags = set()
    omni._arm_idle_sleep()
    timer = omni._idle_timer
    abort = mocker.patch.object(omni, "_abort_internal_requests")
    with pytest.raises(ValueError, match="single prompt"):
        await anext(omni.generate(["one", "two"], "invalid", sampling_params_list=[None]))
    abort.assert_not_called()
    assert omni._admitting == 0
    assert omni._idle_timer is not None
    assert omni._idle_timer is not timer
    omni.shutdown()


async def test_auto_rpc_timeout_is_finite_and_closed_blocks_later_rpc():
    omni = make_omni()
    omni.collective_rpc = AsyncOmni.collective_rpc.__get__(omni)
    omni._idle_transition = asyncio.current_task()
    await omni.collective_rpc("sleep")
    assert omni.engine.collective_rpc_async.await_args.kwargs["timeout"] == 120.0
    omni.engine.collective_rpc_async.reset_mock()
    omni._idle_closed = True
    with pytest.raises(OmniEngineDeadError):
        await omni.collective_rpc("resume_scheduler")
    omni.engine.collective_rpc_async.assert_not_awaited()
    omni._idle_transition = None


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_cli_rejects_invalid_idle_timeout(value):
    from vllm_omni.entrypoints.cli.serve import OmniServeCommand

    with pytest.raises(ValueError, match="finite and positive"):
        OmniServeCommand().validate(Namespace(idle_sleep_timeout=value))


@pytest.mark.parametrize(
    "overrides", [{"headless": True}, {"stage_id": 0}, {"stage_id": "none"}, {"api_server_count": 2}]
)
def test_cli_rejects_nonlocal_or_multiple_frontends(overrides):
    from vllm_omni.entrypoints.cli.serve import OmniServeCommand

    args = {"idle_sleep_timeout": 300, "headless": False, "stage_id": None, "api_server_count": 1}
    args.update(overrides)
    with pytest.raises(ValueError, match="full local pipeline"):
        OmniServeCommand().validate(Namespace(**args))
