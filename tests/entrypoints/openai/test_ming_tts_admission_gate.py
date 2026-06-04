import asyncio
from types import SimpleNamespace

import pytest

from vllm_omni.entrypoints.openai.serving_speech import MingTTSAdmissionGate, OmniOpenAIServingSpeech


@pytest.mark.asyncio
async def test_ming_tts_admission_gate_releases_full_batch_together():
    gate = MingTTSAdmissionGate(max_batch_size=2, max_wait_ms=1000)

    async def wait(name):
        cohort = await gate.wait("same-key")
        return name, cohort

    first = asyncio.create_task(wait("first"))
    await asyncio.sleep(0)
    assert not first.done()

    second = asyncio.create_task(wait("second"))
    results = await asyncio.wait_for(asyncio.gather(first, second), timeout=1)

    assert results[0][0] == "first"
    assert results[1][0] == "second"
    assert results[0][1][0] == results[1][1][0]
    assert results[0][1][1] == 2


@pytest.mark.asyncio
async def test_ming_tts_admission_gate_yields_after_full_batch_release():
    gate = MingTTSAdmissionGate(max_batch_size=2, max_wait_ms=1000)
    resumed = []

    async def wait(name):
        cohort = await gate.wait("same-key")
        resumed.append((name, cohort))

    first = asyncio.create_task(wait("first"))
    await asyncio.sleep(0)
    assert not first.done()

    second = asyncio.create_task(wait("second"))
    await asyncio.sleep(0)

    assert resumed == []
    await asyncio.wait_for(asyncio.gather(first, second), timeout=1)


@pytest.mark.asyncio
async def test_ming_tts_admission_gate_separates_keys():
    gate = MingTTSAdmissionGate(max_batch_size=2, max_wait_ms=20)

    first = asyncio.create_task(gate.wait("a"))
    second = asyncio.create_task(gate.wait("b"))

    cohorts = await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
    assert cohorts[0][0] != cohorts[1][0]
    assert cohorts[0][1] == 1
    assert cohorts[1][1] == 1


@pytest.mark.asyncio
async def test_ming_tts_admission_gate_cleans_cancelled_waiter():
    gate = MingTTSAdmissionGate(max_batch_size=2, max_wait_ms=1000)

    waiter = asyncio.create_task(gate.wait("same-key"))
    await asyncio.sleep(0)
    assert gate._queues

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert gate._queues == {}
    assert gate._timers == {}


def test_ming_tts_admission_config_reads_stage_config():
    serving = object.__new__(OmniOpenAIServingSpeech)
    serving._tts_stage = SimpleNamespace(
        speech_admission_config={
            "max_batch_size": 4,
            "max_wait_ms": 25.0,
        }
    )

    config = serving._resolve_ming_tts_admission_config()

    assert config == {"max_batch_size": 4, "max_wait_ms": 25.0}


def test_ming_tts_admission_config_defaults_without_stage_config():
    serving = object.__new__(OmniOpenAIServingSpeech)
    serving._tts_stage = None

    config = serving._resolve_ming_tts_admission_config()

    assert config == {"max_batch_size": 8, "max_wait_ms": 15.0}
