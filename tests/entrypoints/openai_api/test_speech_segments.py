# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from vllm import SamplingParams

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.speech_segments import generate_speech_segments
from vllm_omni.entrypoints.openai.speech_usage import SpeechOutputTokenCounter
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def output(audio, *, sr=1000, tokens=3, finished=True, reason="stop"):
    return OmniRequestOutput(
        finished=finished,
        _multimodal_output={"audio": audio, "sr": sr},
        metrics={"stage_metrics": {"0": {"num_tokens_out": tokens, "finish_reason": reason}}},
    )


def generate(engine, **kwargs):
    return generate_speech_segments(
        engine,
        prompts=[{"part": 0}, {"part": 1}],
        request_id="speech-test",
        sampling_params_list=[SamplingParams(seed=42, max_tokens=1500)],
        silence_ms=2,
        extract_audio=OmniOpenAIServingSpeech._extract_audio_output,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_segments_emit_one_complete_waveform_and_sum_usage():
    calls = []
    parameters = []

    async def engine_generate(**kwargs):
        part = kwargs["prompt"]["part"]
        calls.append((part, "start"))
        assert kwargs["request_id"] == f"speech-test-segment-{part}"
        params = kwargs["sampling_params_list"][0]
        assert params.seed == 42 and params.max_tokens == 1500
        parameters.append(params)
        params.seed = 99
        wave = torch.tensor([part + 1.0, part + 1.0])
        yield output(wave[:1], tokens=2, finished=False)
        yield output([wave[:1], wave], tokens=part + 3)
        calls.append((part, "done"))

    engine = Mock(spec=AsyncOmni, generate=engine_generate, abort=AsyncMock())
    results = [result async for result in generate(engine)]
    assert calls == [(0, "start"), (0, "done"), (1, "start"), (1, "done")]
    assert parameters[0] is not parameters[1]
    assert len(results) == 1
    result = results[0]
    assert result.request_id == "speech-test" and result.finished
    assert result.multimodal_output["audio"].tolist() == [1, 1, 0, 0, 2, 2]
    counter = SpeechOutputTokenCounter()
    counter.observe(result)
    assert counter.total() == 7
    engine.abort.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exception", "empty", "length", "unfinished", "nan", "rate"])
async def test_segment_failure_does_not_return_partial_audio(failure):
    calls = []

    async def engine_generate(**kwargs):
        calls.append(kwargs["prompt"]["part"])
        if failure == "exception":
            raise RuntimeError("engine failed")
        yield output(
            torch.tensor([] if failure == "empty" else [float("nan") if failure == "nan" else 1.0]),
            finished=failure != "unfinished",
            reason="length" if failure == "length" else "stop",
            sr=None if failure == "rate" else 1000,
        )

    engine = Mock(spec=AsyncOmni, generate=engine_generate, abort=AsyncMock())
    with pytest.raises((ValueError, RuntimeError)):
        async for _ in generate(engine):
            pytest.fail("A failed request must not return a partial waveform")
    assert calls == [0]
    engine.abort.assert_awaited_once_with("speech-test-segment-0")


@pytest.mark.asyncio
async def test_cancellation_aborts_current_segment_and_stops_scheduling():
    started = asyncio.Event()
    calls = []

    async def engine_generate(**kwargs):
        calls.append(kwargs["prompt"]["part"])
        started.set()
        await asyncio.Event().wait()
        yield output(torch.ones(2))

    engine = Mock(spec=AsyncOmni, generate=engine_generate, abort=AsyncMock())

    async def consume():
        return [result async for result in generate(engine)]

    task = asyncio.create_task(consume())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == [0]
    engine.abort.assert_awaited_once_with("speech-test-segment-0")


@pytest.mark.asyncio
async def test_segments_reject_mismatched_sample_rates():
    async def engine_generate(**kwargs):
        yield output(torch.ones(2), sr=1000 + kwargs["prompt"]["part"])

    engine = Mock(spec=AsyncOmni, generate=engine_generate, abort=AsyncMock())
    with pytest.raises(ValueError, match="inconsistent sample rates"):
        _ = [result async for result in generate(engine)]
