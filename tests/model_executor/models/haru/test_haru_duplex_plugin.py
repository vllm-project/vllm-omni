# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Haru duplex plugin: append planning, sampling policy and session options (CPU)."""

from __future__ import annotations

import asyncio
import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError
from vllm_omni.model_executor.models.haru.duplex.plugin import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    HaruDuplexPlugin,
    _decode_reference,
)
from vllm_omni.model_executor.models.haru.prompt import prompt_length

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _plugin() -> HaruDuplexPlugin:
    return HaruDuplexPlugin(lambda audio, *_: None)


def _payload(samples: int = FRAME_SAMPLES, rate: int = SAMPLE_RATE) -> dict[str, object]:
    pcm = np.zeros(samples, dtype="<f4")
    return {"format": "pcm_f32le", "sample_rate_hz": rate, "audio": base64.b64encode(pcm.tobytes()).decode()}


def _wav_uri(seconds: float, rate: int) -> str:
    t = np.arange(int(seconds * rate)) / rate
    buffer = io.BytesIO()
    sf.write(buffer, 0.3 * np.sin(2 * np.pi * 220 * t), rate, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode()


def _plan(plugin: HaruDuplexPlugin, seq: int, payload: object | None = None, slots: int = 4):
    return plugin.plan_append(
        request_id="req",
        fence=DuplexFence("session", epoch=2, turn_id=1),
        session_config={},
        runtime_config={"haru": {"seed": 3}, "haru_prompt_slots": slots},
        seq=seq,
        turn_seq=1,
        payload=_payload() if payload is None else payload,
        final=False,
        sampling_params=SamplingParams(),
    )


def test_the_first_append_schedules_the_prompt_and_later_appends_one_frame() -> None:
    plugin = _plugin()
    first, second = _plan(plugin, seq=1), _plan(plugin, seq=2)

    assert first.prompt["prompt_token_ids"] == [0] * 4
    assert second.prompt["prompt_token_ids"] == [0]
    buffer = first.prompt["model_intermediate_buffer"]
    assert buffer["request_id"] == "req" and buffer["global_request_id"] == ["session"]
    assert buffer["duplex"]["runtime_config"] == {"haru": {"seed": 3}}
    assert "runtime_config" not in second.prompt["model_intermediate_buffer"]["duplex"]


def _nonfinite_payload() -> dict[str, object]:
    pcm = np.zeros(FRAME_SAMPLES, dtype="<f4")
    pcm[7] = np.nan
    return {"format": "pcm_f32le", "sample_rate_hz": SAMPLE_RATE, "audio": base64.b64encode(pcm.tobytes()).decode()}


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (_payload(FRAME_SAMPLES - 1), "exactly 3200 samples"),
        (_payload(rate=16_000), "sample_rate_hz must be 24000"),
        (_nonfinite_payload(), "must be finite"),
    ],
)
def test_appends_are_exactly_one_finite_24k_frame(payload: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _plan(_plugin(), seq=2, payload=payload)


def test_sampling_is_one_delta_token_per_step_with_session_overrides() -> None:
    (params,) = _plugin().configure_sampling_params(
        runtime_config={"haru_sampling": {"top_k": 5, "temperature": 0.0}},
        defaults=(SamplingParams(max_tokens=99, temperature=0.7, top_k=25, repetition_penalty=1.3),),
    )
    assert params.max_tokens == 1
    assert params.output_kind == RequestOutputKind.DELTA
    assert params.repetition_penalty == 1.0 and params.detokenize is False
    assert (params.top_k, params.temperature) == (5, 0.0)


def test_reference_audio_is_resampled_to_24k_pcm16_and_validated() -> None:
    encoded, samples = _decode_reference(_wav_uri(2.0, 16_000))
    pcm = np.frombuffer(base64.b64decode(encoded), dtype="<i2")
    assert samples == pcm.size == 2 * SAMPLE_RATE
    assert _decode_reference(None) == ("", 0)
    with pytest.raises(DuplexRuntimeConfigError, match="data URI"):
        _decode_reference("https://example.com/voice.wav")
    with pytest.raises(DuplexRuntimeConfigError, match="could not be decoded"):
        _decode_reference("data:audio/wav;base64," + base64.b64encode(b"not audio").decode())


def test_prepare_runtime_config_resolves_options_and_prompt_slots(checkpoint_dir) -> None:
    plugin = _plugin()
    model_config = SimpleNamespace(model=str(checkpoint_dir))
    config = DuplexSessionConfig(
        instructions="hi",
        ref_audio=_wav_uri(1.0, SAMPLE_RATE),
        temperature=0.0,
        extra_body={"guidance_scale": "1.25", "top_k": 7, "seed": 11},
    )

    runtime = asyncio.run(plugin.prepare_runtime_config(config, model_config=model_config))

    options = runtime["haru"]
    assert (options["system_prompt"], options["guidance_scale"], options["seed"]) == ("hi", 1.25, 11)
    assert runtime["haru_sampling"] == {"top_k": 7, "temperature": 0.0}
    reference_samples = len(base64.b64decode(options["reference_pcm16"])) // 2
    assert runtime["haru_prompt_slots"] == prompt_length(
        plugin._tokenizer, "hi", reference_samples, False, FRAME_SAMPLES
    )
    assert plugin.data_plane.silent_ids == {248076, 248071}


@pytest.mark.parametrize(
    ("extra_body", "temperature", "match"),
    [
        ({"guidance_scale": 2.5}, None, "guidance_scale"),
        ({"guidance_scale": float("nan")}, None, "guidance_scale"),
        ({"seed": 2**64}, None, "seed"),
        ({"seed": -1}, None, "seed"),
        ({"top_p": 0.0}, None, "top_p"),
        ({"top_k": -2}, None, "top_k"),
        ({"eop_bias": float("nan")}, None, "eop_bias"),
        ({"eop_bias": 1e39}, None, "eop_bias"),
        ({}, float("inf"), "temperature"),
    ],
)
def test_session_options_that_would_fail_in_the_worker_are_rejected(
    checkpoint_dir, extra_body: dict, temperature: float | None, match: str
) -> None:
    config = DuplexSessionConfig(extra_body=extra_body, temperature=temperature)
    with pytest.raises(DuplexRuntimeConfigError, match=match):
        asyncio.run(_plugin().prepare_runtime_config(config, model_config=SimpleNamespace(model=str(checkpoint_dir))))


def test_instructions_are_fixed_once_the_session_exists() -> None:
    current = {"haru": {"system_prompt": "card A"}}
    plugin = _plugin()
    assert plugin.runtime_config_for_update(DuplexSessionConfig(instructions="card A"), current) == current
    with pytest.raises(DuplexRuntimeConfigError, match="instructions cannot be changed"):
        plugin.runtime_config_for_update(DuplexSessionConfig(instructions="card B"), current)


@pytest.mark.parametrize("max_sessions", [1, 4])
def test_capabilities_follow_the_session_limit(max_sessions: int) -> None:
    caps = _plugin().capabilities(max_sessions=max_sessions)
    assert caps.supports_multi_session is (max_sessions > 1)
    assert caps.supports_core_resumable_request and caps.supports_realtime_endpoint
    assert caps.chunk_period_ms == 134
