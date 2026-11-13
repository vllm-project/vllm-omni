# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""GPU coverage for Haru-4B on the unified full-duplex framework.

Boots the Haru deploy (``vllm_omni/deploy/haru.yaml``) and runs two lockstep
realtime sessions at once over ``/v1/realtime?duplex=1`` (the second one joins
after the first has started): every appended 133 ms frame (3200 samples of
24 kHz ``pcm_f32le``) must be answered by exactly one 3200-sample PCM16 model
frame, and both sessions must speak (``speak_first``). Outputs are not compared across sessions:
the engine's kernels are not batch-invariant, so a session's text can change
with the batch its prompt ran in.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

websockets = pytest.importorskip("websockets")

pytestmark = pytest.mark.omni

MODEL = os.environ.get("HARU_MODEL_PATH") or "higurashi-ai/Haru-4B"
SAMPLE_RATE = 24_000
FRAME_SAMPLES = 3200
NUM_APPENDS = 150  # 20 s of conversation
ASSETS = Path(__file__).resolve().parents[2] / "assets" / "minicpmo_4_5"
# A character card in the fixed layout the model was trained on (see the Haru README).
CARD = "\n".join(
    [
        "You are the person described below, talking with the other person by voice.",
        "[Your Persona]",
        "Haru is 22 and a senior in college. Easygoing and friendly, talks at a quick pace and asks questions.",
        "[Your Name]",
        "Haru",
        "[Your Backchannels]",
        "moderate",
        "[Relationship]",
        "Haru and the other person have never met before.",
        "[Situation]",
        "The two are meeting for the first time and start by introducing themselves.",
    ]
)

SERVER_PARAMS = [
    pytest.param(
        OmniServerParams(model=MODEL, stage_config_path=get_deploy_config_path("haru.yaml"), use_stage_cli=False),
        id="haru-single-stage",
    )
]


def _load_24k(path: Path) -> np.ndarray:
    from scipy.signal import resample_poly

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if rate == SAMPLE_RATE:
        return mono
    gcd = math.gcd(rate, SAMPLE_RATE)
    return resample_poly(mono, SAMPLE_RATE // gcd, rate // gcd).astype(np.float32)


def _wav_data_uri(audio: np.ndarray) -> str:
    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _user_frames() -> np.ndarray:
    """The caller stays silent: with ``speak_first`` the model opens the conversation (append 0 starts the prompt)."""
    return np.zeros((NUM_APPENDS, FRAME_SAMPLES), np.float32)


async def _next_frame(ws) -> tuple[bytes, str]:
    """Lockstep: read events until the one model frame that answers an append; return it and its text."""
    text = ""
    while True:
        event = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
        assert event["type"] != "error", event
        if event["type"] == "response.output_audio_transcript.delta":
            text += event.get("delta", "")
        elif event["type"] == "response.output_audio.delta":
            return base64.b64decode(event["delta"]), text


async def _session(
    url: str, voice: str, frames: np.ndarray, started: asyncio.Event | None = None
) -> tuple[list[bytes], str]:
    session = {
        "modalities": ["audio", "text"],
        "instructions": CARD,
        "input_audio_format": "pcm_f32le",
        "output_audio_format": "pcm16",
        "sample_rate_hz": SAMPLE_RATE,
        "audio": {"input": {"sample_rate_hz": SAMPLE_RATE}, "output": {"sample_rate_hz": SAMPLE_RATE}},
        "turn_detection": None,
        "temperature": 0.0,
        "ref_audio": voice,
        # speak_first: the model opens the conversation, so the session is audible.
        "extra_body": {"auto_response": True, "seed": 1234, "speak_first": True},
    }
    outputs: list[bytes] = []
    transcript = ""
    async with websockets.connect(url, max_size=None, compression=None) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": session}))
        while (event := json.loads(await ws.recv()))["type"] != "session.created":
            assert event["type"] != "error", event
        for frame in frames:
            audio = base64.b64encode(np.ascontiguousarray(frame, dtype="<f4").tobytes()).decode("ascii")
            append = {"type": "input_audio_buffer.append", "audio": audio, "format": "pcm_f32le"}
            await ws.send(json.dumps({**append, "sample_rate_hz": SAMPLE_RATE}))
            pcm, text = await _next_frame(ws)
            outputs.append(pcm)
            if started is not None:
                started.set()
            transcript += text
        await ws.send(json.dumps({"type": "session.close"}))
    return outputs, transcript


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_haru_realtime_duplex_two_sessions(omni_server) -> None:
    url = f"ws://{omni_server.host}:{omni_server.port}/v1/realtime?duplex=1"
    voice = _wav_data_uri(_load_24k(ASSETS / "response_required_16k.wav"))
    frames = _user_frames()

    async def staggered():
        started = asyncio.Event()
        first = asyncio.create_task(_session(url, voice, frames, started))
        await started.wait()  # the second conversation joins while the first is live
        return await asyncio.gather(first, _session(url, voice, frames))

    for outputs, transcript in asyncio.run(staggered()):
        assert len(outputs) == NUM_APPENDS
        assert {len(chunk) for chunk in outputs} == {2 * FRAME_SAMPLES}  # PCM16
        pcm = np.frombuffer(b"".join(outputs), dtype="<i2").astype(np.float32) / 32768.0
        assert float(np.abs(pcm).max()) > 0.05, "the model never spoke"
        assert transcript.strip(), "the text stream stayed silent"
