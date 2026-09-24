# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E online tests for Qwen3-TTS WebSocket streaming speech.
"""

import asyncio
import json
import os

import pytest
import websockets

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"


tts_ws_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("qwen3_tts.yaml"),
            server_args=["--trust-remote-code", "--enforce-eager", "--disable-log-stats"],
            env_dict={"VLLM_DISABLE_COMPILE_CACHE": "1"},
        ),
        id="qwen3-tts-ws-customvoice",
    )
]


def _session_config(model: str) -> str:
    return json.dumps(
        {
            "type": "session.config",
            "model": model,
            "voice": "vivian",
            "language": "English",
            "response_format": "pcm",
            "stream_audio": True,
        }
    )


async def _collect_utterance(ws) -> dict:
    """Read frames until session.done closes out one flushed utterance."""
    starts: list[dict] = []
    dones: list[dict] = []
    chunk_lengths: dict[int, list[int]] = {}
    session_done: dict | None = None

    while True:
        message = await asyncio.wait_for(ws.recv(), timeout=180)
        if isinstance(message, bytes):
            if not starts:
                raise AssertionError("Received audio bytes before audio.start")
            sentence_index = starts[-1]["sentence_index"]
            chunk_lengths.setdefault(sentence_index, []).append(len(message))
            continue

        payload = json.loads(message)
        msg_type = payload.get("type")
        if msg_type == "audio.start":
            starts.append(payload)
            chunk_lengths.setdefault(payload["sentence_index"], [])
        elif msg_type == "audio.done":
            dones.append(payload)
        elif msg_type == "session.done":
            session_done = payload
            break
        elif msg_type == "error":
            raise AssertionError(f"WebSocket error: {payload['message']}")
        else:
            raise AssertionError(f"Unexpected WebSocket message: {payload}")

    return {
        "starts": starts,
        "dones": dones,
        "chunk_lengths": chunk_lengths,
        "session_done": session_done,
    }


async def _run_ws_session(host: str, port: int, model: str) -> dict:
    uri = f"ws://{host}:{port}/v1/audio/speech/stream"

    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(_session_config(model))
        await ws.send(
            json.dumps(
                {
                    "type": "input.text",
                    "text": (
                        "Hello, this is a websocket streaming test for Qwen three TTS, "
                        "and this sentence is intentionally long enough to produce audio chunks. "
                        "This is the second sentence."
                    ),
                }
            )
        )
        await ws.send(json.dumps({"type": "input.done"}))

        return await _collect_utterance(ws)


async def _run_ws_reused_connection(host: str, port: int, model: str, texts: list[str]) -> list[dict]:
    """Synthesize several utterances over a single connection."""
    uri = f"ws://{host}:{port}/v1/audio/speech/stream"
    results: list[dict] = []

    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(_session_config(model))
        for text in texts:
            await ws.send(json.dumps({"type": "input.text", "text": text}))
            await ws.send(json.dumps({"type": "input.done"}))
            results.append(await _collect_utterance(ws))
        await ws.send(json.dumps({"type": "session.close"}))

    return results


class TestQwen3TTSWebSocket:
    @pytest.mark.advanced_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    @pytest.mark.parametrize("omni_server", tts_ws_server_params, indirect=True)
    def test_streaming_pcm_output(self, omni_server) -> None:
        result = asyncio.run(_run_ws_session(omni_server.host, omni_server.port, omni_server.model))

        starts = result["starts"]
        dones = result["dones"]
        chunk_lengths = result["chunk_lengths"]
        session_done = result["session_done"]

        assert session_done is not None
        assert session_done["utterance_index"] == 0
        assert session_done["total_sentences"] == 1
        assert len(starts) == 1
        assert len(dones) == 1

        for idx, start in enumerate(starts):
            assert start["type"] == "audio.start"
            assert start["sentence_index"] == idx
            assert start["format"] == "pcm"
            assert start["sample_rate"] == 24000
            assert start["sentence_text"]

        for done in dones:
            sentence_index = done["sentence_index"]
            total_bytes = done["total_bytes"]
            assert done["error"] is False
            assert total_bytes > 0
            assert chunk_lengths[sentence_index], f"Expected binary PCM frames for sentence {sentence_index}"
            assert sum(chunk_lengths[sentence_index]) == total_bytes

    @pytest.mark.advanced_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    @pytest.mark.parametrize("omni_server", tts_ws_server_params, indirect=True)
    def test_input_done_flushes_without_closing_connection(self, omni_server) -> None:
        # input.done must flush the buffer and leave the connection usable:
        # the second utterance is synthesized without a second handshake.
        texts = [
            "This is the first utterance sent over the websocket connection.",
            "This is the second utterance, reusing the very same connection.",
        ]
        results = asyncio.run(_run_ws_reused_connection(omni_server.host, omni_server.port, omni_server.model, texts))

        assert len(results) == len(texts)
        for expected_index, result in enumerate(results):
            assert result["session_done"] == {
                "type": "session.done",
                "utterance_index": expected_index,
                "total_sentences": 1,
            }
            assert len(result["starts"]) == 1
            assert len(result["dones"]) == 1
            # utterance_index counts the flushes of the connection, while
            # sentence_index stays within the flush it belongs to.
            assert result["starts"][0]["utterance_index"] == expected_index
            assert result["starts"][0]["sentence_index"] == 0
            assert result["dones"][0]["error"] is False
            assert result["dones"][0]["total_bytes"] > 0
