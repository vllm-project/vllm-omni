# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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


def _session_config(model: str, *, text_input_mode: str = "buffered") -> str:
    return json.dumps(
        {
            "type": "session.config",
            "model": model,
            "voice": "vivian",
            "language": "English",
            "response_format": "pcm",
            "stream_audio": True,
            "text_input_mode": text_input_mode,
        }
    )


async def _collect_utterance(ws, *, prefix_messages: list[str | bytes] | None = None) -> dict:
    """Read frames until session.done closes out one flushed utterance."""
    starts: list[dict] = []
    dones: list[dict] = []
    chunk_lengths: dict[int, list[int]] = {}
    session_done: dict | None = None
    buffered_messages = iter(prefix_messages or [])

    while True:
        try:
            message = next(buffered_messages)
        except StopIteration:
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


async def _run_ws_commitment_session(host: str, port: int, model: str) -> tuple[dict, dict, int]:
    """Exercise incremental commitment, EOF flushing, and connection reuse."""
    uri = f"ws://{host}:{port}/v1/audio/speech/stream"

    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(_session_config(model, text_input_mode="commitment"))

        # The number is unresolved: a following unit can still change its
        # reading, so commitment mode must not begin irreversible audio yet.
        await ws.send(json.dumps({"type": "input.text", "text": "The total is 2026"}))
        try:
            unexpected = await asyncio.wait_for(ws.recv(), timeout=1)
        except asyncio.TimeoutError:
            pass
        else:
            raise AssertionError(f"Received a frame for unresolved text: {unexpected!r}")

        # Completing the unit and sentence boundary must produce audio without
        # waiting for input.done. Preserve the frames already observed so the
        # normal utterance collector can validate the complete event stream.
        await ws.send(json.dumps({"type": "input.text", "text": " dollars. "}))
        prefix_messages: list[str | bytes] = []
        saw_start = False
        first_pcm_bytes = 0
        while first_pcm_bytes == 0:
            message = await asyncio.wait_for(ws.recv(), timeout=180)
            prefix_messages.append(message)
            if isinstance(message, bytes):
                if not saw_start:
                    raise AssertionError("Received commitment PCM before audio.start")
                first_pcm_bytes = len(message)
                continue

            payload = json.loads(message)
            if payload.get("type") == "error":
                raise AssertionError(f"WebSocket error: {payload['message']}")
            if payload.get("type") != "audio.start":
                raise AssertionError(f"Unexpected pre-input.done message: {payload}")
            saw_start = True

        # No boundary follows this final fragment. input.done is EOF and must
        # release it as the second sentence while keeping the socket reusable.
        await ws.send(json.dumps({"type": "input.text", "text": "Thank you"}))
        await ws.send(json.dumps({"type": "input.done"}))
        first_result = await _collect_utterance(ws, prefix_messages=prefix_messages)

        reused_text = "The same connection still works!"
        await ws.send(json.dumps({"type": "input.text", "text": reused_text}))
        await ws.send(json.dumps({"type": "input.done"}))
        reused_result = await _collect_utterance(ws)
        await ws.send(json.dumps({"type": "session.close"}))

    return first_result, reused_result, first_pcm_bytes


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

    @pytest.mark.advanced_model
    @pytest.mark.tts
    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    @pytest.mark.parametrize("omni_server", tts_ws_server_params, indirect=True)
    def test_commitment_mode_streams_before_input_done_and_reuses_connection(self, omni_server) -> None:
        first_result, reused_result, first_pcm_bytes = asyncio.run(
            _run_ws_commitment_session(omni_server.host, omni_server.port, omni_server.model)
        )

        # This PCM frame was received after the confirmed sentence boundary
        # and before the client sent input.done.
        assert first_pcm_bytes > 0

        assert first_result["session_done"] == {
            "type": "session.done",
            "utterance_index": 0,
            "total_sentences": 2,
        }
        starts = first_result["starts"]
        dones = first_result["dones"]
        assert len(starts) == len(dones) == 2
        assert [(event["utterance_index"], event["sentence_index"]) for event in starts] == [
            (0, 0),
            (0, 1),
        ]
        assert [(event["utterance_index"], event["sentence_index"]) for event in dones] == [
            (0, 0),
            (0, 1),
        ]
        assert starts[0]["sentence_text"] == "The total is 2026 dollars."
        assert starts[1]["sentence_text"].lstrip() == "Thank you"

        for start, done in zip(starts, dones, strict=True):
            sentence_index = start["sentence_index"]
            chunk_lengths = first_result["chunk_lengths"][sentence_index]
            assert start["format"] == "pcm"
            assert start["sample_rate"] == 24000
            assert done["error"] is False
            assert chunk_lengths
            assert sum(chunk_lengths) == done["total_bytes"] > 0

        # input.done ended only the first utterance; the sticky commitment
        # session remains usable without another handshake.
        assert reused_result["session_done"] == {
            "type": "session.done",
            "utterance_index": 1,
            "total_sentences": 1,
        }
        assert len(reused_result["starts"]) == len(reused_result["dones"]) == 1
        reused_start = reused_result["starts"][0]
        reused_done = reused_result["dones"][0]
        assert reused_start["utterance_index"] == reused_done["utterance_index"] == 1
        assert reused_start["sentence_index"] == reused_done["sentence_index"] == 0
        assert reused_start["sentence_text"].strip() == "The same connection still works!"
        reused_chunks = reused_result["chunk_lengths"][0]
        assert reused_done["error"] is False
        assert reused_chunks
        assert sum(reused_chunks) == reused_done["total_bytes"] > 0
