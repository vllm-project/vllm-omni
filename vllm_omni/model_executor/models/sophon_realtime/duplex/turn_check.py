# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""How the session runs the turn question (see ``core/turn_question.py``) during a conversation.

The question has to run while the user is speaking, sometimes while a reply is playing, and its latency gates the
reply, so it cannot wait for the session's own stage-0 request. The framework has no path yet for a short, private,
non-resumable stage-0 request from a plugin (vllm-project/vllm-omni#10406, item 4). Until it does, this module is the
only place that knows how the question is run:

* ``TurnChecker`` is the interface the session uses: ``(audio, lang, said) -> (P(finished), P(back-channel))``.
* ``LoopbackTurnChecker`` is a stopgap that sends the question to this server's own ``/v1/chat/completions``
  (text out, one token, logprobs). It will be replaced by the framework path once that exists.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import threading
import time
import urllib.error
import urllib.request
import wave
from typing import Protocol

import numpy as np
import pybase64 as base64
from vllm.logger import init_logger

from vllm_omni.model_executor.models.sophon_realtime.core.turn_question import (
    BACKCHANNEL,
    FINISHED,
    answer_probabilities,
    turn_question,
)

from .env import api_key as _api_key
from .env import port as _port

logger = init_logger(__name__)
SR = 16000
_REQUEST_TIMEOUT_S = 10


class TurnChecker(Protocol):
    def __call__(self, audio: np.ndarray, lang: str, said: str) -> tuple[float, float]:
        """P(finished), P(back-channel) for the user's utterance ``audio`` (16 kHz float32); ``said`` is the
        assistant's last words. Raises on failure (the session then waits for the next checkpoint or the fallback)."""
        ...


class LoopbackTurnChecker:
    """Stopgap (#10406 item 4): ask this server's own ``/v1/chat/completions``."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = base_url or "http://127.0.0.1:" + _port()
        self.api_key = api_key if api_key is not None else _api_key()
        self._model_id: str | None = None

    def auth_headers(self) -> dict[str, str]:
        """``Authorization`` for this server's own endpoints when it was started with ``--api-key``."""
        return {"Authorization": "Bearer " + self.api_key} if self.api_key else {}

    def _get(self, path: str, timeout: float):
        return urllib.request.urlopen(
            urllib.request.Request(self.base_url + path, headers=self.auth_headers()), timeout=timeout
        )

    def _model(self) -> str:
        if self._model_id is None:
            with self._get("/v1/models", 5) as r:
                self._model_id = json.load(r)["data"][0]["id"]
        return self._model_id

    def __call__(self, audio: np.ndarray, lang: str, said: str) -> tuple[float, float]:
        bio = io.BytesIO()
        with wave.open(bio, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        text, order = turn_question(lang, said)
        body = {
            "model": self._model(),
            "modalities": ["text"],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 20,
            "sampling_params_list": [{"temperature": 0.0, "max_tokens": 1, "logprobs": 20}],
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": base64.b64encode(bio.getvalue()).decode(), "format": "wav"},
                        },
                    ],
                }
            ],
        }
        req = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", **self.auth_headers()},
        )
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as r:
            j = json.load(r)
        top = {t["token"]: t["logprob"] for t in j["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}
        probs = answer_probabilities(top, order)
        return probs[FINISHED], probs[BACKCHANNEL]

    def wait_ready(self, timeout_s: float = 3600) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                with self._get("/v1/models", 5) as r:
                    if r.status == 200:
                        return True
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    logger.warning(
                        "sophon duplex: own endpoint refused the API key (%d); turn checks cannot run", exc.code
                    )
                    return False
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1)
        return False


_WARM_STARTED = False


def start_warmup(checker: TurnChecker) -> None:
    """Once per process, after the server answers: one turn question and one short realtime session, so the first
    real session does not pay the cold-start cost of either path."""
    global _WARM_STARTED
    if _WARM_STARTED or not isinstance(checker, LoopbackTurnChecker):
        return
    _WARM_STARTED = True
    threading.Thread(target=_warm, args=(checker,), daemon=True, name="sophon-warmup").start()


def _warm(checker: LoopbackTurnChecker) -> None:
    if not checker.wait_ready():
        return
    for i in range(3):
        try:
            t = time.monotonic()
            checker(np.zeros(SR, np.float32), "en", "")
            logger.info("sophon duplex: turn check warm-up %d done in %.2fs", i, time.monotonic() - t)
        except Exception as exc:  # noqa: BLE001
            logger.warning("sophon duplex: turn check warm-up %d failed: %r", i, exc)
    for i in range(2):
        try:
            t = time.monotonic()
            ok = _warm_realtime(checker.base_url, checker.auth_headers())
            logger.info(
                "sophon duplex: realtime warm-up %d %s in %.1fs",
                i,
                "got a reply" if ok else "got no reply",
                time.monotonic() - t,
            )
            if ok:
                break
        except Exception as exc:  # noqa: BLE001
            logger.warning("sophon duplex: realtime warm-up %d failed: %r", i, exc)


def _ws_connect(url: str, headers: dict[str, str]):
    """websockets.connect with request headers (the keyword changed name in websockets 14)."""
    import inspect

    import websockets

    kw = (
        "additional_headers"
        if "additional_headers" in inspect.signature(websockets.connect).parameters
        else "extra_headers"
    )
    return websockets.connect(url, max_size=None, **({kw: headers} if headers else {}))


def _warm_realtime(base_url: str, headers: dict[str, str] | None = None) -> bool:
    """One short realtime session on this server (bundled clip, then silence) until a reply has finished."""
    path = os.path.join(os.path.dirname(__file__), "warmup.wav")
    with wave.open(path, "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    pcm = np.concatenate([np.zeros(SR // 2, np.float32), pcm, np.zeros(SR * 25, np.float32)]).astype("<f4")
    url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/v1/realtime"

    async def run() -> bool:
        async with _ws_connect(url, headers or {}) as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "modalities": ["audio", "text"],
                            "input_audio_format": "pcm_f32le",
                            "output_audio_format": "pcm16",
                            "turn_detection": {"type": "server_vad"},
                            "extra_body": {"auto_response": True, "language": "zh"},
                        },
                    }
                )
            )
            done = asyncio.Event()

            async def read() -> None:
                async for m in ws:
                    try:
                        t = json.loads(m).get("type", "")
                    except Exception:  # noqa: BLE001
                        continue
                    if t == "response.done":
                        done.set()
                        return

            task = asyncio.ensure_future(read())
            n = SR * 80 // 1000
            for k in range(0, pcm.size, n):
                if done.is_set():
                    break
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(pcm[k : k + n].tobytes()).decode(),
                            "format": "pcm_f32le",
                            "sample_rate_hz": SR,
                        }
                    )
                )
                await asyncio.sleep(0.08)
            ok = done.is_set()
            task.cancel()
            try:
                await ws.send(json.dumps({"type": "session.close"}))
            except Exception:  # noqa: BLE001
                pass
            return ok

    return asyncio.run(run())
