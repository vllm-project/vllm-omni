# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PersonaPlex full-duplex server, compatible with the OFFICIAL Moshi web client.

Serves the official PersonaPlex web UI (``dist.tgz`` from ``nvidia/personaplex-7b-v1``,
the same bundle ``moshi.server`` downloads) and implements its WebSocket protocol at
``/api/chat`` -- but driven by our native-component engine instead of moshi's server.

Built on **aiohttp**, mirroring ``moshi.server`` exactly: aiohttp writes+drains each
WebSocket frame promptly, whereas uvicorn's ``websockets`` backend coalesces many small
(~80 ms Opus) frames into a send buffer, which makes browser playback choppy even when
the engine is real-time. aiohttp ships as a moshi dependency, so this adds nothing new.

Moshi protocol (binary WS messages, first byte = tag):
  server -> client  ``\\x00``                 handshake (ready, after system prompts)
  client -> server  ``\\x01`` + opus bytes    mic audio (Opus @ 24kHz)
  server -> client  ``\\x01`` + opus bytes    agent audio (Opus @ 24kHz)
  server -> client  ``\\x02`` + utf8          inner-monologue text
Query params on connect: ``text_prompt`` (persona), ``voice_prompt`` (voice file).

Run (auto-downloads the official UI; open http://<host>:8124/):
    HF_TOKEN=... CUDA_VISIBLE_DEVICES=0 python -m \\
        vllm_omni.experimental.fullduplex.personaplex.serving.server --port 8124

A raw-PCM endpoint (``/v1/audio/duplex``, JSON ``open`` + float32 frames) remains for
tooling/tests that do not want an Opus dependency.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import json
import logging
import tarfile
import threading
from pathlib import Path

import numpy as np
import sphn
from aiohttp import WSMsgType, web
from huggingface_hub import hf_hub_download

from vllm_omni.experimental.fullduplex.personaplex.config import PersonaPlexConfig
from vllm_omni.experimental.fullduplex.personaplex.engine import PersonaPlexEngine
from vllm_omni.experimental.fullduplex.personaplex.serving.batched import BatchedSessionManager
from vllm_omni.experimental.fullduplex.personaplex.session import (
    PersonaPlexServingSessionState,
    PersonaPlexSession,
    offer_realtime,
)

logger = logging.getLogger(__name__)
_HF_REPO = "nvidia/personaplex-7b-v1"


def _opus_encode(writer: sphn.OpusStreamWriter, pcm: np.ndarray) -> bytes:
    """Encode one PCM frame to Opus bytes, across sphn API versions.

    sphn 0.2.x returns the encoded bytes straight from ``append_pcm``; the older
    0.1.x buffers and requires a ``read_bytes`` drain. Handle both.
    """
    out = writer.append_pcm(np.ascontiguousarray(pcm, dtype=np.float32))
    if out is None and hasattr(writer, "read_bytes"):
        out = writer.read_bytes()
    return out or b""


def _opus_decode(reader: sphn.OpusStreamReader, data: bytes) -> np.ndarray | None:
    """Decode Opus bytes to PCM, across sphn API versions (see ``_opus_encode``)."""
    out = reader.append_bytes(data)
    if out is None and hasattr(reader, "read_pcm"):
        out = reader.read_pcm()
    if out is None or getattr(out, "shape", (0,))[-1] == 0:
        return None
    return np.asarray(out, dtype=np.float32)


def _official_web_dir() -> Path | None:
    """Download + extract the official PersonaPlex web client (dist.tgz)."""
    try:
        tgz = Path(hf_hub_download(_HF_REPO, "dist.tgz"))
        dist = tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(tgz, "r:gz") as tar:
                tar.extractall(path=tgz.parent)  # noqa: S202 - trusted first-party bundle
        return dist if dist.exists() else None
    except Exception as exc:  # network / auth / layout
        logger.warning("official web client unavailable (%s); / will return an error note", exc)
        return None


class DuplexServer:
    """Owns the loaded engine; serves one duplex conversation at a time.

    A capacity-one lease covers the entire WebSocket lifetime on BOTH endpoints:
    the engine is a single shared streaming state, so a second concurrent
    connection would reset and corrupt the live conversation. Extra connections
    are rejected with 1013 (try again later). Engine work (``open``/``feed``,
    synchronous CUDA) runs on a worker thread so the event loop stays free for
    WebSocket traffic and close handling.
    """

    def __init__(self, config: PersonaPlexConfig) -> None:
        self.config = config
        self.engine = PersonaPlexEngine(config)
        self._active = False  # capacity-one session lease (event-loop confined)
        self._lease_release_tasks: set[asyncio.Task] = set()

    def load(self) -> None:
        self.engine.load()

    async def _lease(self, ws: web.WebSocketResponse) -> bool:
        """Try to claim the single-session lease; reject the connection if held."""
        if self._active:
            await ws.close(code=1013, message=b"another conversation is active")
            return False
        self._active = True
        return True

    async def _engine_call(self, state: PersonaPlexServingSessionState, fn, *args, **kwargs):
        """Run one synchronous engine call on a worker thread, tracked for drain.

        Cancelling the awaiting task does NOT stop the worker thread, and it
        marks the asyncio wrapper future cancelled (``done()``) even while the
        thread keeps running, so the future itself cannot signal completion.
        The worker instead sets a plain ``threading.Event`` when the call truly
        finishes; :meth:`_drain_inflight` waits on that event before the lease
        is released. Without this, a fast reconnect could acquire the lease and
        call ``open()`` while the previous connection's final ``step()`` is
        still mutating the shared engine on its thread.
        """
        call = functools.partial(fn, *args, **kwargs)
        finished = threading.Event()

        def run_and_signal():
            try:
                return call()
            finally:
                finished.set()

        state.inflight = finished
        result = await asyncio.get_running_loop().run_in_executor(None, run_and_signal)
        state.inflight = None  # on cancellation, the event stays referenced for drain
        return result

    @staticmethod
    async def _drain_inflight(state: PersonaPlexServingSessionState, timeout: float = 5.0) -> bool:
        """Wait out any engine call still running on a worker thread."""
        finished = state.inflight
        if finished is None or finished.is_set():
            return True
        return bool(await asyncio.to_thread(finished.wait, timeout))

    async def _release_lease_when_finished(self, state: PersonaPlexServingSessionState) -> None:
        finished = state.inflight
        if finished is not None and not finished.is_set():
            await asyncio.to_thread(finished.wait)
        self._active = False

    async def _release_lease_after_drain(
        self,
        state: PersonaPlexServingSessionState,
        *,
        timeout: float = 5.0,
    ) -> bool:
        """Release now when drained, otherwise retain the lease until completion."""
        if await self._drain_inflight(state, timeout):
            self._active = False
            return True

        logger.error(
            "PersonaPlex engine cleanup exceeded %.1fs; retaining the session lease until the worker thread finishes",
            timeout,
        )
        task = asyncio.create_task(self._release_lease_when_finished(state))
        self._lease_release_tasks.add(task)
        task.add_done_callback(self._lease_release_tasks.discard)
        return False

    # ---- official Moshi protocol (Opus over aiohttp WS) ---------------------

    async def handle_chat(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        if not await self._lease(ws):
            return ws
        state = PersonaPlexServingSessionState()
        try:
            persona = (request.query.get("text_prompt") or "").strip() or None
            voice = (request.query.get("voice_prompt") or "").strip() or None
            sr = self.engine.sample_rate
            session = PersonaPlexSession(self.engine, self.config)
            # System-prompt prefill is seconds of GPU work; keep it off the loop.
            await self._engine_call(state, session.open, voice_prompt=voice, persona=persona)

            reader = sphn.OpusStreamReader(sr)
            writer = sphn.OpusStreamWriter(sr)
            await ws.send_bytes(b"\x00")  # ready handshake

            async def recv_loop() -> None:
                try:
                    async for msg in ws:
                        if msg.type == WSMsgType.BINARY:
                            data = msg.data
                            if data and data[0] == 1:
                                pcm = _opus_decode(reader, data[1:])
                                if pcm is not None:
                                    offer_realtime(state.pcm_queue, pcm)
                        elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.ERROR):
                            break
                finally:
                    state.closed = True

            async def proc_loop() -> None:
                # Drain decoded PCM and step the model, streaming agent Opus + text
                # out. The synchronous engine step runs on a worker thread.
                while not state.closed:
                    try:
                        pcm = await asyncio.wait_for(state.pcm_queue.get(), timeout=0.25)
                    except asyncio.TimeoutError:
                        continue
                    for frame_out in await self._engine_call(state, session.feed, pcm):
                        if frame_out.audio is not None:
                            opus = _opus_encode(writer, frame_out.audio)
                            if opus:
                                await ws.send_bytes(b"\x01" + opus)
                        if frame_out.text:
                            await ws.send_bytes(b"\x02" + frame_out.text.encode("utf8"))

            tasks = [asyncio.create_task(recv_loop()), asyncio.create_task(proc_loop())]
            try:
                _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await t
            finally:
                with contextlib.suppress(Exception):
                    await ws.close()
        finally:
            # A cancelled feed/open may still be running on its worker thread;
            # wait it out BEFORE freeing the lease, or a fast reconnect races
            # the shared engine (CUDA illegal access / cross-session bleed).
            await self._release_lease_after_drain(state)
        return ws

    # ---- simple raw-PCM protocol (no Opus, for tests) ----------------------

    async def handle_raw(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        if not await self._lease(ws):
            return ws
        session = PersonaPlexSession(self.engine, self.config)
        state = PersonaPlexServingSessionState()
        opened = False
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    req = json.loads(msg.data)
                    if req.get("type") == "open":
                        await self._engine_call(
                            state, session.open, voice_prompt=req.get("voice"), persona=req.get("persona")
                        )
                        opened = True
                        await ws.send_json({"type": "ready"})
                    elif req.get("type") == "close":
                        if opened:
                            await self._emit(ws, await self._engine_call(state, session.flush))
                        await ws.send_json({"type": "done"})
                        break
                elif msg.type == WSMsgType.BINARY:
                    if not opened:
                        await self._engine_call(state, session.open)
                        opened = True
                    pcm = np.frombuffer(msg.data, dtype=np.float32)
                    await self._emit(ws, await self._engine_call(state, session.feed, pcm))
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.ERROR):
                    break
        finally:
            # Same as handle_chat: wait out any worker-thread engine call
            # before freeing the lease (see _engine_call).
            await self._release_lease_after_drain(state)
            with contextlib.suppress(Exception):
                await ws.close()
        return ws

    @staticmethod
    async def _emit(ws: web.WebSocketResponse, outputs: list) -> None:
        for frame_out in outputs:
            if frame_out.audio is not None:
                await ws.send_bytes(np.ascontiguousarray(frame_out.audio, dtype=np.float32).tobytes())
            if frame_out.text:
                await ws.send_json({"type": "text", "text": frame_out.text})


class BatchedDuplexServer:
    """Serve ``batch_size`` concurrent duplex conversations on one engine.

    Same wire protocols as :class:`DuplexServer`; each connection claims a slot
    from the shared :class:`BatchedSessionManager`, whose stepping thread ticks
    all live slots in lockstep at 12.5 Hz wall clock. A connection with a custom
    ``text_prompt`` (or reusing a previously used slot) waits while the slot's
    system prompt is replayed per-row (voice embeddings + persona), then gets the
    ready handshake. Per-connection custom voices are not supported in batched
    mode (fixed default voice; ``voice_prompt`` is ignored with a warning).
    """

    def __init__(self, config: PersonaPlexConfig) -> None:
        self.config = config
        self.engine = PersonaPlexEngine(config)
        self.manager: BatchedSessionManager | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def load(self) -> None:
        self.engine.load()
        self.engine.open_batch()
        self.manager = BatchedSessionManager(self.engine)
        self._thread = threading.Thread(
            target=self.manager.run_forever, args=(self._stop,), daemon=True, name="pplex-tick"
        )
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    async def _acquire(self, ws: web.WebSocketResponse, persona: str | None):
        """Claim a slot and wait until it is live.

        Returns ``PersonaPlexServingSessionState`` (slot + bounded output queue)
        or ``None`` when the server is full. Exception-safe: if this coroutine
        is cancelled (client gone mid-prefill) or the wait fails, the slot is
        released instead of leaking in the PREFILL/LIVE phase.
        """
        acquired = self.manager.acquire(persona=persona)
        if acquired is None:
            await ws.close(code=1013, message=b"all conversation slots busy")
            return None
        slot, ready_now = acquired
        try:
            loop = asyncio.get_running_loop()
            state = PersonaPlexServingSessionState(slot=slot, epoch=slot.epoch)
            ready = asyncio.Event()
            with self.manager.lock:
                # Bounded handoff: the tick thread produces one frame per 80 ms;
                # a client that stops reading gets drop-oldest, not unbounded RAM.
                slot.on_output = lambda frame_out: loop.call_soon_threadsafe(offer_realtime, state.out_queue, frame_out)
                if ready_now:
                    ready.set()
                else:
                    slot.on_ready = lambda: loop.call_soon_threadsafe(ready.set)
            await ready.wait()
        except BaseException:
            self.manager.release(slot)
            raise
        return state

    async def handle_chat(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        persona = (request.query.get("text_prompt") or "").strip() or None
        if (request.query.get("voice_prompt") or "").strip():
            logger.warning("batched mode ignores per-connection voice_prompt (fixed default voice)")
        state = await self._acquire(ws, persona)
        if state is None:
            return ws
        try:
            sr = self.engine.sample_rate
            reader = sphn.OpusStreamReader(sr)
            writer = sphn.OpusStreamWriter(sr)
            await ws.send_bytes(b"\x00")  # ready handshake

            async def recv_loop() -> None:
                try:
                    async for msg in ws:
                        if msg.type == WSMsgType.BINARY:
                            data = msg.data
                            if data and data[0] == 1:
                                pcm = _opus_decode(reader, data[1:])
                                if pcm is not None:
                                    self.manager.feed(state.slot, pcm, state.epoch)
                        elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.ERROR):
                            break
                finally:
                    state.closed = True

            async def out_loop() -> None:
                while not state.closed:
                    try:
                        frame_out = await asyncio.wait_for(state.out_queue.get(), timeout=0.25)
                    except asyncio.TimeoutError:
                        continue
                    if frame_out.audio is not None:
                        msg = _opus_encode(writer, frame_out.audio)
                        if msg:
                            await ws.send_bytes(b"\x01" + msg)
                    if frame_out.text:
                        await ws.send_bytes(b"\x02" + frame_out.text.encode("utf8"))

            tasks = [asyncio.create_task(recv_loop()), asyncio.create_task(out_loop())]
            _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t
        finally:
            self.manager.release(state.slot)
            with contextlib.suppress(Exception):
                await ws.close()
        return ws

    async def handle_raw(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        state: PersonaPlexServingSessionState | None = None
        sender: asyncio.Task | None = None

        async def pump(outq: asyncio.Queue) -> None:
            # Forward slot output continuously (the tick loop produces in realtime).
            while True:
                frame_out = await outq.get()
                if frame_out.audio is not None:
                    await ws.send_bytes(np.ascontiguousarray(frame_out.audio, dtype=np.float32).tobytes())
                if frame_out.text:
                    await ws.send_json({"type": "text", "text": frame_out.text})

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    req = json.loads(msg.data)
                    if req.get("type") == "open":
                        if state is not None:
                            # One slot lease per connection; a second `open` would
                            # silently leak the first slot and its pump.
                            await ws.send_json({"type": "error", "error": "session already open"})
                            continue
                        state = await self._acquire(ws, (req.get("persona") or "").strip() or None)
                        if state is None:
                            return ws
                        sender = asyncio.create_task(pump(state.out_queue))
                        await ws.send_json({"type": "ready"})
                    elif req.get("type") == "close":
                        await asyncio.sleep(0.3)  # let in-flight frames flush via pump
                        await ws.send_json({"type": "done"})
                        break
                elif msg.type == WSMsgType.BINARY and state is not None:
                    self.manager.feed(state.slot, np.frombuffer(msg.data, dtype=np.float32), state.epoch)
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.ERROR):
                    break
        finally:
            # Release FIRST and unconditionally: a pump that already failed (e.g.
            # peer reset mid-send) re-raises on await and must not leak the slot.
            if state is not None:
                self.manager.release(state.slot)
            if sender is not None:
                sender.cancel()
                try:
                    await sender
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.debug("raw output pump ended with an error", exc_info=True)
            with contextlib.suppress(Exception):
                await ws.close()
        return ws


def create_app(config: PersonaPlexConfig) -> web.Application:
    server = BatchedDuplexServer(config) if config.batch_size > 1 else DuplexServer(config)
    web_dir = _official_web_dir()

    async def _startup(_app: web.Application) -> None:
        logger.info("PersonaPlex duplex server: loading engine (batch_size=%d)", config.batch_size)
        server.load()
        logger.info("PersonaPlex duplex server: ready (official web: %s)", bool(web_dir))

    async def _cleanup(_app: web.Application) -> None:
        if isinstance(server, BatchedDuplexServer):
            server.shutdown()

    app = web.Application()
    app.on_startup.append(_startup)
    app.on_cleanup.append(_cleanup)

    async def health(_request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    app.router.add_get("/health", health)
    app.router.add_get("/api/chat", server.handle_chat)
    app.router.add_get("/v1/audio/duplex", server.handle_raw)

    # Serve the official PersonaPlex web client (explicit routes are matched first).
    if web_dir is not None:

        async def index(_request: web.Request) -> web.FileResponse:
            return web.FileResponse(web_dir / "index.html")

        app.router.add_get("/", index)
        app.router.add_static("/", path=str(web_dir), follow_symlinks=True, name="web")
    else:

        async def index_missing(_request: web.Request) -> web.Response:
            return web.json_response({"error": f"official web client unavailable; check access to {_HF_REPO}"})

        app.router.add_get("/", index_missing)

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8124)
    ap.add_argument("--persona", default=None)
    ap.add_argument("--voice", default="NATF2.pt")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Concurrent conversation slots on one engine (elastic batching when > 1).",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = PersonaPlexConfig(
        voice_prompt=args.voice,
        batch_size=args.batch_size,
        **({"persona": args.persona} if args.persona else {}),
    )
    web.run_app(create_app(cfg), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
