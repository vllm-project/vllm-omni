# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drive complete native duplex turns while Stage-0 repeatedly rebuilds KV."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

from vllm_omni.clients.duplex import DuplexClient, EventCollector, read_pcm16_wav, wait_for_condition
from vllm_omni.clients.minicpmo_4_5 import create_duplex_session_config


async def run_window_turn(
    *,
    url: str,
    model: str,
    input_wav: Path,
    mode: str,
    ref_audio: Path | None = None,
    repeats: int = 2,
    video_frames: list[str] | None = None,
    buffered_flush: bool = False,
    timeout_s: float = 240.0,
) -> dict[str, object]:
    """Use aggressive limits to exercise multiple windows in one audio turn."""
    pcm = read_pcm16_wav(input_wav) * repeats
    collector = EventCollector()
    reference = (
        "data:audio/wav;base64," + base64.b64encode(ref_audio.read_bytes()).decode("ascii")
        if ref_audio is not None
        else None
    )
    config = create_duplex_session_config(
        modalities=("text", "audio") if ref_audio is not None else ("text",),
        ref_audio=reference,
        temperature=0.0,
        extra_body={
            "sliding_window_mode": mode,
            "context_max_units": 1,
            "context_previous_max_tokens": 16,
            "basic_window_high_tokens": 64,
            "basic_window_low_tokens": 32,
        },
    )
    client = DuplexClient(
        url, model=model, config=config, reconnect=None, heartbeat_interval_s=None, handshake_timeout_s=timeout_s
    )
    async with client:
        reader = asyncio.create_task(collector.consume(client))
        collector.add({"type": "session.created", "session": client.session_info})
        try:
            frames_sent = await client.stream_pcm(
                pcm,
                chunk_ms=2500 if buffered_flush else 200,
                realtime=not buffered_flush,
                is_speech=True,
                video_frames=video_frames,
            )
            await client.commit(final=True)
            await wait_for_condition(
                lambda: collector.count("input_audio_buffer.committed") > 0 or bool(collector.errors()),
                timeout_s=timeout_s,
                label="final input commit",
            )
            await wait_for_condition(
                lambda: collector.count("response.done") > 0 or bool(collector.errors()),
                timeout_s=timeout_s,
                label="completed windowed response",
            )
            # Final execution follows the commit acknowledgement asynchronously.
            await asyncio.sleep(3.0)
            await client.close(timeout_s=timeout_s)
            await asyncio.wait_for(reader, timeout=5.0)
        finally:
            if not reader.done():
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
    return {
        "created": collector.count("session.created"),
        "committed": collector.count("input_audio_buffer.committed"),
        "done": collector.count("response.done"),
        "closed": collector.count("session.closed"),
        "errors": collector.errors(),
        "audio_bytes": len(collector.audio_bytes()),
        "frames_sent": frames_sent,
        "input_seconds": len(pcm) / 32000,
    }
