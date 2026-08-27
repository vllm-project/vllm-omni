# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Single-sample MiniCPM-o native-duplex generation."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pybase64 as base64

from vllm_omni.experimental.fullduplex.client import RealtimeDuplexClient, build_realtime_url, wait_for

from .omni_duplex_eval_clock import extract_timed_sentences
from .omni_duplex_eval_media import iter_av_units, iter_jpegs, materialize_media, read_audio_pcm16, video_duration


def _ref_audio(path: str | Path) -> str:
    value = Path(path).expanduser().read_bytes()
    return "data:audio/wav;base64," + base64.b64encode(value).decode("ascii")


async def generate_sample(
    sample: Any,
    *,
    url: str,
    model: str,
    ref_audio: str | Path,
    output_root: str | Path,
    fps: float = 1.0,
    mix: str = "question",
    pace: str = "realtime",
    clock: str = "media",
    overwrite: bool = False,
    unit_ms: int = 1000,
) -> Path:
    output = Path(output_root) / sample.split / f"{sample.id}.json"
    meta_path = output.with_name(output.stem + ".meta.json")
    if output.exists() and not overwrite:
        return output
    if mix != "question":
        raise NotImplementedError("v1 supports mix=question; soundtrack mixing is reserved for P1")
    media_dir = output.parent / ".media"
    audio_path = materialize_media(sample.question_audio, media_dir, f"{sample.id}_question", ".wav")
    video_path = materialize_media(sample.video, media_dir, sample.id, ".mp4")
    pcm = read_audio_pcm16(audio_path)
    duration = sample.video_duration or video_duration(video_path)
    frames = iter_jpegs(video_path, fps=fps, duration=duration)
    realtime = pace == "realtime"
    if pace not in {"realtime", "as-fast-as-possible"}:
        raise ValueError("pace must be realtime or as-fast-as-possible")
    client = RealtimeDuplexClient(build_realtime_url(url, model, autostart=False))
    response_done = False
    drain_timeout = None
    close_timeout = None
    async with client:
        await client.configure(model, ref_audio=_ref_audio(ref_audio), instructions="Streaming Omni Conversation.")
        ack_task = asyncio.create_task(_ack_playback(client))
        try:
            await client.stream_av_units(iter_av_units(pcm, frames, unit_ms=unit_ms), realtime=realtime)
            await client.commit()
            try:
                await wait_for(
                    lambda: client.events.count("response.done") > 0,
                    timeout_s=max(20.0, duration + 20.0),
                    label="response.done",
                )
                response_done = True
            except TimeoutError as exc:
                drain_timeout = str(exc)
        finally:
            ack_task.cancel()
            try:
                await ack_task
            except asyncio.CancelledError:
                pass
        await client.acknowledge_playback()
        try:
            await client.close_session(timeout_s=20.0)
        except TimeoutError as exc:
            close_timeout = str(exc)
        events = list(client.events.events)
    timed = [sentence.as_dict() for sentence in extract_timed_sentences(events, clock=clock)]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(timed, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "id": sample.id,
        "split": sample.split,
        "clock": clock if realtime else "invalid",
        "pace": pace,
        "mix": mix,
        "fps": fps,
        "unit_ms": unit_ms,
        "model": model,
        "response_done": response_done,
        "drain_timeout": drain_timeout,
        "close_timeout": close_timeout,
        "ref_audio_sha256": hashlib.sha256(Path(ref_audio).read_bytes()).hexdigest(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return output


async def _ack_playback(client: RealtimeDuplexClient) -> None:
    while True:
        await asyncio.sleep(0.25)
        await client.acknowledge_playback()
