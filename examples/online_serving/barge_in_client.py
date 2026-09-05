# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full-duplex use case: talk over the assistant (barge-in).

Scenario (see ``barge_in_client_flow.md`` for the diagram)
---------------------------------------------------------------
1. The user asks a question (``--question-wav``); the model answers in speech.
2. Mid-answer, the user starts talking again (``--interrupt-wav``) — a
   follow-up or correction — while assistant audio is still streaming.
3. The model handles the overlap: depending on the session overlap policy and
   speech length, the in-flight response is barged in (cancelled) or the
   commit is deferred to the turn end, and a new response answers the
   interruption. Either way the user never waits for the first answer.

The demo is model-neutral: it drives any duplex model through the public
:class:`vllm_omni.clients.duplex.DuplexClient` over
``/v1/realtime?duplex=1``. Model-specific session shape comes from the
per-model presets under ``vllm_omni.clients.<model>`` selected with
``--preset`` (default ``minicpmo_4_5``, whose model decides listen/speak on
its own — no VAD).

Run the server first (for MiniCPM-o 4.5: ``bash minicpmo/barge_in_serve.sh``, which
starts the ``vllm_omni/deploy/minicpmo_4_5.yaml`` profile with ``session_mode: duplex``),
then:

    python barge_in_client.py \
        --url ws://127.0.0.1:8099/v1/realtime \
        --model openbmb/MiniCPM-o-4_5 \
        --ref-audio /path/to/reference_voice.wav \
        --question-wav question_16k.wav \
        --interrupt-wav follow_up_16k.wav \
        --output-dir ./duplex_out

Input WAVs must be mono 16 kHz PCM16; when a preset's session takes a
different capture format (the PersonaPlex preset streams 24 kHz float32),
the audio is converted before streaming. ``--ref-audio`` is required by the
MiniCPM-o preset (without it the session is rejected with
``ref_audio_required``). Outputs land in ``--output-dir`` as one WAV per
response — ``response_1_cancelled.wav`` for a barged-in answer,
``response_1_completed.wav`` for one that finished — plus ``summary.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni.clients.duplex import (  # noqa: E402
    AudioFormat,
    DuplexClient,
    EventCollector,
    SessionConfig,
    acknowledge_collected_playback,
    audio_data_url,
    read_pcm16_wav,
    wait_for_condition,
    write_pcm16_wav,
)

_TRANSCRIPT_DELTA_EVENTS = {
    "response.audio_transcript.delta",
    "response.output_text.delta",
    "response.text.delta",
}

# Input WAVs are mono 16 kHz PCM16; each preset's session may negotiate a
# different capture format (PersonaPlex takes 24 kHz float32) and a different
# model-unit-aligned chunk size (PersonaPlex frames are exactly 80 ms).
_PRESET_DEFAULT_CHUNK_MS = {"minicpmo_4_5": 100, "personaplex": 80, "none": 100}


def _convert_to_session_format(pcm16_16k: bytes, target: AudioFormat) -> bytes:
    """Convert mono 16 kHz PCM16 to the session's input format."""
    if target.encoding == "pcm16" and target.sample_rate_hz == 16_000:
        return pcm16_16k
    import numpy as np

    samples = np.frombuffer(pcm16_16k, dtype=np.int16).astype(np.float32) / 32768.0
    if target.sample_rate_hz != 16_000 and len(samples) > 1:
        target_len = round(len(samples) * target.sample_rate_hz / 16_000)
        positions = np.linspace(0.0, len(samples) - 1, num=target_len)
        samples = np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)
    if target.encoding == "pcm_f32le":
        return samples.tobytes()
    if target.encoding == "pcm16":
        return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    raise ValueError(f"Unsupported session input encoding: {target.encoding!r}")


def _session_config(preset: str, *, ref_audio_path: str | None) -> SessionConfig:
    ref_audio = audio_data_url(Path(ref_audio_path).expanduser()) if ref_audio_path else None
    if preset == "minicpmo_4_5":
        from vllm_omni.clients.minicpmo_4_5 import create_duplex_session_config

        return create_duplex_session_config(ref_audio=ref_audio)
    if preset == "personaplex":
        from vllm_omni.clients.personaplex import create_duplex_session_config

        return create_duplex_session_config()
    return SessionConfig(ref_audio=ref_audio)


def _fold_responses(collector: EventCollector) -> dict[str, dict[str, object]]:
    """Fold the event stream into per-response audio, text, and status."""
    out: dict[str, dict[str, object]] = {}
    for response_id in collector.response_ids:
        out[response_id] = {
            "audio": collector.audio_bytes(response_id),
            "text": "",
            "status": "in_progress",
            "sample_rate_hz": collector.output_sample_rate_hz,
        }

    for event in collector.events:
        etype = event.get("type")
        response_id = collector.response_id(event)
        item = out.get(response_id) if isinstance(response_id, str) else None
        if etype in _TRANSCRIPT_DELTA_EVENTS and item is not None:
            delta = event.get("delta")
            if isinstance(delta, str):
                item["text"] += delta
        elif etype == "response.done" and item is not None:
            response = event.get("response")
            status = response.get("status") if isinstance(response, dict) else None
            if item["status"] == "in_progress":
                item["status"] = "cancelled" if status == "cancelled" else "completed"
        elif etype == "audio.cancelled":
            targets = [item] if item is not None else list(out.values())
            for target in targets:
                if target["status"] == "in_progress":
                    target["status"] = "cancelled"
    return out


async def run(args: argparse.Namespace) -> int:
    config = _session_config(args.preset, ref_audio_path=args.ref_audio)
    question = _convert_to_session_format(read_pcm16_wav(Path(args.question_wav)), config.input_audio)
    interrupt = _convert_to_session_format(read_pcm16_wav(Path(args.interrupt_wav)), config.input_audio)
    chunk_ms = args.chunk_ms if args.chunk_ms is not None else _PRESET_DEFAULT_CHUNK_MS.get(args.preset, 100)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    collector = EventCollector()
    client = DuplexClient(
        args.url,
        model=args.model,
        config=config,
        reconnect=None,
        heartbeat_interval_s=None,
        handshake_timeout_s=args.timeout_s,
    )
    # GREET: open the voice session (handshake happens on enter).
    async with client:
        consume_task = asyncio.create_task(collector.consume(client))

        # ASK: stream the question at realtime pace, then commit the turn.
        # The model decides on its own to answer (no VAD).
        print("[1] streaming question ...", file=sys.stderr)
        await client.stream_pcm(question, chunk_ms=chunk_ms)
        await client.commit(create_response=False)
        await wait_for_condition(
            lambda: collector.count("response.created") > 0,
            timeout_s=args.timeout_s,
            label="first response.created",
        )
        print("[1] assistant is answering", file=sys.stderr)

        # Let the answer play for a moment, as a listening user would.
        await asyncio.sleep(args.listen_s)

        # INTERRUPT: talk over the assistant while its audio still streams.
        # The serving overlap policy decides: enough overlapped speech barges
        # in (the first response is cancelled) or a short remark is deferred
        # until the model yields the turn.
        print("[2] interrupting mid-answer ...", file=sys.stderr)
        ids_before_interrupt = set(collector.response_ids)
        await client.stream_pcm(interrupt, chunk_ms=chunk_ms)
        await client.commit(create_response=False)

        # ANSWER AGAIN: wait for a response created *after* the interruption
        # to finish. The first answer may still be streaming here; matching on
        # any response.done would let its completion satisfy the wait and cut
        # off the actual follow-up answer at close.
        def follow_up_answer_done() -> bool:
            return any(
                event.get("type") == "response.done"
                and collector.response_id(event) is not None
                and collector.response_id(event) not in ids_before_interrupt
                for event in collector.events
            )

        await wait_for_condition(
            follow_up_answer_done,
            timeout_s=args.timeout_s,
            label="response.done for the response answering the interruption",
        )

        # HANG UP: report playback, close the session cleanly.
        await acknowledge_collected_playback(client, collector)
        await client.close(timeout_s=args.timeout_s)
        try:
            await asyncio.wait_for(consume_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            consume_task.cancel()

    # Report: one WAV per response; statuses show what the overlap did.
    responses = _fold_responses(collector)
    summary: list[dict[str, object]] = []
    for index, (response_id, item) in enumerate(responses.items(), start=1):
        audio = item["audio"]
        assert isinstance(audio, bytes)
        sample_rate_hz = int(item["sample_rate_hz"])  # type: ignore[arg-type]
        wav_path = output_dir / f"response_{index}_{item['status']}.wav"
        if audio:
            write_pcm16_wav(wav_path, audio, sample_rate_hz=sample_rate_hz)
        summary.append(
            {
                "response_id": response_id,
                "status": item["status"],
                "audio_s": round(len(audio) / (sample_rate_hz * 2), 2),
                "text": item["text"],
                "wav": wav_path.name if audio else None,
            }
        )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (output_dir / "events.jsonl").write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in collector.events),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    cancelled = sum(1 for item in summary if item["status"] == "cancelled")
    print(
        f"\n{len(summary)} responses ({cancelled} interrupted); outputs in {output_dir}",
        file=sys.stderr,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="ws://127.0.0.1:8099/v1/realtime")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument(
        "--preset",
        choices=["minicpmo_4_5", "personaplex", "none"],
        default="minicpmo_4_5",
        help="per-model session preset from vllm_omni.clients.<model>",
    )
    parser.add_argument("--ref-audio", help="reference voice WAV (the MiniCPM-o preset requires it)")
    parser.add_argument("--question-wav", required=True, help="mono 16 kHz PCM16 question")
    parser.add_argument("--interrupt-wav", required=True, help="mono 16 kHz PCM16 follow-up spoken mid-answer")
    parser.add_argument("--output-dir", default="./duplex_out")
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=None,
        help="append chunk size; defaults to the preset's model-unit-aligned size (minicpmo_4_5: 100, personaplex: 80)",
    )
    parser.add_argument("--listen-s", type=float, default=2.0, help="how long to listen before interrupting")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
