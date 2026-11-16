# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""JoyAI long-session lifecycle acceptance on real model stages.

The continuous test runs one twelve-minute cycle by default. Set
JOYAI_SOAK_CYCLES to a positive integer to select more cycles. For a one-hour
soak, run from the repository root with the model assets available::

    JOYAI_SOAK_CYCLES=5 PYTHONPATH=. python -m pytest -o addopts='' \
        --run-level=advanced_model -m local_model -k twelve_minute \
        tests/e2e/online_serving/test_joyai_vl_interaction_expansion.py

Each cycle includes cancellation and same-session recovery. All five cycles
share one session; close and fresh-session admission are checked at the end.
"""

import asyncio
import io
import json
import os
import time
from contextlib import AsyncExitStack

import pybase64 as base64
import pytest
from PIL import Image, ImageDraw
from websockets.asyncio.client import connect

from tests.e2e.online_serving.helpers.joyai_vl_interaction import joyai_server_params
from tests.helpers.mark import hardware_test
from vllm_omni.clients.duplex import DuplexClient, EventCollector, ReconnectPolicy, SessionConfig

pytestmark = [pytest.mark.local_model, pytest.mark.omni]

# Validate at collection, before starting the model server fixture.
try:
    SOAK_CYCLES = int(os.environ.get("JOYAI_SOAK_CYCLES", "1"))
except ValueError as exc:
    raise pytest.UsageError("JOYAI_SOAK_CYCLES must be a positive integer") from exc
if SOAK_CYCLES < 1:
    raise pytest.UsageError("JOYAI_SOAK_CYCLES must be a positive integer")

CONFIG = SessionConfig(
    auto_response=False,
    temperature=0.0,
    instructions="For every new image, say only its dominant color in English.",
    extra_body={"tts_language": "English"},
)


def encode_frame(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


async def query(client, collector, color):
    done = asyncio.create_task(client.wait_for("response.done", timeout_s=180))
    await asyncio.sleep(0)
    try:
        await client.send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "What color is the latest image?"}],
                },
            }
        )
        await client.append_audio(b"", video_frames=[encode_frame(Image.new("RGB", (224, 224), color))])
        await client.commit(create_response=True)
        terminal = await done
        await asyncio.sleep(0)
        assert terminal.raw["response"]["status"] == "completed", terminal.raw
        text = collector.response_text(terminal.response_id)
        assert color in text.lower(), (color, text)
        pcm = collector.audio_bytes(terminal.response_id)
        assert len(pcm) >= 2400
        await client.ack_playback(len(pcm) * 500 / collector.output_sample_rate_hz, response_id=terminal.response_id)
        return {
            "session_id": client.session_id,
            "response_id": terminal.response_id,
            "color": color,
            "text": text,
            "audio_bytes": len(pcm),
        }
    finally:
        done.cancel()
        await asyncio.gather(done, return_exceptions=True)


async def run_limits(url, model, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    stack = AsyncExitStack()
    queries: list[dict] = []
    result = {"passed": False, "queries": queries}
    started = time.monotonic()
    clients, readers, collectors = [], [], []
    try:
        for _ in range(2):
            client = DuplexClient(url, model=model, config=CONFIG, reconnect=None)
            await stack.enter_async_context(client)
            collector = EventCollector()
            clients.append(client)
            collectors.append(collector)
            readers.append(asyncio.create_task(collector.consume(client)))
        oversized = encode_frame(Image.effect_noise((1024, 1024), 100).convert("RGB"))
        assert len(oversized) > 128 * 1024
        waiter = asyncio.create_task(clients[0].wait_for("error", timeout_s=30))
        await asyncio.sleep(0)
        try:
            await clients[0].append_audio(b"", video_frames=[oversized])
            error = await waiter
            assert error.raw["error"]["code"] == "input_backpressure", error.raw
            result["input_backpressure"] = error.raw
        finally:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
        queries.extend(
            await asyncio.gather(query(clients[0], collectors[0], "red"), query(clients[1], collectors[1], "blue"))
        )
        assert set(collectors[0].response_ids).isdisjoint(collectors[1].response_ids)
        assert len(collectors[0].errors()) == 1 and not collectors[1].errors()
        first = clients[0]
        sid, token = first.session_id, first.resume_token
        await clients[1].close()
        await first._ws.close()
        await asyncio.wait_for(first._reader_task, 10)
        # Production defaults: 30-second grace plus a 5-second reaper interval.
        await asyncio.sleep(40)
        async with connect(first._target_url()) as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "session.resume",
                        "session_id": sid,
                        "resume_token": token,
                        "last_received_server_event_seq": 0,
                    }
                )
            )
            while True:
                event = json.loads(await asyncio.wait_for(ws.recv(), 20))
                if event.get("type") == "error":
                    result["expired_resume_error"] = event
                    assert event["error"]["code"] == "session_resume_expired", event
                    break
                assert event.get("type") != "session.resumed", event
        # Fill both admission slots again: a leaked detached session would refuse one.
        for _ in range(2):
            client = DuplexClient(url, model=model, config=CONFIG, reconnect=None)
            await stack.enter_async_context(client)
            collector = EventCollector()
            clients.append(client)
            collectors.append(collector)
            readers.append(asyncio.create_task(collector.consume(client)))
        queries.extend(
            await asyncio.gather(query(clients[2], collectors[2], "green"), query(clients[3], collectors[3], "yellow"))
        )
        result["passed"] = True
    finally:
        await stack.aclose()
        for reader in readers:
            reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
        for index, collector in enumerate(collectors):
            (output_dir / f"events-{index}.json").write_text(json.dumps(collector.events))
        result["elapsed_s"] = time.monotonic() - started
        (output_dir / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    return result["passed"]


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=1)
@pytest.mark.parametrize(
    "omni_server",
    [
        joyai_server_params(
            session_limits={
                "max_sessions": 2,
                "max_pending_input_bytes_per_session": 128 * 1024,
            }
        )
    ],
    indirect=True,
)
def test_backpressure_recovery_isolation_and_disconnect_expiry(omni_server, tmp_path):
    assert asyncio.run(
        run_limits(
            f"ws://{omni_server.host}:{omni_server.port}/v1/realtime?duplex=1",
            omni_server.model,
            tmp_path,
        )
    )


def observation_frame(index):
    """A changing, timestamped camera scene; the cycle repeats after six minutes."""
    colors = ("blue", "red", "green", "orange", "purple", "cyan")
    image = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(image)
    color = colors[(index // 60) % len(colors)]
    if index < 3:
        draw.rectangle((40, 180, 472, 330), fill="yellow", outline="black", width=4)
        draw.text((95, 235), "MANGO 73", fill="black", font_size=50)
    elif (index // 60) % 2:
        draw.rectangle((130, 130, 382, 382), fill=color)
    else:
        draw.ellipse((130, 130, 382, 382), fill=color)
    draw.text((20, 20), f"FRAME {index:04d}  TIME {index}s", fill="black", font_size=24)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


async def run_continuous(url, model, output_dir, *, cycles=1):
    """Run twelve-minute cycles with cancellation and same-session recovery."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config = SessionConfig(
        auto_response=False,
        temperature=0.0,
        instructions="Observe the video silently. Answer explicit questions aloud in English.",
        extra_body={"tts_language": "English"},
    )
    result: dict = {
        "passed": False,
        "sent": 0,
        "frames": 0,
        "errors": [],
        "interactions": [],
        "action_pauses_s": [],
        "max_outstanding": 0,
        "committed": 0,
        "vision_follow": 0,
        "max_pacing_lag_s": 0.0,
    }
    created: set[str] = set()
    finished: set[str] = set()
    cancelled: set[str] = set()
    texts: dict[str, str] = {}
    audio: dict[str, int] = {}
    started = time.monotonic()
    last_progress = started
    stream = (output_dir / "events.jsonl").open("w", buffering=1)

    async def consume(client):
        nonlocal last_progress
        output_sample_rate_hz = config.output_audio.sample_rate_hz
        async for event in client:
            stream.write(json.dumps({"elapsed_s": time.monotonic() - started, "event": event.raw}) + "\n")
            rid = event.response_id
            if event.type == "input_audio_buffer.committed":
                result["committed"] += 1
                last_progress = time.monotonic()
            if event.type == "overlap.decision" and event.raw.get("reason") == "vision_follow":
                result["vision_follow"] += 1
            if event.type == "error":
                result["errors"].append(event.raw)
            if event.type in (
                "response.output_audio.delta",
                "response.output_audio_transcript.delta",
                "response.output_text.delta",
            ):
                if rid in finished:
                    result["errors"].append({"late_delta": event.type, "response_id": rid})
                if event.type == "response.output_audio.delta":
                    output_sample_rate_hz = event.sample_rate_hz or output_sample_rate_hz
                    audio[rid] = audio.get(rid, 0) + len(base64.b64decode(event.raw["delta"]))
                else:
                    texts[rid] = texts.get(rid, "") + event.raw.get("delta", "")
            if event.type == "response.created":
                created.add(rid)
                last_progress = time.monotonic()
            if event.type == "response.done":
                finished.add(rid)
                last_progress = time.monotonic()
                status = event.raw["response"]["status"]
                if status == "cancelled":
                    cancelled.add(rid)
                elif status != "completed":
                    result["errors"].append(event.raw)
                if status == "completed" and audio.get(rid, 0):
                    # The sink consumes spontaneous speech as well as explicit answers.
                    # Otherwise unacknowledged playback keeps vision-follow active.
                    await client.ack_playback(audio[rid] * 500 / output_sample_rate_hz, response_id=rid)
            if event.type == "session.closed":
                result["closed"] = True

    def check(reader):
        if reader.done():
            reader.result()
            raise RuntimeError("Reader stopped during input")
        assert not result["errors"], result["errors"]

    async def submit(client, frame, *, count=True):
        raw = observation_frame(frame)
        await client.append_audio(b"", video_frames=[base64.b64encode(raw).decode()])
        await client.commit(create_response=True)
        if count:
            result["sent"] += 1

    async def drain(reader):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            check(reader)
            # Camera observations received during speech can be coalesced.
            # Every commit needs an ACK, but only created responses need done.
            if (
                result["committed"] == result["sent"]
                and created == finished
                and 0 <= result["sent"] - len(finished) <= result["vision_follow"]
                and time.monotonic() - last_progress >= 2
            ):
                return
            await asyncio.sleep(0.1)
        raise AssertionError(
            f"Input did not drain: {result['sent']=}, {result['committed']=}, "
            f"created={len(created)}, finished={len(finished)}, {result['vision_follow']=}"
        )

    async def question(client, text, frame, *, cancel=False, expected=None):
        done = asyncio.create_task(client.wait_for("response.done", timeout_s=180))
        first = (
            asyncio.create_task(client.wait_for("response.output_audio_transcript.delta", timeout_s=180))
            if cancel
            else None
        )
        await asyncio.sleep(0)
        try:
            await client.send(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
                }
            )
            await submit(client, frame)
            if first is not None:
                await asyncio.wait((first, done), return_when=asyncio.FIRST_COMPLETED)
                assert first.done(), "Counting request completed without a transcript delta"
                event = await first
                assert not done.done(), "Response finished before cancellation"
                await client.cancel_response(event.response_id)
            terminal = await done
            await asyncio.sleep(0)
            assert terminal.raw["response"]["status"] == ("cancelled" if cancel else "completed"), terminal.raw
            rid = terminal.response_id
            answer = texts.get(rid, "")
            if expected:
                assert expected in answer.lower(), (expected, answer)
                assert audio.get(rid, 0) >= 2400, "Expected spoken answer"
            result["interactions"].append(
                {
                    "frame": frame,
                    "response_id": rid,
                    "text": answer,
                    "status": terminal.raw["response"]["status"],
                    "audio_bytes": audio.get(rid, 0),
                }
            )
        finally:
            done.cancel()
            if first is not None:
                first.cancel()
            await asyncio.gather(done, *([first] if first else []), return_exceptions=True)

    try:
        async with DuplexClient(
            url, model=model, config=config, reconnect=ReconnectPolicy(max_attempts=5, backoff_s=(0.25, 1.0))
        ) as client:
            reader = asyncio.create_task(consume(client))
            try:
                # Warm up the same image size before timing steady input.
                await submit(client, 0)
                await drain(reader)
                started = time.monotonic()
                next_frame = started
                for index in range(720 * cycles):
                    if index % 720 in (180, 360):
                        await drain(reader)
                        action_started = time.monotonic()
                        if index % 720 == 180:
                            await question(
                                client,
                                "What color is the shape in the latest image? Say its color, then count from one to one hundred aloud without skipping numbers.",
                                index - 1,
                                cancel=True,
                            )
                            await question(
                                client,
                                "What color is the shape in the latest image? Give only the color.",
                                index - 1,
                                expected="green",
                            )
                        else:
                            sid, token = client.session_id, client.resume_token
                            resumed = asyncio.create_task(client.wait_for("session.resumed", timeout_s=90))
                            await asyncio.sleep(0)
                            try:
                                await client._ws.close()
                                await resumed
                            finally:
                                resumed.cancel()
                                await asyncio.gather(resumed, return_exceptions=True)
                            assert client.session_id == sid
                            assert client.resume_token and client.resume_token != token
                            result["resumed"] = True
                            await question(
                                client,
                                "What color is the shape in the latest image? Give only the color.",
                                index - 1,
                                expected="cyan",
                            )
                        result["action_pauses_s"].append(time.monotonic() - action_started)
                        next_frame = time.monotonic()
                    await asyncio.sleep(max(0, next_frame - time.monotonic()))
                    check(reader)
                    lag = time.monotonic() - next_frame
                    next_frame += 1
                    assert lag < 2, f"Pacing lag {lag:.3f}s"
                    outstanding = max(result["sent"] - result["committed"], len(created - finished))
                    assert outstanding < 4, f"Backlog {outstanding}"
                    assert result["sent"] - len(finished) <= result["vision_follow"] + 4
                    await submit(client, index)
                    result["frames"] += 1
                    result["max_outstanding"] = max(result["max_outstanding"], outstanding + 1)
                    result["max_pacing_lag_s"] = max(result["max_pacing_lag_s"], lag)
                    if index % 60 == 59:
                        print(
                            json.dumps({"frames": result["frames"], "sent": result["sent"], "done": len(finished)}),
                            flush=True,
                        )
                await asyncio.sleep(max(0, next_frame - time.monotonic()))
                result["input_elapsed_s"] = time.monotonic() - started
                await drain(reader)
                assert created == finished
                assert len(cancelled) == cycles
                assert result.get("resumed") and result["input_elapsed_s"] >= 720 * cycles
                await asyncio.sleep(30)
                check(reader)
            finally:
                await client.close()
                await asyncio.wait_for(reader, 30)
        assert result["closed"] and not result["errors"]
        await asyncio.sleep(10)
        # A fresh session must actually infer after the long session releases admission.
        async with DuplexClient(url, model=model, config=config, reconnect=None) as client:
            done = asyncio.create_task(client.wait_for("response.done", timeout_s=180))
            await asyncio.sleep(0)
            await submit(client, 720 * cycles - 1, count=False)
            terminal = await done
            assert terminal.raw["response"]["status"] == "completed", terminal.raw
        result["passed"] = True
    except Exception as exc:
        result["failure"] = repr(exc)
        raise
    finally:
        result.update(elapsed_s=time.monotonic() - started, completed=len(finished), cancelled=len(cancelled))
        stream.close()
        (output_dir / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    return result["passed"]


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=1)
@pytest.mark.parametrize(
    "omni_server",
    [joyai_server_params(session_limits={"max_sessions": 1})],
    indirect=True,
)
def test_twelve_minute_observation_cancel_resume_and_readmission(omni_server, tmp_path):
    assert asyncio.run(
        run_continuous(
            f"ws://{omni_server.host}:{omni_server.port}/v1/realtime?duplex=1",
            omni_server.model,
            tmp_path,
            cycles=SOAK_CYCLES,
        )
    )
