# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real JoyAI-VL-Interaction -> Qwen3-TTS -> Code2Wav over the shared Realtime client."""

import asyncio
import base64
import io
import json
import wave
from pathlib import Path

import pytest
from PIL import Image

from tests.e2e.online_serving.helpers.joyai_vl_interaction import joyai_server_params
from tests.helpers.mark import hardware_test
from vllm_omni.clients.duplex import DuplexClient, EventCollector, SessionConfig

pytestmark = [pytest.mark.advanced_model, pytest.mark.full_model, pytest.mark.omni]


async def run_observations(url, model, output_dir, *, expect_streaming=False, instruction_pcm=b""):
    config = SessionConfig(
        auto_response=False,
        temperature=0.0,
        instructions=(
            None
            if instruction_pcm
            else "Describe the color of the current image in one short spoken sentence whenever it changes."
        ),
        extra_body={"tts_language": "English"},
    )
    collector = EventCollector()
    async with DuplexClient(url, model=model, config=config, reconnect=None) as client:
        consumer = asyncio.create_task(collector.consume(client))
        await asyncio.sleep(0)
        try:
            if not instruction_pcm:
                await client.send(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "What color is the latest image? Answer aloud whenever I send a new image.",
                                }
                            ],
                        },
                    }
                )
            pending_playback = None
            for index, color in enumerate(("red", "blue", "green", "yellow")):
                image = io.BytesIO()
                Image.new("RGB", (224, 224), color).save(image, format="JPEG")
                done = asyncio.create_task(client.wait_for("response.done", timeout_s=180))
                progress = (
                    asyncio.create_task(client.wait_for("response.output_audio_transcript.delta", timeout_s=180))
                    if index == 2
                    else None
                )
                await asyncio.sleep(0)
                try:
                    await client.append_audio(
                        instruction_pcm if index in (0, 2) else b"",
                        video_frames=[base64.b64encode(image.getvalue()).decode()],
                    )
                    await client.commit(create_response=True)
                    if pending_playback is not None:
                        # The next observation was committed after response.done
                        # but before the previous audio finished playing.
                        played_ms, response_id = pending_playback
                        await client.ack_playback(played_ms, response_id=response_id)
                    if progress is not None:
                        # Cancel after the model begins replying, rather than at
                        # response.created, which precedes engine submission.
                        event = await progress
                        assert not done.done(), "Response finished before cancellation"
                        await client.cancel_response(event.response_id)
                    terminal = await done
                finally:
                    if not done.done():
                        done.cancel()
                    if progress is not None:
                        progress.cancel()
                        await asyncio.gather(progress, return_exceptions=True)
                    await asyncio.gather(done, return_exceptions=True)
                if index == 2:
                    assert terminal.raw["response"]["status"] == "cancelled", terminal.raw
                    pending_playback = None
                    continue
                assert terminal.raw["response"]["status"] == "completed", terminal.raw
                assert color in collector.response_text(terminal.response_id).lower()
                pcm = collector.audio_bytes(terminal.response_id)
                assert len(pcm) >= 2400 and len(pcm) % 2 == 0, "Expected nonempty PCM16 speech"
                if expect_streaming:
                    chunks = [
                        event
                        for event in collector.events
                        if event.get("type") == "response.output_audio.delta"
                        and event.get("response_id") == terminal.response_id
                    ]
                    assert len(chunks) > 1, "Expected streamed audio before the final waveform"
                with wave.open(str(output_dir / f"observation-{index}.wav"), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(collector.output_sample_rate_hz)
                    wav.writeframes(pcm)
                pending_playback = (len(pcm) * 500 / collector.output_sample_rate_hz, terminal.response_id)
            assert pending_playback is not None
            played_ms, response_id = pending_playback
            await client.ack_playback(played_ms, response_id=response_id)
            assert len(set(collector.response_ids)) == 4
            if instruction_pcm:
                transcripts = [
                    event["transcript"]
                    for event in collector.events
                    if event.get("type") == "conversation.item.input_audio_transcription.completed"
                ]
                assert len(transcripts) == 2 and all("color" in text.lower() for text in transcripts), transcripts
            assert not collector.errors(), collector.errors()
            await client.close()
            await consumer
        finally:
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
            (output_dir / "events.json").write_text(json.dumps(collector.events, indent=2), encoding="utf-8")
    # The default profile admits one session: close must release its slot.
    async with DuplexClient(url, model=model, config=config, reconnect=None):
        pass


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=1)
@pytest.mark.parametrize(
    "omni_server,with_asr",
    [
        pytest.param(joyai_server_params(), False, id="visual-duplex"),
        pytest.param(joyai_server_params(asr=True), True, id="asr-duplex"),
    ],
    indirect=["omni_server"],
)
def test_committed_observations_produce_speech_cancel_and_release_session(omni_server, tmp_path, with_asr):
    instruction_pcm = b""
    if with_asr:
        path = Path(__file__).resolve().parents[2] / "assets/joyai_vl_interaction/color_instruction_16k.wav"
        with wave.open(str(path), "rb") as audio:
            assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 16000)
            instruction_pcm = audio.readframes(audio.getnframes())
    asyncio.run(
        run_observations(
            f"ws://{omni_server.host}:{omni_server.port}/v1/realtime?duplex=1",
            omni_server.model,
            tmp_path,
            expect_streaming=not with_asr,
            instruction_pcm=instruction_pcm,
        )
    )
