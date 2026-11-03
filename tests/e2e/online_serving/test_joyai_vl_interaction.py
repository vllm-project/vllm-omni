# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real JoyAI-VL-Interaction -> Qwen3-TTS -> Code2Wav over the shared Realtime client."""

import asyncio
import base64
import io
import json
import wave

import pytest
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.clients.duplex import DuplexClient, EventCollector, SessionConfig

pytestmark = [pytest.mark.advanced_model, pytest.mark.omni]


async def run_observations(url, model, output_dir):
    config = SessionConfig(
        auto_response=False,
        temperature=0.0,
        instructions="Describe the color of the current image in one short spoken sentence whenever it changes.",
        extra_body={"tts_language": "English"},
    )
    collector = EventCollector()
    async with DuplexClient(url, model=model, config=config, reconnect=None) as client:
        consumer = asyncio.create_task(collector.consume(client))
        await asyncio.sleep(0)
        try:
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
            for index, color in enumerate(("red", "blue")):
                image = io.BytesIO()
                Image.new("RGB", (224, 224), color).save(image, format="JPEG")
                done = asyncio.create_task(client.wait_for("response.done", timeout_s=180))
                await asyncio.sleep(0)
                try:
                    await client.append_audio(b"", video_frames=[base64.b64encode(image.getvalue()).decode()])
                    await client.commit(create_response=True)
                    if pending_playback is not None:
                        # The next observation was committed after response.done
                        # but before the previous audio finished playing.
                        played_ms, response_id = pending_playback
                        await client.ack_playback(played_ms, response_id=response_id)
                    terminal = await done
                finally:
                    if not done.done():
                        done.cancel()
                    await asyncio.gather(done, return_exceptions=True)
                assert terminal.raw["response"]["status"] == "completed", terminal.raw
                pcm = collector.audio_bytes(terminal.response_id)
                assert len(pcm) >= 2400 and len(pcm) % 2 == 0, "Expected nonempty PCM16 speech"
                with wave.open(str(output_dir / f"observation-{index}.wav"), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(collector.output_sample_rate_hz)
                    wav.writeframes(pcm)
                pending_playback = (len(pcm) * 500 / collector.output_sample_rate_hz, terminal.response_id)
            assert pending_playback is not None
            played_ms, response_id = pending_playback
            await client.ack_playback(played_ms, response_id=response_id)
            assert len(set(collector.response_ids)) == 2
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
    "omni_server",
    [
        OmniServerParams(
            model="jdopensource/JoyAI-VL-Interaction-Preview",
            # The fixture patches load_format on explicit stages; materialize the
            # duplex override on the same base used by joyai_vl_interaction_duplex.yaml.
            stage_config_path=modify_stage_config(
                get_deploy_config_path("joyai_vl_interaction.yaml"),
                updates={"session_mode": "duplex"},
            ),
        )
    ],
    indirect=True,
)
def test_committed_video_observations_produce_speech_and_release_session(omni_server, tmp_path):
    asyncio.run(
        run_observations(
            f"ws://{omni_server.host}:{omni_server.port}/v1/realtime?duplex=1",
            omni_server.model,
            tmp_path,
        )
    )
