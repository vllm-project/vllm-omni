#!/usr/bin/env -S uv run --script
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "livekit-agents[openai,turn-detector,silero]>=1.6.10",
# ]
# ///
import logging
import os

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    TurnHandlingOptions,
    cli,
    function_tool,
    inference,
)
from livekit.plugins.openai.realtime import RealtimeModel

logger = logging.getLogger("voice-assistant")

VLLM_BASE_URL = f"http://{os.environ['VLLM_OMNI_HOST']}:8000/v1"
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")

server = AgentServer(
    ws_url=LIVEKIT_URL,
    api_key=os.environ.get("LIVEKIT_API_KEY", "devkey"),
    api_secret=os.environ.get("LIVEKIT_API_SECRET", "secret"),
)


class VoiceAssistant(Agent):
    @function_tool()
    async def lookup_weather(
        self,
        context: RunContext,
        location: str,
    ) -> dict[str, str | int]:
        """Return deterministic sample data for the demo tool."""
        return {
            "location": location,
            "weather": "clear skies",
            "temperature_f": 72,
        }

    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice assistant. Respond naturally and concisely.",
        )


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Qwen3-Omni only supports turn_detection=null (manual mode) -- it does
    # not perform voice activity detection itself, so turn-taking must be
    # decided client-side. semantic_vad (MiniCPM-o only) is rejected by
    # vLLM-Omni's OpenAI-compatible realtime endpoint for Qwen3-Omni.
    # Omni serves one model, so an empty model lets the server select it.
    model = RealtimeModel(
        base_url=VLLM_BASE_URL,
        model="",
        api_key="unused",
    )

    # turn_handling's client-side TurnDetector means the server isn't asked
    # to decide turns, so turn_detection is left NOT_GIVEN on RealtimeModel.
    session = AgentSession(
        llm=model,
        turn_handling=TurnHandlingOptions(turn_detection=inference.TurnDetector()),
    )

    await session.start(
        agent=VoiceAssistant(),
        room=ctx.room,
    )

    logger.info(
        "Voice assistant started, connected to vLLM-Omni at %s",
        VLLM_BASE_URL,
    )


if __name__ == "__main__":
    cli.run_app(server)
