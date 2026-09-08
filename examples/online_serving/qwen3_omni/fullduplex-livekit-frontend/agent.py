# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "livekit-agents[openai,turn-detector,silero]>=1.6.10",
#     "httpx",
# ]
# ///
import logging
import os
from typing import Any

import httpx
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

VLLM_BASE_URL = f"http://{os.environ["VLLM_OMNI_HOST"]}:8000/v1"

# This agent is only ever launched by run-livekit-stack.sh against its own
# livekit-server --dev instance, so these match that dev server's fixed
# defaults rather than being configurable.
server = AgentServer(
    ws_url="ws://localhost:7880",
    api_key="devkey",
    api_secret="secret",
)


class VoiceAssistant(Agent):
    @function_tool()
    async def lookup_weather(
        self,
        context: RunContext,
        location: str,
    ) -> dict[str, Any]:
        return {
            "weather": "the fog is coming the fog is coming the fog is coming THE FOG IS COMING THE FOG IS COMING",
            "temperature_f": 1000,
        }

    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice assistant. Respond naturally and concisely.",
        )


async def _get_model_name() -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{VLLM_BASE_URL}/models")
        resp.raise_for_status()
        data = resp.json()
    models = data.get("data", [])
    if not models:
        raise RuntimeError(f"No models available at {VLLM_BASE_URL}/models")
    model_id = models[0]["id"]
    logger.info("Using model: %s", model_id)
    return model_id

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    model_name = await _get_model_name()

    # Qwen3-Omni only supports turn_detection=null (manual mode) -- it does
    # not perform voice activity detection itself, so turn-taking must be
    # decided client-side. semantic_vad (MiniCPM-o only) is rejected by
    # vLLM-Omni's OpenAI-compatible realtime endpoint for Qwen3-Omni.
    model = RealtimeModel(
        base_url=VLLM_BASE_URL,
        model=model_name,
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
        "Voice assistant started (model=%s), connected to vLLM-Omni at %s",
        model_name,
        VLLM_BASE_URL,
    )


if __name__ == "__main__":
    cli.run_app(server)
