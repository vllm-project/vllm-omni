#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LiveKit voice agent on vLLM-Omni's OpenAI Live endpoint (``/v1/live/sessions``).

    VLLM_OMNI_HOST=127.0.0.1 LIVE_MODEL=<served model name> LIVE_VOICE=default \\
        python agent.py console        # local mic/speaker
    python agent.py start              # worker for a LiveKit server (LIVEKIT_URL)

Requires a livekit-agents checkout whose OpenAI plugin has ``GPTLiveModel``
(livekit-agents >= 1.8.3).
"""

from __future__ import annotations

import logging
import os

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins.openai.realtime import GPTLiveModel

logger = logging.getLogger("live-agent")

VLLM_BASE_URL = f"http://{os.environ.get('VLLM_OMNI_HOST', '127.0.0.1')}:{os.environ.get('VLLM_OMNI_PORT', '8000')}/v1"
# delegation.responses.model must be the served model name.
LIVE_MODEL = os.environ["LIVE_MODEL"]
# A voice id of the served model (MiniCPM-o: default, PersonaPlex: NATF2, ...).
# OpenAI voice names are rejected.
LIVE_VOICE = os.environ.get("LIVE_VOICE", "default")

server = AgentServer(
    ws_url=os.environ.get("LIVEKIT_URL", "ws://localhost:7880"),
    api_key=os.environ.get("LIVEKIT_API_KEY", "devkey"),
    api_secret=os.environ.get("LIVEKIT_API_SECRET", "secret"),
)


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice assistant. Keep replies short and conversational.",
        )

    # No on_enter greeting: session.commentary.append (generate_reply) is ignored
    # by vLLM-Omni, so a greeting request would produce nothing.


def build_model() -> GPTLiveModel:
    return GPTLiveModel(
        base_url=VLLM_BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY", "unused"),
        model=LIVE_MODEL,
        voice=LIVE_VOICE,
        responses_options={"model": LIVE_MODEL},
    )


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(llm=build_model())

    async def log_usage() -> None:
        logger.info("usage: %s", session.usage)

    ctx.add_shutdown_callback(log_usage)
    await session.start(agent=Assistant(), room=ctx.room)
    logger.info("Live agent connected to %s (model=%s, voice=%s)", VLLM_BASE_URL, LIVE_MODEL, LIVE_VOICE)


if __name__ == "__main__":
    cli.run_app(server)
