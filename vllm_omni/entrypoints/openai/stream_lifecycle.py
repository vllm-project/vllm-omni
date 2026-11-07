# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Keep ownership of the engine generator until the HTTP stream closes."""

import logging

import anyio

logger = logging.getLogger(__name__)


async def owned_chat_stream(response, result_generator, engine, request_id):
    completed = False
    try:
        async for chunk in response:
            yield chunk
        completed = True
    finally:
        # StreamingResponse cancels an AnyIO task group on disconnect. Shield
        # cleanup from that cancel scope so a suspended inner generator can
        # receive GeneratorExit and release its own engine/orchestrator state.
        with anyio.CancelScope(shield=True):
            for generator in (response, result_generator):
                close = getattr(generator, "aclose", None)
                if close is not None:
                    try:
                        await close()
                    except Exception:
                        logger.exception("Could not close chat generator for %s", request_id)
            if not completed:
                await engine.abort(request_id)
