# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Speech/TTS endpoint family.

This package owns all routes under ``/v1/audio/`` (speech synthesis, voice
management, and streaming speech websocket).  The :data:`speech_router` is
included into the main FastAPI app by ``api_server.build_app`` via
``app.include_router``.
"""

from vllm_omni.entrypoints.openai.speech.router import speech_router

__all__ = ["speech_router"]
