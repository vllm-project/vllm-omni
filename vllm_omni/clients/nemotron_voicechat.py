# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Client-side duplex session preset for Nemotron VoiceChat."""

from __future__ import annotations

from vllm_omni.clients.duplex import AudioFormat, SessionConfig

__all__ = ["create_duplex_session_config"]


def create_duplex_session_config(
    *,
    instructions: str | None = None,
    tools: list[dict[str, object]] | None = None,
    idle_timeout_s: float | None = None,
    extra_body: dict[str, object] | None = None,
) -> SessionConfig:
    """Use 16 kHz float32 input and 22.05 kHz PCM16 output; append 80 ms frames."""
    session_extra_body = dict(extra_body or {})
    if tools is not None:
        session_extra_body["realtime_tools"] = tools
    return SessionConfig(
        input_audio=AudioFormat("pcm_f32le", 16_000),
        output_audio=AudioFormat("pcm16", 22_050),
        instructions=instructions,
        idle_timeout_s=idle_timeout_s,
        extra_body=session_extra_body,
    )
