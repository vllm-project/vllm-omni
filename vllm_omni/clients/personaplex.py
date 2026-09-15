# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Client-side duplex session preset for PersonaPlex duplex serving."""

from __future__ import annotations

from dataclasses import replace

from vllm_omni.clients.duplex import AudioFormat, SessionConfig

__all__ = ["create_duplex_session_config"]


def create_duplex_session_config(*, voice: str = "NATF2.pt", persona: str = "", **overrides: object) -> SessionConfig:
    """Session preset matching the PersonaPlex duplex deployment (24 kHz float32 in)."""
    config = SessionConfig(
        input_audio=AudioFormat("pcm_f32le", 24_000),
        output_audio=AudioFormat("pcm16", 24_000),
        voice=voice,
        instructions=persona,
    )
    return replace(config, **overrides) if overrides else config
