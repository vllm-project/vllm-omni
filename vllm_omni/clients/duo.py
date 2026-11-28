# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Official duplex client preset for synchronized Duo video/stereo audio."""

from vllm_omni.clients.duplex import AudioFormat, SessionConfig


def create_duplex_session_config(*, idle_timeout_s=None):
    """Negotiate 24 fps / 32 kHz stereo; block geometry is server-configured.

    AudioDelta.audio_bytes contains interleaved little-endian PCM16. Its video
    extension carries JPEG frames and the matching integer sample clock.
    """
    return SessionConfig(
        input_audio=AudioFormat("pcm_f32le", 16000),
        output_audio=AudioFormat("pcm16", 32000),
        auto_response=True,
        turn_detection=None,
        temperature=0.0,
        max_output_tokens=128,
        idle_timeout_s=idle_timeout_s,
    )
