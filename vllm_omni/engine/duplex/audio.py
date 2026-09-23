# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Compatibility re-export: the Realtime audio codec moved out of the duplex engine.

Decoding PCM / G.711 / WAV, resampling and re-encoding are wire-format work with
no duplex session in them, so the implementation now lives in
``vllm_omni.protocol.realtime.audio`` where a non-duplex Realtime surface can
use it too. Duplex code reaches it through ``vllm_omni.protocol.duplex``, the
one door the engine uses --- which is also where a duplex-specific conversion
would be introduced. Import from there in new code.
"""

from __future__ import annotations

from vllm_omni.protocol.duplex import (
    MAX_INPUT_SAMPLE_RATE_HZ,
    MIN_INPUT_SAMPLE_RATE_HZ,
    convert_input_audio_with_rate,
    convert_output_audio,
    decode_g711_alaw,
    decode_g711_ulaw,
    encode_float32_mono_wav_base64,
    encode_g711_alaw,
    encode_g711_ulaw,
    resample_pcm16_mono,
    validate_input_sample_rate_hz,
    wav_payload_to_pcm16,
)

__all__ = [
    "MAX_INPUT_SAMPLE_RATE_HZ",
    "MIN_INPUT_SAMPLE_RATE_HZ",
    "convert_input_audio_with_rate",
    "convert_output_audio",
    "decode_g711_alaw",
    "decode_g711_ulaw",
    "encode_float32_mono_wav_base64",
    "encode_g711_alaw",
    "encode_g711_ulaw",
    "resample_pcm16_mono",
    "validate_input_sample_rate_hz",
    "wav_payload_to_pcm16",
]
