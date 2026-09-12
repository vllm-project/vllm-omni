# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Compatibility imports for the shared Realtime codec."""

from vllm_omni.entrypoints.realtime.audio import (
    MAX_INPUT_SAMPLE_RATE_HZ as MAX_INPUT_SAMPLE_RATE_HZ,
)
from vllm_omni.entrypoints.realtime.audio import (
    MIN_INPUT_SAMPLE_RATE_HZ as MIN_INPUT_SAMPLE_RATE_HZ,
)
from vllm_omni.entrypoints.realtime.audio import (
    convert_input_audio_with_rate as convert_input_audio_with_rate,
)
from vllm_omni.entrypoints.realtime.audio import (
    convert_output_audio as convert_output_audio,
)
from vllm_omni.entrypoints.realtime.audio import (
    decode_g711_alaw as decode_g711_alaw,
)
from vllm_omni.entrypoints.realtime.audio import (
    decode_g711_ulaw as decode_g711_ulaw,
)
from vllm_omni.entrypoints.realtime.audio import (
    encode_float32_mono_wav_base64 as encode_float32_mono_wav_base64,
)
from vllm_omni.entrypoints.realtime.audio import (
    encode_g711_alaw as encode_g711_alaw,
)
from vllm_omni.entrypoints.realtime.audio import (
    encode_g711_ulaw as encode_g711_ulaw,
)
from vllm_omni.entrypoints.realtime.audio import (
    resample_pcm16_mono as resample_pcm16_mono,
)
from vllm_omni.entrypoints.realtime.audio import (
    validate_input_sample_rate_hz as validate_input_sample_rate_hz,
)
from vllm_omni.entrypoints.realtime.audio import (
    wav_payload_to_pcm16 as wav_payload_to_pcm16,
)
