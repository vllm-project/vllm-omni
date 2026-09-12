# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Compatibility imports for the shared Realtime codec."""

from vllm_omni.entrypoints.realtime.state import (
    REALTIME_ERROR_TYPES_BY_CODE as REALTIME_ERROR_TYPES_BY_CODE,
)
from vllm_omni.entrypoints.realtime.state import (
    REALTIME_G711_DEFAULT_SAMPLE_RATE_HZ as REALTIME_G711_DEFAULT_SAMPLE_RATE_HZ,
)
from vllm_omni.entrypoints.realtime.state import (
    REALTIME_INPUT_AUDIO_FORMATS as REALTIME_INPUT_AUDIO_FORMATS,
)
from vllm_omni.entrypoints.realtime.state import (
    REALTIME_OUTPUT_AUDIO_FORMATS as REALTIME_OUTPUT_AUDIO_FORMATS,
)
from vllm_omni.entrypoints.realtime.state import (
    REALTIME_PCM_DEFAULT_SAMPLE_RATE_HZ as REALTIME_PCM_DEFAULT_SAMPLE_RATE_HZ,
)
from vllm_omni.entrypoints.realtime.state import (
    REALTIME_PCM_F32_DEFAULT_SAMPLE_RATE_HZ as REALTIME_PCM_F32_DEFAULT_SAMPLE_RATE_HZ,
)
from vllm_omni.entrypoints.realtime.state import (
    RealtimeSessionState as RealtimeSessionState,
)
from vllm_omni.entrypoints.realtime.state import (
    RealtimeStateOwner as RealtimeStateOwner,
)
from vllm_omni.entrypoints.realtime.state import (
    _RealtimeResponseState as _RealtimeResponseState,
)
from vllm_omni.entrypoints.realtime.state import (
    realtime_default_sample_rate_hz as realtime_default_sample_rate_hz,
)
