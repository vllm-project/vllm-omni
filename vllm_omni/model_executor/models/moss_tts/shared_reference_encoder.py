# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MOSS-TTS shares one reference encoder across API processes.

The host/client machinery is the framework's
(``vllm_omni.utils.shared_reference_encoder``); MOSS clips are waveform
tensors and its artifacts are codes tensors (``CODES_WIRE``, the default).
"""

from vllm_omni.utils.shared_reference_encoder import (
    CODES_WIRE,
    SharedReferenceEncoderClient,
    SharedReferenceEncoderHost,
    SharedReferenceEncoderStartupError,
    elect_host,
    socket_path_fits,
)

__all__ = [
    "CODES_WIRE",
    "SharedReferenceEncoderClient",
    "SharedReferenceEncoderHost",
    "SharedReferenceEncoderStartupError",
    "elect_host",
    "socket_path_fits",
]
