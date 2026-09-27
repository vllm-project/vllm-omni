# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PersonaPlex full-duplex integration.

PersonaPlex (``nvidia/personaplex-7b-v1``) is a Moshi finetune: a pure-lockstep
speech-to-speech model. This package plugs it into the unified full-duplex
framework through the one seam the framework has, ``PipelineConfig.duplex_plugin``:

- :class:`PersonaPlexDuplexPlugin`  the ``DuplexModelPlugin`` (engine + session policy)
- :class:`PersonaPlexStage0DuplexRuntime`  worker-side lockstep state and first-append prefill
- :class:`PersonaPlexPcmAppendBuffer`  80 ms PCM input framing
"""

from .input import PersonaPlexPcmAppendBuffer
from .plugin import PersonaPlexDuplexPlugin
from .policy import PrefillStep
from .stage0 import PersonaPlexStage0DuplexRuntime

__all__ = [
    "PersonaPlexDuplexPlugin",
    "PersonaPlexPcmAppendBuffer",
    "PersonaPlexStage0DuplexRuntime",
    "PrefillStep",
]
