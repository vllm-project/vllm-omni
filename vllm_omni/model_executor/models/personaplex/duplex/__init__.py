# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PersonaPlex full-duplex integration.

PersonaPlex (``nvidia/personaplex-7b-v1``) is a Moshi finetune: a pure-lockstep
speech-to-speech model. This package plugs it into the generic duplex serving
stack through the standard plugin seams (``duplex_serving_adapter`` /
``duplex_runtime_extension`` dotted strings in the model's ``pipeline.py``):

- :class:`PersonaPlexConfig`  immutable session config (voice / persona / sampling)
- :class:`PersonaPlexServingRuntimeAdapter`  the ``ServingRuntimeAdapter`` impl
- :class:`PersonaPlexDuplexRuntimeExtension`  the engine ``DuplexRuntimeExtension``
- :class:`PersonaPlexStage0DuplexRuntime`  Stage 0 session state and prefill
- :class:`PersonaPlexPcmAppendBuffer`  PCM input framing
"""

from .config import (
    PersonaPlexConfig,
)
from .input import (
    PersonaPlexPcmAppendBuffer,
)
from .policy import PrefillStep
from .runtime_extension import (
    PersonaPlexDuplexRuntimeExtension,
)
from .serving_adapter import (
    PersonaPlexServingRuntimeAdapter,
)
from .stage0 import (
    PersonaPlexStage0DuplexRuntime,
)

__all__ = [
    "PersonaPlexConfig",
    "PersonaPlexDuplexRuntimeExtension",
    "PersonaPlexPcmAppendBuffer",
    "PersonaPlexServingRuntimeAdapter",
    "PersonaPlexStage0DuplexRuntime",
    "PrefillStep",
]
