# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PersonaPlex duplex protocol home (moshi-free).

Defines the narrow ``FrameStepper`` seam (one 80 ms user frame in, one agent
frame + text piece out) that the session driver, the duplex adapter and the
GPU-free tests depend on, and re-exports the native lockstep engine that
implements it. The implementation lives in ``runtime.py`` and is built
entirely from vllm-omni-native components (streaming Helium temporal, input
embeddings, depformer, streaming Mimi codec) — the external ``moshi`` package
is no longer used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameOutput:
    """Result of stepping the model by exactly one 80 ms frame.

    Attributes:
        audio: Agent PCM for this frame (``frame_size`` mono float32 samples), or
            ``None`` during the model's initial delay/warmup frames.
        text: Inner-monologue text piece for this frame, or ``None`` when the
            model emitted a padding token (silence between words).
    """

    audio: NDArray[np.float32] | None
    text: str | None


@runtime_checkable
class FrameStepper(Protocol):
    """The single seam between PersonaPlex and the duplex framework.

    A stepper consumes one user audio frame and returns one agent frame. It owns
    all conversation state (KV caches / streaming buffers); callers never re-feed
    history. ``PersonaPlexEngine`` is the real implementation; tests pass a stub.
    """

    sample_rate: int
    frame_size: int

    def open_session(self, voice_prompt: str | None = None, persona: str | None = None) -> None:
        """Reset streaming state and inject the voice clone + persona prompt."""
        ...

    def step(self, user_pcm: NDArray[np.float32]) -> FrameOutput:
        """Advance the conversation by one frame of exactly ``frame_size`` samples."""
        ...


from vllm_omni.experimental.fullduplex.personaplex.runtime import (  # noqa: E402
    PersonaPlexEngine,
    PrefillStep,
)

__all__ = [
    "FrameOutput",
    "FrameStepper",
    "PersonaPlexEngine",
    "PrefillStep",
]
