# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Transport-independent streaming TTS frontend contract and profile registry."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


class StreamingTextFrontend(Protocol):
    """One utterance's text processing, including commitment and segmentation.

    ``feed`` accepts arbitrary transport chunks and returns ordered, irreversible
    non-whitespace synthesis segments. ``finish`` flushes the text at EOF. Neither
    operation may block or perform synthesis. Raise ``ValueError`` on invalid
    input or a resource limit; no segments from that call are submitted.

    A frontend owns its text state and boundaries, so normalization or a future
    TN/CAPS composition can be substituted without transport changes. Factories
    must return a fresh instance for every utterance. Instances are discarded
    after EOF, failure or cancellation; they own no external resources.
    """

    def feed(self, text: str) -> tuple[str, ...]: ...

    def finish(self) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class StreamingTextFrontendConfig:
    language: str
    max_pending_chars: int


StreamingTextFrontendFactory = Callable[[StreamingTextFrontendConfig], StreamingTextFrontend]


def _create_zh_en_frontend(config: StreamingTextFrontendConfig) -> StreamingTextFrontend:
    # Load the concrete implementation only when its profile is selected.
    from vllm_omni.model_executor.stage_input_processors.streaming_text_commitment import CommitmentTextFrontend

    return CommitmentTextFrontend(max_pending_chars=config.max_pending_chars)


_FRONTEND_FACTORIES: dict[str, StreamingTextFrontendFactory] = {
    "zh_en_special_v1": _create_zh_en_frontend,
}


def register_streaming_text_frontend(profile: str, factory: StreamingTextFrontendFactory) -> None:
    """Register a profile at startup; adapters advertise it in their capability.

    Reject duplicate registrations to keep versioned profile behavior stable.
    """
    if profile in _FRONTEND_FACTORIES:
        raise ValueError(f"Streaming text frontend profile already registered: {profile!r}")
    _FRONTEND_FACTORIES[profile] = factory


def get_streaming_text_frontend_factory(profile: str) -> StreamingTextFrontendFactory:
    """Resolve an adapter's declared profile without constructing text state."""
    try:
        return _FRONTEND_FACTORIES[profile]
    except KeyError:
        raise ValueError(f"unsupported commitment profile: {profile!r}") from None
