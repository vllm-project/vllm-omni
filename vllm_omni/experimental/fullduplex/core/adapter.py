# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from vllm_omni.experimental.fullduplex.core.session import DuplexSession


@dataclass
class DuplexCapability:
    input_modalities: frozenset[str]
    output_modalities: frozenset[str]

    proactive: bool = False


@dataclass
class OutputChunk:
    modality: str
    data: Any


class DuplexAdapter(ABC):
    @abstractmethod
    def capabilities(self) -> DuplexCapability: ...

    @abstractmethod
    async def on_input(self, session: DuplexSession, modality: str, data: Any) -> None: ...

    @abstractmethod
    def respond(self, session: DuplexSession) -> AsyncIterator[OutputChunk]: ...

    def should_respond(self, session: DuplexSession) -> bool:
        return True

    async def on_barge_in(self, session: DuplexSession) -> None: ...

    async def on_playback_ack(self, session: DuplexSession, cursor: int) -> None: ...
