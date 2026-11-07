# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Worker-local random streams for request-ordered multimodal sampling."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeGuard

import torch

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalFeatureSpec

    from vllm_omni.inputs.mm_rng import MMRNGContext

MM_RNG_ITEM_KEY = "mm_rng_item"


@dataclass(frozen=True)
class RandomDrawSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass
class _RequestStream:
    seed: int
    specs: dict[int, RandomDrawSpec]
    generator: torch.Generator
    checkpoints: dict[int, torch.Tensor] = field(default_factory=dict)
    positions: dict[int, int] = field(init=False)
    draws: tuple[RandomDrawSpec, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.positions = {index: position for position, index in enumerate(self.specs)}
        self.draws = tuple(self.specs.values())


class RequestRNGProvider:
    """Own generators/checkpoints, delegating exact draw semantics to the model.

    Checkpoints denote *logical* image positions, not executed encoder calls.
    They survive preemption and encoder eviction, but never request completion.
    All access runs on the worker's serialized eager encoder execution path.
    """

    def __init__(self, advance: Callable[[torch.Generator, RandomDrawSpec], None]):
        self.advance = advance
        self.handles: dict[str, int] = {}
        self.streams: dict[int, _RequestStream] = {}
        self._next_handle = 0

    def register(self, req_id: str, seed: int, specs: dict[int, RandomDrawSpec]) -> int:
        if req_id in self.handles:
            handle = self.handles[req_id]
            stream = self.streams[handle]
            if stream.seed != seed or stream.specs != specs:
                raise ValueError("Multimodal RNG plan changed within a request")
            return handle
        if not specs or len({spec.device for spec in specs.values()}) != 1:
            raise ValueError("Multimodal RNG requires a nonempty plan on one device")
        generator = torch.Generator(device=next(iter(specs.values())).device).manual_seed(seed)
        handle = self._next_handle
        self._next_handle += 1
        self.handles[req_id] = handle
        self.streams[handle] = _RequestStream(seed, dict(sorted(specs.items())), generator, {0: generator.get_state()})
        return handle

    def discard(self, req_id: str) -> None:
        handle = self.handles.pop(req_id, None)
        if handle is not None:
            self.streams.pop(handle, None)

    @contextmanager
    def sample(self, handle: int, mm_index: int, actual: RandomDrawSpec) -> Iterator[torch.Generator]:
        stream = self.streams[handle]
        if stream.specs.get(mm_index) != actual:
            raise ValueError("VAE posterior does not match the request's random draw specification")
        position = stream.positions[mm_index]
        # Replay records every intermediate boundary, so checkpoints always
        # cover the contiguous prefix [0, len(checkpoints) - 1], even on failure.
        start = min(position, len(stream.checkpoints) - 1)
        stream.generator.set_state(stream.checkpoints[start])
        for i in range(start, position):
            self.advance(stream.generator, stream.draws[i])
            stream.checkpoints[i + 1] = stream.generator.get_state()
        yield stream.generator
        # An exception in sampling does not commit the failed image's state.
        stream.checkpoints[position + 1] = stream.generator.get_state()


class SupportsMMRequestRNG(Protocol):
    supports_mm_request_rng: bool

    def bind_mm_rng_provider(self, provider: RequestRNGProvider) -> None: ...

    def get_mm_rng_specs(
        self, features: list["MultiModalFeatureSpec"], context: "MMRNGContext"
    ) -> dict[int, RandomDrawSpec]: ...

    def advance_mm_rng(self, generator: torch.Generator, spec: RandomDrawSpec) -> None: ...


def supports_mm_request_rng(model: object) -> TypeGuard[SupportsMMRequestRNG]:
    return bool(getattr(model, "supports_mm_request_rng", False))
