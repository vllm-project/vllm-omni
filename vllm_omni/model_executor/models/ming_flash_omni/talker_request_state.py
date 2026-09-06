# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright (c) Ant Group. All rights reserved.
"""Per-request audio-side state for the native-paged Ming talker.

When the AR loop moves out of the model and into the scheduler/runner, the
LLM KV cache is paged and engine-managed, but the *audio half* of the pipeline
(latent history, CFM condition, per-request RNG, step counter, stop flag) is
NOT KV and must be carried across decode steps explicitly. This is the audio
analog of KV, keyed by request id and evicted on finish.

Reference: voxcpm2/voxcpm2_talker.py ``_RequestState`` + ``_switch_to_request``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class MingTalkerRequestState:
    """Audio-side decode state for a single in-flight talker request."""

    req_id: str

    # Rolling latent history fed into CFM each step (B, his_patch_size, latent_dim).
    his_lat: torch.Tensor | None = None
    # Accumulated generated latents, decoded to waveform on finish.
    all_latents: list[torch.Tensor] = field(default_factory=list)

    # Decode bookkeeping.
    step: int = 0
    min_steps: int = 10
    max_steps: int = 200
    prefill_done: bool = False
    finished: bool = False

    # Sampling knobs resolved once at prefill (cfg / sigma / temperature).
    cfg: float = 2.0
    sigma: float = 0.25
    temperature: float = 0.0
    stream_decode: bool = True

    # Per-request seeded RNG so SDE noise is deterministic and reproducible.
    generator: torch.Generator | None = None
    seed: int | None = None

    # Aggregator output embedding to feed as next step's inputs_embeds.
    next_inputs_embed: torch.Tensor | None = None


class MingTalkerStateManager:
    """Owns ``MingTalkerRequestState`` objects keyed by request id.

    Mirrors the voxcpm2 pattern: create on prefill, look up / switch on each
    decode step, evict on finish. Lives on the talker model instance.
    """

    def __init__(self) -> None:
        self._states: dict[str, MingTalkerRequestState] = {}

    def create(self, req_id: str, **init_kwargs) -> MingTalkerRequestState:
        """Allocate state at prefill; seed the per-request RNG."""
        if req_id in self._states:
            raise ValueError(f"Ming talker request state already exists for {req_id!r}")

        device = init_kwargs.pop("generator_device", None)
        seed = init_kwargs.pop("seed", None)
        if seed is None:
            seed = _stable_request_seed(req_id)

        generator = init_kwargs.pop("generator", None)
        if generator is None:
            generator = _make_generator(device)
            generator.manual_seed(int(seed))

        state = MingTalkerRequestState(
            req_id=req_id,
            generator=generator,
            seed=int(seed),
            **init_kwargs,
        )
        self._states[req_id] = state
        return state

    def get(self, req_id: str) -> MingTalkerRequestState:
        """Look up the state for a decode step (the ``_switch_to_request`` analog)."""
        try:
            return self._states[req_id]
        except KeyError as e:
            raise KeyError(f"No Ming talker request state for {req_id!r}") from e

    def evict(self, req_id: str) -> None:
        """Drop state when the request finishes (free his_lat / latents)."""
        self._states.pop(req_id, None)

    def __contains__(self, req_id: str) -> bool:
        return req_id in self._states

    def __len__(self) -> int:
        return len(self._states)


def _stable_request_seed(req_id: str) -> int:
    digest = hashlib.blake2b(req_id.encode("utf-8"), digest_size=8).digest()
    # torch.Generator.manual_seed accepts a signed 64-bit-ish Python int, keep
    # it in the portable positive 63-bit range.
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _make_generator(device: torch.device | str | None) -> torch.Generator:
    if device is None:
        return torch.Generator()
    try:
        return torch.Generator(device=device)
    except (RuntimeError, TypeError):
        return torch.Generator()
