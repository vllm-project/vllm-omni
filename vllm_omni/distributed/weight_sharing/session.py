# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Worker-local bootstrap for a rank-paired weight-sharing session.

The startup/lifecycle layer owns the fields that must agree across processes:
group id, generation, session id, peer ids, and the per-rank physical GPU UUID.
:class:`SharedWeightGroupConfig` is the worker-local view of that agreement.

The backend is transport-agnostic: this module never opens sockets or CUDA
resources. Callers relay the opaque bundle/receipt bytes over their own trusted
control plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .contracts import Identity, SessionContext

DEFAULT_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class SharedWeightGroupConfig:
    """Static agreement for one rank of a sharing group.

    Models select their adapter through ``get_weight_sharing_adapter()``.

    ``device_index`` is intentionally absent: it is a per-process, torch-visible
    ordinal supplied by the worker when it assembles the :class:`SessionContext`.

    ``payload`` carries the provider's claimed bundle bytes on the consumer
    rank. The provider produces it with ``claim_export`` and leaves this field
    unset; the startup coordination layer (or a test) fills it on the consumer
    before model load.
    """

    group_id: str
    generation: int
    session_id: str
    provider_id: str
    consumer_id: str
    role: Literal["provider", "consumer"]
    rank: int
    tp_size: int
    device_uuid: str
    model_id: str
    checkpoint_revision: str
    layout: str
    payload: bytes | None = None


def build_identity(group: SharedWeightGroupConfig) -> Identity:
    return Identity(
        group_id=group.group_id,
        generation=group.generation,
        session_id=group.session_id,
        provider_id=group.provider_id,
        consumer_id=group.consumer_id,
        rank=group.rank,
        tp_size=group.tp_size,
        device_uuid=group.device_uuid,
        protocol_version=DEFAULT_PROTOCOL_VERSION,
    )


def build_session_context(group: SharedWeightGroupConfig, device_index: int) -> SessionContext:
    """Assemble the backend context for a torch-visible ordinal."""
    return SessionContext(
        identity=build_identity(group),
        device_index=device_index,
    )
