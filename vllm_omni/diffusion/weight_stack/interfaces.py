# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Data types and protocols shared by the unified weight stack.

This module is the contract between the startup driver (sub-RFC 01-5), the
Loader (02), the host weight manager (03), the Offloader (04) and the
configuration layer (05).  It only depends on the standard library and
``torch``: types owned by the other sub-RFCs are referenced under
``TYPE_CHECKING`` only, so importing this module never pulls in a concrete
implementation and cannot introduce an import cycle.

The definitions mirror RFC #9536, ``rfcs/01-startup-transaction.md`` section 2.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
from torch import nn

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup
    from vllm.config import VllmConfig

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.weight_stack.components import ComponentCatalog
    from vllm_omni.diffusion.weight_stack.host import (
        Election,
        ElectionRecord,
        HostLayout,
        LocalRecord,
        MemoryRange,
        SharePlan,
    )
    from vllm_omni.diffusion.weight_stack.loader import ExpectedManifest, FinalManifest, Preflight
    from vllm_omni.diffusion.weight_stack.offload import PlacementPlan, Unit
    from vllm_omni.diffusion.weight_stack.policy import OffloadPolicy
    from vllm_omni.diffusion.weight_stack.startup import FaultPlan  # type: ignore[attr-defined]


class LoadState(enum.Enum):
    """State of one unified startup transaction (RFC 01 section 4)."""

    STRUCTURE_READY = "structure_ready"
    PLANNED = "planned"
    SOURCES_PREPARED = "sources_prepared"
    ADMITTED = "admitted"
    TRANSACTION_READY = "transaction_ready"
    WEIGHTS_READY = "weights_ready"
    FINALIZED = "finalized"
    COMMITTED = "committed"
    IN_SERVICE = "in_service"
    FAILED = "failed"  # terminal: startup failed, this rank released its resources
    CLOSED = "closed"  # terminal: close completed


@dataclass(frozen=True)
class ParallelCoordinate:
    """Size and rank of one parallel dimension."""

    size: int
    rank: int


@dataclass(frozen=True)
class RankCoordinates:
    """This rank's position in every parallel dimension of the replica."""

    tp: ParallelCoordinate
    sp: ParallelCoordinate
    ulysses: ParallelCoordinate
    ring: ParallelCoordinate
    cfg: ParallelCoordinate
    dp: ParallelCoordinate
    pp: ParallelCoordinate
    ep: ParallelCoordinate | None  # None when expert parallelism is not enabled
    fs: ParallelCoordinate | None  # None when HSDP is not enabled


@dataclass(frozen=True)
class Topology:
    """Replica-wide topology; every rank of the replica gets the same value."""

    deployment_id: str  # 32 hexadecimal characters
    deployment_scoped: bool  # False: no deployment id was configured; rank 0 generated a replica-local one
    stage_id: int
    world_size: int
    records: tuple[LocalRecord, ...]  # ordered by rank, length == world_size
    coordinates: tuple[RankCoordinates, ...]  # ordered by rank, length == world_size


@dataclass(frozen=True)
class RankFingerprints:
    """What one rank contributes to the first confirmation point."""

    rank: int
    content_identities: Mapping[str, str]  # component name -> content identity digest; only components declared here
    plan_fingerprint: str  # digest of the node-local backend part of HostLayout
    # component name -> digest of this component's AllGather Units, their layout and Transport (RFC 04 section 2);
    # only components that have allgather Units; content identity is grouped by content_identities above
    collective: Mapping[str, str]


Fingerprints = tuple[RankFingerprints, ...]  # ordered by rank


@dataclass(frozen=True)
class AllGatherGroup:
    """One AllGather process group of one component."""

    component: str
    ranks: tuple[int, ...]  # ascending, at least 2 ranks
    group: ProcessGroup | None  # None when this rank is not a member of the group


AllGatherGroups = Mapping[str, tuple[AllGatherGroup, ...]]  # component name -> its groups, ordered by smallest rank


@dataclass(frozen=True)
class Elections:
    """Election results exchanged at the first confirmation point."""

    # this rank's election result; the Election that wins the write process owns the listening socket/thread
    local: Election
    records: tuple[ElectionRecord, ...]  # election results collected in the second round of point 1, ordered by rank


class MemoryPools(Protocol):
    """Sleep-mode memory pools entered by the Loader and the Offloader."""

    def weights(self) -> AbstractContextManager[None]:
        """Pool whose contents are backed up across sleep."""

    def slots(self) -> AbstractContextManager[None]:
        """Pool whose contents are released (not backed up) across sleep."""


@dataclass(frozen=True)
class StartupContext:
    """Everything the startup driver needs for one unified startup."""

    od_config: OmniDiffusionConfig
    vllm_config: VllmConfig
    device: torch.device
    rank: int
    local_rank: int
    world_size: int
    load_format: str  # "default", "diffusers" or "custom_pipeline"
    custom_pipeline_name: str | None
    memory_pools: MemoryPools
    policy: OffloadPolicy | None  # None means no offload
    compile_enabled: bool  # not enforce_eager, and the platform supports torch inductor
    faults: FaultPlan | None  # only used by fault-injection tests


class Loader(Protocol):
    """Builds the structure, reads the checkpoint and writes every tensor."""

    def build_structure(self) -> tuple[nn.Module, ComponentCatalog]:
        """Build the pipeline in the structure-only context (step 2)."""

    def preflight(self, catalog: ComponentCatalog, plan: PlacementPlan) -> Preflight:
        """Resolve sources and compute content identities and the expected manifest (step 5)."""

    def write(self, model: nn.Module, storage: HostStorage, transaction: LoadTransaction) -> None:
        """Write every tensor through the transaction targets (step 9)."""

    def finalize(self, model: nn.Module, storage: HostStorage) -> FinalManifest:
        """Run the finalize sequence and return the final manifest (step 10)."""

    def close(self) -> None:
        """Close the construction context, read traps and readers. Idempotent."""


class Offloader(Protocol):
    """Plans Units, materializes the device side and owns the service-time runtime."""

    def plan(self, catalog: ComponentCatalog, topology: Topology, policy: OffloadPolicy | None) -> PlacementPlan:
        """Pure planning step: no large allocations, no model mutation, no hooks (step 4)."""

    def prepare_materialization(self, model: nn.Module, plan: PlacementPlan, storage: HostStorage) -> LoadTransaction:
        """Acquire every capacity-bearing runtime resource before the first write (step 8)."""

    def setup_compile(self, model: nn.Module, catalog: ComponentCatalog, runtime: OffloadRuntime) -> None:
        """Configure compilation targets after commit (step 12)."""


class LoadTransaction(Protocol):
    """Owner of the host storage and of every runtime resource until commit."""

    def target(self, tensor_key: str) -> torch.Tensor | None:
        """Write target for a host-resident tensor; None for device-resident or HSDP-managed tensors."""

    def commit(self, final_manifest: FinalManifest) -> OffloadRuntime:
        """Hand every resource over to the returned runtime; on failure resources stay here."""

    def abort(self) -> None:
        """Release everything held by the transaction in ledger order. Idempotent; no-op after commit."""


class OffloadRuntime(Protocol):
    """Service-time owner of the offload resources."""

    def close(self) -> None:
        """Close in RFC 04 section 7.5 order; no CPU restore, no collectives. Idempotent."""

    def grad_mode(self) -> AbstractContextManager[None]:
        """Gradient mode used for forward passes."""

    def execution(self) -> AbstractContextManager[None]:
        """Wrap one forward pass; re-entrant or concurrent use raises."""

    def before_sleep(self) -> None:
        """Called before the worker sleeps."""

    def after_wake(self) -> None:
        """Called after the worker wakes up."""


class HostWeightManager(Protocol):
    """Owns host records, sharing, elections, admission and host storage allocation."""

    def local_record(self) -> LocalRecord:
        """This rank's node-local record."""

    def deployment_id(self) -> tuple[str, bool]:
        """(deployment id, whether it was issued by the StageRuntime)."""

    def layout(self, plan: PlacementPlan, expected_manifest: ExpectedManifest) -> HostLayout:
        """Compute the host layout; its plan_fingerprint digests the node-local backend part."""

    def share_plan(self, topology: Topology, fingerprints: Fingerprints) -> SharePlan:
        """Compute the share sets and their representatives; pure function, identical on every rank."""

    def elect(self, share_plan: SharePlan) -> Election:
        """Every rank calls this; a rank that represents no share set returns an empty election."""

    def admit(self, topology: Topology, share_plan: SharePlan, elections: Elections, layout: HostLayout) -> None:
        """Pure admission check; reads elections.records only."""

    def allocate(self, layout: HostLayout, share_plan: SharePlan, elections: Elections) -> HostStorage:
        """Allocate host storage; on success it takes over elections.local's socket and listener thread."""


class HostStorage(Protocol):
    """Host-side storage: private memory, shared mappings or rank shards."""

    def view(self, tensor_key: str) -> torch.Tensor:
        """Write target (or read-only view) for a tensor key."""

    def ranges(self, unit: Unit) -> list[MemoryRange]:
        """Byte ranges of one Unit in this process."""

    def seal(self, final_manifest: FinalManifest) -> None:
        """Finalize, seal and hand over the shared storage, then bind read-only mappings."""

    def connect_writer(self) -> None:
        """After confirmation point 2: waiters connect to the write process; no-op for non-waiters."""

    def note_written(self, tensor_key: str) -> None:
        """Called by the Loader once a host-resident tensor is fully written; no collectives."""

    def close(self) -> None:
        """Release this process's mappings. Idempotent."""
