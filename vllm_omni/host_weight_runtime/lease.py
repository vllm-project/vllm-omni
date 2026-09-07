# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Process-local ownership for immutable mmap-backed host tensors."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import NoReturn, SupportsIndex

import torch

from .identity import WeightArtifactIdentity
from .manifest import WeightManifest


@dataclass(frozen=True)
class MappedHostRegion:
    """One process-local virtual range belonging to an artifact payload file."""

    file_name: str
    address: int
    length: int
    read_only: bool = True

    def __post_init__(self) -> None:
        if not self.file_name or self.address < 0 or self.length < 0:
            raise ValueError("mapped host region contains invalid fields")


@dataclass(frozen=True)
class LeaseProvenance:
    resolution_id: str
    identity_digest: str
    artifact_content_sha256: str
    domain_uuid: str
    publication_id: str
    resolution_source: str
    artifact_origin: str
    producer_id: str | None
    provider_id: str | None
    validation_level: str
    backing_fingerprint: str


class HostWeightLease:
    """Stable tensor views and resources retained until transport teardown."""

    def __init__(
        self,
        *,
        identity: WeightArtifactIdentity,
        manifest: WeightManifest,
        tensors: Mapping[str, torch.Tensor],
        mapped_regions: tuple[MappedHostRegion, ...],
        provenance: LeaseProvenance,
        resource_closers: tuple[Callable[[], None], ...],
        release_artifact_lock: Callable[[], None],
    ) -> None:
        if manifest.identity.canonical_bytes != identity.canonical_bytes:
            raise ValueError("lease manifest identity differs from the requested identity")
        if set(tensors) != {entry.name for entry in manifest.tensors}:
            raise ValueError("lease tensor names differ from the validated manifest")
        entries = {entry.name: entry for entry in manifest.tensors}
        for name, tensor in tensors.items():
            entry = entries[name]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tuple(tensor.shape) != entry.shape
                or str(tensor.dtype) != entry.dtype
                or tuple(tensor.stride()) != entry.stride
                or tensor.numel() * tensor.element_size() != entry.nbytes
            ):
                raise ValueError(f"lease tensor {name!r} differs from the validated host metadata")
        if provenance.identity_digest != identity.key:
            raise ValueError("lease provenance identity digest differs from the requested identity")
        if provenance.artifact_content_sha256 != manifest.artifact_content_sha256:
            raise ValueError("lease provenance content digest differs from the validated manifest")
        if not isinstance(mapped_regions, tuple) or any(
            not isinstance(region, MappedHostRegion) for region in mapped_regions
        ):
            raise ValueError("lease mapped regions must be an immutable tuple")
        if not isinstance(resource_closers, tuple) or any(not callable(closer) for closer in resource_closers):
            raise ValueError("lease resource closers must be an immutable tuple of callables")
        if not callable(release_artifact_lock):
            raise ValueError("lease artifact lock release must be callable")
        self.identity = identity
        self.manifest = manifest
        self._tensor_storage = dict(tensors)
        self.tensors = MappingProxyType(self._tensor_storage)
        self.mapped_regions = mapped_regions
        self.provenance = provenance
        self._resource_closers = resource_closers
        self._release_artifact_lock = release_artifact_lock
        self._state_lock = threading.RLock()
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._state_lock:
            return self._closed

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True

            errors: list[BaseException] = []
            self._tensor_storage.clear()
            for closer in reversed(self._resource_closers):
                try:
                    closer()
                except BaseException as exc:  # pragma: no cover - best-effort teardown aggregation
                    errors.append(exc)
            try:
                self._release_artifact_lock()
            except BaseException as exc:  # pragma: no cover - best-effort teardown aggregation
                errors.append(exc)
            if errors:
                raise RuntimeError(f"failed to close {len(errors)} host weight lease resources") from errors[0]

    def __enter__(self) -> HostWeightLease:
        if self.closed:
            raise RuntimeError("cannot re-enter a closed HostWeightLease")
        return self

    def __exit__(self, exc_type: object, exc: object, _traceback: object) -> None:
        self.close()

    def __reduce_ex__(self, _protocol: SupportsIndex) -> NoReturn:
        raise TypeError("HostWeightLease is process-local and cannot be serialized")

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


class HostWeightLeaseCarrier:
    """Process-local, single-take ownership carrier for a host-weight lease.

    The loader creates the carrier after a restore transaction has committed.
    The runner/backend takes the lease exactly once before starting any
    asynchronous transport work.  If setup fails before that handoff, closing
    the carrier releases the loader-owned lease.
    """

    def __init__(self, lease: HostWeightLease) -> None:
        if not isinstance(lease, HostWeightLease):
            raise TypeError("HostWeightLeaseCarrier requires one HostWeightLease")
        if lease.closed:
            raise ValueError("cannot carry a closed HostWeightLease")
        self._lease: HostWeightLease | None = lease
        self._taken = False
        self._lock = threading.RLock()

    @property
    def taken(self) -> bool:
        with self._lock:
            return self._taken

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._lease is None or self._lease.closed

    def take(self) -> HostWeightLease:
        """Transfer the lease to the transport owner exactly once."""
        with self._lock:
            if self._taken:
                raise RuntimeError("HostWeightLeaseCarrier was already taken")
            lease = self._lease
            if lease is None or lease.closed:
                raise RuntimeError("HostWeightLeaseCarrier does not own an open lease")
            self._lease = None
            self._taken = True
            return lease

    def close(self) -> None:
        """Close the lease if ownership has not been transferred."""
        with self._lock:
            lease = self._lease
            self._lease = None
        if lease is not None:
            lease.close()

    def __reduce_ex__(self, _protocol: SupportsIndex) -> NoReturn:
        raise TypeError("HostWeightLeaseCarrier is process-local and cannot be serialized")

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


__all__ = ["HostWeightLease", "HostWeightLeaseCarrier", "LeaseProvenance", "MappedHostRegion"]
