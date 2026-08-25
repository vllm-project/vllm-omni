# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Crash-safe node-local filesystem implementation of HostWeightStore."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import shutil
import stat
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import regex as re
import torch
from safetensors import SafetensorError, safe_open

from ..config import CapacityPolicy, IntegrityPolicy, StorageClass, StorageDomainPolicy, ValidationLevel
from ..errors import FailureCode, HostWeightError, HostWeightFailure, ResolutionStage
from ..identity import CanonicalJson, WeightArtifactIdentity, canonical_json
from ..inspection import (
    ArtifactInspection,
    ArtifactInventoryEntry,
    ArtifactInventoryState,
    DomainInspection,
)
from ..lease import HostWeightLease, LeaseProvenance, MappedHostRegion
from ..manifest import ManifestValidationError, PublicationMarker, TensorManifestEntry, WeightManifest
from ..outcomes import StoreResult, StoreStatus
from ..protocols import BuildRequest, CoordinationScope, WeightProducer
from .locks import FileLock, FileLockTimeoutError, lock_is_active
from .writer import FilesystemArtifactWriter

_DOMAIN_FILE = "domain.json"
_DOMAIN_POLICY_FILE = "domain-policy.json"
_MANIFEST_FILE = "manifest.json"
_READY_FILE = "READY.json"
_MAX_METADATA_BYTES = 64 * 1024**2
_FILE_HASH_CHUNK_BYTES = 8 * 1024**2
_ARTIFACT_KEY_RE = re.compile(r"[0-9a-f]{64}")
_LOCAL_DISK_FILESYSTEM_TYPES = frozenset(
    {
        "aufs",
        "bcachefs",
        "btrfs",
        "erofs",
        "ext2",
        "ext3",
        "ext4",
        "f2fs",
        "fuseblk",
        "jfs",
        "ntfs3",
        "overlay",
        "reiserfs",
        "rootfs",
        "xfs",
        "zfs",
    }
)
_REMOTE_FILESYSTEM_TYPES = frozenset(
    {
        "9p",
        "afs",
        "beegfs",
        "ceph",
        "cifs",
        "fuse.gcsfuse",
        "fuse.s3fs",
        "fuse.sshfs",
        "gcsfuse",
        "glusterfs",
        "gpfs",
        "lustre",
        "nfs",
        "nfs4",
        "orangefs",
        "panfs",
        "s3fs",
        "smb3",
        "sshfs",
        "virtiofs",
        "wekafs",
    }
)


class _ExitContext(Protocol):
    def __exit__(self, exc_type: None, exc: None, _traceback: None) -> object: ...


def _make_fd_closer(fd: int) -> Callable[[], None]:
    def close() -> None:
        os.close(fd)

    return close


def _make_context_closer(context: _ExitContext) -> Callable[[], None]:
    def close() -> None:
        context.__exit__(None, None, None)

    return close


def _failure(
    stage: ResolutionStage,
    code: FailureCode,
    message: str,
    *,
    retryable: bool = False,
    details: object | None = None,
) -> HostWeightFailure:
    return HostWeightFailure(
        stage=stage,
        code=code,
        retryable=retryable,
        message=message,
        details=CanonicalJson.from_value({} if details is None else details),
    )


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_atomic_json(path: Path, value: object, *, mode: int = 0o444) -> None:
    data = canonical_json(value)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_CLOEXEC, 0o600)
    try:
        offset = 0
        while offset < len(data):
            count = os.write(fd, data[offset:])
            if count <= 0:
                raise OSError(errno.EIO, f"short write for {temporary}")
            offset += count
        os.fsync(fd)
        os.fchmod(fd, mode)
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        temporary.unlink(missing_ok=True)
        raise
    else:
        os.close(fd)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _read_json_file(path: Path) -> dict[str, object]:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        return _read_json_fd(fd, path.name)
    finally:
        os.close(fd)


def _read_json_at(directory_fd: int, name: str, *, require_read_only: bool = False) -> dict[str, object]:
    fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=directory_fd)
    try:
        return _read_json_fd(fd, name, require_read_only=require_read_only)
    finally:
        os.close(fd)


def _read_json_fd(fd: int, name: str, *, require_read_only: bool = False) -> dict[str, object]:
    file_stat = os.fstat(fd)
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
        raise ManifestValidationError(f"metadata file {name!r} is not one regular file")
    if file_stat.st_size > _MAX_METADATA_BYTES:
        raise ManifestValidationError(f"metadata file {name!r} exceeds {_MAX_METADATA_BYTES} bytes")
    if require_read_only and (file_stat.st_uid != os.geteuid() or file_stat.st_mode & 0o222):
        raise ManifestValidationError(f"published metadata file {name!r} is not immutable")
    chunks: list[bytes] = []
    offset = 0
    while offset < file_stat.st_size:
        chunk = os.pread(fd, min(1024**2, file_stat.st_size - offset), offset)
        if not chunk:
            raise ManifestValidationError(f"metadata file {name!r} ended unexpectedly")
        chunks.append(chunk)
        offset += len(chunk)
    try:
        value = json.loads(b"".join(chunks))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(f"metadata file {name!r} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ManifestValidationError(f"metadata file {name!r} must contain an object")
    return value


def _sha256_fd(fd: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = os.pread(fd, min(_FILE_HASH_CHUNK_BYTES, size - offset), offset)
        if not chunk:
            raise OSError(errno.EIO, "artifact file ended while hashing")
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _mount_unescape(value: str) -> str:
    return value.replace("\\040", " ").replace("\\011", "\t").replace("\\012", "\n").replace("\\134", "\\")


def detect_filesystem_type(path: Path) -> str:
    """Return the kernel filesystem type for the most specific mount."""
    resolved = path.resolve()
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():  # pragma: no cover - vLLM-Omni production is Linux
        raise ValueError("cannot verify filesystem locality without /proc/self/mountinfo")
    best_length = -1
    best_type = ""
    for line in mountinfo.read_text(encoding="utf-8").splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        left_fields = left.split()
        right_fields = right.split()
        if len(left_fields) < 5 or not right_fields:
            continue
        mount_point = Path(_mount_unescape(left_fields[4]))
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        if len(str(mount_point)) > best_length:
            best_length = len(str(mount_point))
            best_type = right_fields[0]
    if not best_type:
        raise ValueError(f"cannot identify the filesystem containing {resolved}")
    return best_type


def _storage_class_for_filesystem_type(filesystem_type: str) -> StorageClass:
    if filesystem_type in {"tmpfs", "ramfs"}:
        return StorageClass.TMPFS
    if filesystem_type in _LOCAL_DISK_FILESYSTEM_TYPES:
        return StorageClass.DISK
    if filesystem_type in _REMOTE_FILESYSTEM_TYPES:
        raise ValueError(f"filesystem type {filesystem_type!r} is not node-local")
    raise ValueError(f"filesystem type {filesystem_type!r} is not recognized as a local filesystem")


def detect_storage_class(path: Path) -> StorageClass:
    """Classify a verified node-local mount without spawning platform tools."""
    return _storage_class_for_filesystem_type(detect_filesystem_type(path))


class FilesystemHostWeightStore:
    """A cooperative local store with stable locks and immutable publication."""

    def __init__(
        self,
        domain: StorageDomainPolicy,
        capacity: CapacityPolicy,
        integrity: IntegrityPolicy,
    ) -> None:
        self.domain_policy = domain
        self.capacity_policy = capacity
        self.integrity_policy = integrity
        self.root = domain.root.expanduser().resolve()
        self.locks_dir = self.root / "locks"
        self.artifacts_dir = self.root / "artifacts"
        self.tmp_dir = self.root / "tmp"
        self.quarantine_dir = self.root / "quarantine"
        self.deny_dir = self.root / "deny"
        if integrity.local_lookup is ValidationLevel.FS_VERITY:
            raise HostWeightError(
                _failure(
                    ResolutionStage.CONFIGURATION,
                    FailureCode.UNSUPPORTED,
                    "filesystem-verity validation is not implemented",
                )
            )
        if domain.numa_prefault or domain.verify_numa_residency:
            raise HostWeightError(
                _failure(
                    ResolutionStage.CONFIGURATION,
                    FailureCode.UNSUPPORTED,
                    "enforced NUMA placement and residency verification are not implemented",
                )
            )
        if integrity.background_scrub:
            raise HostWeightError(
                _failure(
                    ResolutionStage.CONFIGURATION,
                    FailureCode.UNSUPPORTED,
                    "background integrity scrubbing is not implemented",
                )
            )
        try:
            self._initialize_domain()
        except HostWeightError:
            raise
        except OSError as exc:
            raise HostWeightError(
                _failure(
                    ResolutionStage.DOMAIN,
                    FailureCode.DOMAIN_UNAVAILABLE,
                    f"cannot initialize storage domain {self.root}: {exc}",
                    retryable=True,
                )
            ) from exc
        except (ValueError, ManifestValidationError) as exc:
            raise HostWeightError(
                _failure(
                    ResolutionStage.CONFIGURATION,
                    FailureCode.DOMAIN_POLICY_MISMATCH,
                    f"storage domain metadata is invalid under {self.root}: {exc}",
                )
            ) from exc

    def _initialize_domain(self) -> None:
        try:
            self.filesystem_type = detect_filesystem_type(self.root)
            detected_storage_class = _storage_class_for_filesystem_type(self.filesystem_type)
        except ValueError as exc:
            raise HostWeightError(
                _failure(
                    ResolutionStage.CONFIGURATION,
                    FailureCode.DOMAIN_POLICY_MISMATCH,
                    f"local filesystem storage is unavailable for {self.root}: {exc}",
                )
            ) from exc
        if detected_storage_class is not self.domain_policy.storage_class:
            raise HostWeightError(
                _failure(
                    ResolutionStage.DOMAIN,
                    FailureCode.STORAGE_CLASS_MISMATCH,
                    "configured storage class "
                    f"{self.domain_policy.storage_class.value} differs from detected {detected_storage_class.value}",
                )
            )

        try:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.locks_dir.mkdir(exist_ok=True, mode=0o700)
            for directory in (self.artifacts_dir, self.tmp_dir, self.quarantine_dir, self.deny_dir):
                directory.mkdir(exist_ok=True, mode=0o700)
        except OSError as exc:
            raise HostWeightError(
                _failure(
                    ResolutionStage.DOMAIN,
                    FailureCode.DOMAIN_UNAVAILABLE,
                    f"cannot initialize {self.root}: {exc}",
                    retryable=True,
                )
            ) from exc

        if self.integrity_policy.require_trusted_permissions:
            for directory in (
                self.root,
                self.locks_dir,
                self.artifacts_dir,
                self.tmp_dir,
                self.quarantine_dir,
                self.deny_dir,
            ):
                directory_stat = directory.lstat()
                if (
                    not stat.S_ISDIR(directory_stat.st_mode)
                    or directory_stat.st_uid != os.geteuid()
                    or directory_stat.st_mode & 0o022
                ):
                    raise HostWeightError(
                        _failure(
                            ResolutionStage.DOMAIN,
                            FailureCode.DOMAIN_UNAVAILABLE,
                            f"host weight store directory is not exclusively controlled by this user: {directory}",
                        )
                    )

        with FileLock(self.locks_dir / "domain-init.lock", exclusive=True, deadline=None):
            domain_path = self.root / _DOMAIN_FILE
            if domain_path.exists():
                domain_metadata = _read_json_file(domain_path)
                expected = {
                    "schema_version": 1,
                    "scope": self.domain_policy.scope.value,
                    "domain_id": self.domain_policy.domain_id,
                    "backend": "local_filesystem",
                    "storage_class": self.domain_policy.storage_class.value,
                    "filesystem_type": self.filesystem_type,
                }
                if set(domain_metadata) != {*expected, "domain_uuid"}:
                    raise HostWeightError(
                        _failure(
                            ResolutionStage.CONFIGURATION,
                            FailureCode.DOMAIN_POLICY_MISMATCH,
                            f"storage domain metadata has an incompatible schema under {self.root}",
                        )
                    )
                actual = {key: domain_metadata.get(key) for key in expected}
                domain_uuid = domain_metadata.get("domain_uuid")
                if actual != expected or not isinstance(domain_uuid, str) or not domain_uuid:
                    raise HostWeightError(
                        _failure(
                            ResolutionStage.CONFIGURATION,
                            FailureCode.DOMAIN_POLICY_MISMATCH,
                            f"storage domain metadata does not match configured policy under {self.root}",
                        )
                    )
            else:
                domain_metadata = {
                    "schema_version": 1,
                    "domain_uuid": uuid.uuid4().hex,
                    "scope": self.domain_policy.scope.value,
                    "domain_id": self.domain_policy.domain_id,
                    "backend": "local_filesystem",
                    "storage_class": self.domain_policy.storage_class.value,
                    "filesystem_type": self.filesystem_type,
                }
                _write_atomic_json(domain_path, domain_metadata)
            self.domain_uuid = str(domain_metadata["domain_uuid"])

            policy_path = self.root / _DOMAIN_POLICY_FILE
            requested_policy = self._capacity_document(policy_version=1)
            if policy_path.exists():
                actual_policy = _read_json_file(policy_path)
                if set(actual_policy) != set(requested_policy):
                    raise HostWeightError(
                        _failure(
                            ResolutionStage.CONFIGURATION,
                            FailureCode.DOMAIN_POLICY_MISMATCH,
                            f"capacity policy has an incompatible schema under {self.root}",
                        )
                    )
                comparable_actual = {key: actual_policy.get(key) for key in requested_policy if key != "policy_version"}
                comparable_requested = {
                    key: value for key, value in requested_policy.items() if key != "policy_version"
                }
                if comparable_actual != comparable_requested:
                    raise HostWeightError(
                        _failure(
                            ResolutionStage.CONFIGURATION,
                            FailureCode.DOMAIN_POLICY_MISMATCH,
                            f"capacity policy differs from the authoritative policy under {self.root}",
                        )
                    )
                policy_version = actual_policy.get("policy_version")
                if isinstance(policy_version, bool) or not isinstance(policy_version, int) or policy_version < 1:
                    raise HostWeightError(
                        _failure(
                            ResolutionStage.CONFIGURATION,
                            FailureCode.DOMAIN_POLICY_MISMATCH,
                            f"capacity policy has an invalid version under {self.root}",
                        )
                    )
                self.policy_version = policy_version
            else:
                _write_atomic_json(policy_path, requested_policy)
                self.policy_version = 1

    def _capacity_document(self, *, policy_version: int) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy_version": policy_version,
            "max_artifact_bytes": self.capacity_policy.max_artifact_bytes,
            "max_store_bytes": self.capacity_policy.max_store_bytes,
            "min_free_bytes": self.capacity_policy.min_free_bytes,
            "eviction": self.capacity_policy.eviction,
        }

    def _build_lock_path(self, key: str) -> Path:
        return self.locks_dir / f"{key}.build.lock"

    def _artifact_lock_path(self, key: str) -> Path:
        return self.locks_dir / f"{key}.artifact.lock"

    def _entry_path(self, key: str) -> Path:
        return self.artifacts_dir / key

    def _deny_path(self, key: str) -> Path:
        return self.deny_dir / f"{key}.json"

    def lookup(
        self,
        identity: WeightArtifactIdentity,
        *,
        validation: ValidationLevel,
        deadline: float | None = None,
    ) -> StoreResult:
        return self._lookup(identity, validation=validation, deadline=deadline, record_invalid=True)

    def _lookup(
        self,
        identity: WeightArtifactIdentity,
        *,
        validation: ValidationLevel,
        deadline: float | None,
        record_invalid: bool,
    ) -> StoreResult:
        if validation is ValidationLevel.FS_VERITY:
            return StoreResult(
                StoreStatus.FAILED,
                failure=_failure(
                    ResolutionStage.VALIDATION,
                    FailureCode.UNSUPPORTED,
                    "filesystem-verity validation is modeled but not implemented",
                ),
            )
        try:
            artifact_lock = FileLock(self._artifact_lock_path(identity.key), exclusive=False, deadline=deadline)
        except FileLockTimeoutError as exc:
            return StoreResult(
                StoreStatus.TIMEOUT,
                failure=_failure(
                    ResolutionStage.LOOKUP,
                    FailureCode.ACTIVE_BUILD_TIMEOUT,
                    str(exc),
                    retryable=True,
                ),
            )
        except OSError as exc:
            return StoreResult(
                StoreStatus.FAILED,
                failure=_failure(ResolutionStage.DOMAIN, FailureCode.DOMAIN_UNAVAILABLE, str(exc), retryable=True),
            )

        entry_path = self._entry_path(identity.key)
        if not entry_path.exists():
            artifact_lock.close()
            return StoreResult(StoreStatus.MISS)
        if self._deny_path(identity.key).exists():
            artifact_lock.close()
            return StoreResult(
                StoreStatus.INVALID,
                failure=_failure(
                    ResolutionStage.LOOKUP,
                    FailureCode.ARTIFACT_DENIED,
                    f"artifact {identity.key} is denied pending quarantine",
                ),
            )
        try:
            lease = self._open_lease(identity, entry_path, artifact_lock, validation)
            if self._deny_path(identity.key).exists():
                lease.close()
                return StoreResult(
                    StoreStatus.INVALID,
                    failure=_failure(
                        ResolutionStage.LOOKUP,
                        FailureCode.ARTIFACT_DENIED,
                        f"artifact {identity.key} was denied during lookup",
                    ),
                )
            return StoreResult(StoreStatus.HIT, lease=lease)
        except HostWeightError as exc:
            try:
                if record_invalid:
                    with contextlib.suppress(OSError):
                        self._write_deny(identity.key, exc.failure)
            finally:
                artifact_lock.close()
            return StoreResult(StoreStatus.INVALID, failure=exc.failure)
        except (OSError, SafetensorError, ValueError, ManifestValidationError) as exc:
            failure = _failure(
                ResolutionStage.VALIDATION,
                FailureCode.MANIFEST_CORRUPT,
                f"artifact {identity.key} failed validation: {exc}",
            )
            try:
                if record_invalid:
                    with contextlib.suppress(OSError):
                        self._write_deny(identity.key, failure)
            finally:
                artifact_lock.close()
            return StoreResult(StoreStatus.INVALID, failure=failure)

    def _open_lease(
        self,
        identity: WeightArtifactIdentity,
        entry_path: Path,
        artifact_lock: FileLock,
        validation: ValidationLevel,
    ) -> HostWeightLease:
        resources: list[Callable[[], None]] = []
        try:
            entry_fd = os.open(entry_path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
            resources.append(_make_fd_closer(entry_fd))
            entry_stat = os.fstat(entry_fd)
            if self.integrity_policy.require_trusted_permissions and (
                entry_stat.st_uid != os.geteuid() or entry_stat.st_mode & 0o222
            ):
                raise ManifestValidationError("published artifact directory is not immutable")
            names = set(os.listdir(entry_fd))
            marker_document = _read_json_at(
                entry_fd,
                _READY_FILE,
                require_read_only=self.integrity_policy.require_trusted_permissions,
            )
            manifest_document = _read_json_at(
                entry_fd,
                _MANIFEST_FILE,
                require_read_only=self.integrity_policy.require_trusted_permissions,
            )
            marker = PublicationMarker.from_dict(marker_document)
            manifest = WeightManifest.from_dict(manifest_document)
            if marker.schema_version != 1 or manifest.schema_version != 1 or marker.integrity_mode != "full_checksum":
                raise HostWeightError(
                    _failure(
                        ResolutionStage.VALIDATION,
                        FailureCode.MANIFEST_INCOMPATIBLE,
                        "artifact publication schema or integrity mode is unsupported",
                    )
                )
            expected_names = {_READY_FILE, _MANIFEST_FILE, *(entry.file_name for entry in manifest.files)}
            if names != expected_names:
                raise ManifestValidationError(
                    f"artifact directory entries differ: missing={sorted(expected_names - names)}, "
                    f"unexpected={sorted(names - expected_names)}"
                )
            manifest_digest = hashlib.sha256(manifest.canonical_bytes).hexdigest()
            if marker.manifest_sha256 != manifest_digest:
                raise HostWeightError(
                    _failure(ResolutionStage.VALIDATION, FailureCode.CHECKSUM_MISMATCH, "manifest digest mismatch")
                )
            if marker.artifact_content_sha256 != manifest.artifact_content_sha256:
                raise HostWeightError(
                    _failure(ResolutionStage.VALIDATION, FailureCode.CHECKSUM_MISMATCH, "content digest mismatch")
                )
            if manifest.artifact_key != identity.key or manifest.identity.canonical_bytes != identity.canonical_bytes:
                raise HostWeightError(
                    _failure(
                        ResolutionStage.VALIDATION,
                        FailureCode.MANIFEST_INCOMPATIBLE,
                        "artifact semantic identity differs from the exact request",
                    )
                )
            if (
                manifest.producer_schema != identity.producer.manifest_schema
                or manifest.restorer_schema != identity.producer.restorer_schema
            ):
                raise HostWeightError(
                    _failure(
                        ResolutionStage.VALIDATION,
                        FailureCode.MANIFEST_INCOMPATIBLE,
                        "artifact producer or restorer schema differs from the exact request",
                    )
                )

            tensors_by_file: dict[str, list[TensorManifestEntry]] = defaultdict(list)
            for tensor_entry in manifest.tensors:
                tensors_by_file[tensor_entry.file_name].append(tensor_entry)
            tensors: dict[str, torch.Tensor] = {}
            regions: list[MappedHostRegion] = []
            backing_digest = hashlib.sha256()
            for file_entry in manifest.files:
                fd = os.open(
                    file_entry.file_name,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=entry_fd,
                )
                resources.append(_make_fd_closer(fd))
                file_stat = os.fstat(fd)
                if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
                    raise ManifestValidationError(f"payload {file_entry.file_name!r} is not one regular file")
                if file_stat.st_size != file_entry.size:
                    raise HostWeightError(
                        _failure(
                            ResolutionStage.VALIDATION,
                            FailureCode.CHECKSUM_MISMATCH,
                            f"payload size differs for {file_entry.file_name!r}",
                        )
                    )
                if self.integrity_policy.require_trusted_permissions and (
                    file_stat.st_uid != os.geteuid() or file_stat.st_mode & 0o222
                ):
                    raise ManifestValidationError(f"published payload {file_entry.file_name!r} is not immutable")
                if (
                    validation is ValidationLevel.FULL_CHECKSUM
                    and _sha256_fd(fd, file_stat.st_size) != file_entry.sha256
                ):
                    raise HostWeightError(
                        _failure(
                            ResolutionStage.VALIDATION,
                            FailureCode.CHECKSUM_MISMATCH,
                            f"payload digest differs for {file_entry.file_name!r}",
                        )
                    )
                backing_digest.update(
                    canonical_json(
                        {
                            "file": file_entry.file_name,
                            "device": file_stat.st_dev,
                            "inode": file_stat.st_ino,
                            "size": file_stat.st_size,
                        }
                    )
                )
                handle_context = safe_open(f"/proc/self/fd/{fd}", framework="pt", device="cpu")
                handle = handle_context.__enter__()
                resources.append(_make_context_closer(handle_context))
                expected_entries = tensors_by_file[file_entry.file_name]
                if set(handle.keys()) != {entry.storage_key for entry in expected_entries}:
                    raise ManifestValidationError(f"payload tensor keys differ for {file_entry.file_name!r}")
                addresses: list[tuple[int, int]] = []
                for tensor_entry in expected_entries:
                    tensor = handle.get_tensor(tensor_entry.storage_key)
                    actual_nbytes = tensor.numel() * tensor.element_size()
                    if (
                        tuple(tensor.shape) != tensor_entry.shape
                        or str(tensor.dtype) != tensor_entry.dtype
                        or tuple(tensor.stride()) != tensor_entry.stride
                        or actual_nbytes != tensor_entry.nbytes
                        or tensor.device.type != "cpu"
                    ):
                        raise ManifestValidationError(f"mapped tensor metadata differs for {tensor_entry.name!r}")
                    tensors[tensor_entry.name] = tensor
                    if actual_nbytes:
                        addresses.append((tensor.data_ptr(), tensor.data_ptr() + actual_nbytes))
                if addresses:
                    merged: list[list[int]] = []
                    for start, end in sorted(addresses):
                        if merged and start <= merged[-1][1]:
                            merged[-1][1] = max(merged[-1][1], end)
                        else:
                            merged.append([start, end])
                    regions.extend(
                        MappedHostRegion(
                            file_name=file_entry.file_name,
                            address=start,
                            length=end - start,
                        )
                        for start, end in merged
                    )

            backing_digest.update(self.domain_uuid.encode("ascii"))
            backing_digest.update(marker.publication_id.encode("ascii"))
            provenance = LeaseProvenance(
                resolution_id=uuid.uuid4().hex,
                identity_digest=identity.key,
                artifact_content_sha256=manifest.artifact_content_sha256,
                domain_uuid=self.domain_uuid,
                publication_id=marker.publication_id,
                resolution_source="local",
                artifact_origin="producer",
                producer_id=identity.producer.producer_id,
                provider_id=None,
                validation_level=validation.value,
                backing_fingerprint=backing_digest.hexdigest(),
            )
            return HostWeightLease(
                identity=identity,
                manifest=manifest,
                tensors=tensors,
                mapped_regions=tuple(regions),
                provenance=provenance,
                resource_closers=tuple(resources),
                release_artifact_lock=artifact_lock.close,
            )
        except BaseException:
            for closer in reversed(resources):
                with contextlib.suppress(Exception):
                    closer()
            raise

    def get_or_build(
        self,
        request: BuildRequest,
        producer: WeightProducer,
        *,
        validation: ValidationLevel,
        deadline: float,
    ) -> StoreResult:
        identity = request.identity
        if producer.spec.coordination_scope is not CoordinationScope.SINGLE_PROCESS:
            return StoreResult(
                StoreStatus.FAILED,
                failure=_failure(
                    ResolutionStage.PRODUCTION,
                    FailureCode.PRODUCER_UNSUPPORTED,
                    f"coordination scope {producer.spec.coordination_scope.value} is not implemented",
                ),
            )
        if tuple(output.key for output in producer.spec.outputs) != (identity.key,):
            return StoreResult(
                StoreStatus.FAILED,
                failure=_failure(
                    ResolutionStage.PRODUCTION,
                    FailureCode.PRODUCER_UNAVAILABLE,
                    "producer does not declare exactly the requested artifact identity",
                ),
            )

        initial = self.lookup(identity, validation=validation, deadline=deadline)
        if initial.status is StoreStatus.HIT:
            return initial
        if initial.status in {StoreStatus.TIMEOUT, StoreStatus.FAILED}:
            return initial

        try:
            build_lock = FileLock(self._build_lock_path(identity.key), exclusive=True, deadline=deadline)
        except FileLockTimeoutError as exc:
            return StoreResult(
                StoreStatus.TIMEOUT,
                failure=_failure(
                    ResolutionStage.PRODUCTION,
                    FailureCode.ACTIVE_BUILD_TIMEOUT,
                    str(exc),
                    retryable=True,
                ),
            )
        except OSError as exc:
            return StoreResult(
                StoreStatus.FAILED,
                failure=_failure(ResolutionStage.DOMAIN, FailureCode.DOMAIN_UNAVAILABLE, str(exc), retryable=True),
            )

        with build_lock:
            recheck = self.lookup(identity, validation=validation, deadline=deadline)
            if recheck.status is StoreStatus.HIT:
                return StoreResult(StoreStatus.JOINED, lease=recheck.lease)
            if recheck.status in {StoreStatus.TIMEOUT, StoreStatus.FAILED}:
                return recheck

            try:
                with FileLock(self._artifact_lock_path(identity.key), exclusive=True, deadline=deadline):
                    if self._entry_path(identity.key).exists():
                        self._quarantine_locked(identity.key, reason="validation_failure")
                    self._remove_deny_locked(identity.key)
                    self._remove_stale_temps_locked(identity.key)
            except FileLockTimeoutError as exc:
                return StoreResult(
                    StoreStatus.TIMEOUT,
                    failure=_failure(
                        ResolutionStage.PRODUCTION,
                        FailureCode.ACTIVE_BUILD_TIMEOUT,
                        str(exc),
                        retryable=True,
                    ),
                )
            except OSError as exc:
                return StoreResult(
                    StoreStatus.FAILED,
                    failure=_failure(
                        ResolutionStage.LIFECYCLE,
                        FailureCode.QUARANTINE_FAILED,
                        f"failed to prepare artifact rebuild: {exc}",
                    ),
                )

            staging = self.tmp_dir / f"{identity.key}.{os.getpid()}.{uuid.uuid4().hex}"
            writer: FilesystemArtifactWriter | None = None
            manifest: WeightManifest | None = None
            try:
                staging.mkdir(mode=0o700)
                writer = FilesystemArtifactWriter(staging, capacity_check=self._check_capacity)
                metadata = producer.produce(writer)
                manifest, _ = writer.finalize(identity, metadata)
                artifact_bytes = self._tree_bytes(staging)
                self._check_capacity(artifact_bytes, 0)
                os.chmod(staging, 0o555)
                _fsync_directory(staging)
            except HostWeightError as exc:
                if writer is not None:
                    writer.abort()
                self._discard_temp(identity.key, staging)
                return StoreResult(StoreStatus.FAILED, failure=exc.failure)
            except OSError as exc:
                if writer is not None:
                    writer.abort()
                self._discard_temp(identity.key, staging)
                if exc.errno in (errno.ENOSPC, errno.EDQUOT):
                    code = FailureCode.ENOSPC
                    stage = ResolutionStage.CAPACITY
                elif writer is None:
                    code = FailureCode.DOMAIN_UNAVAILABLE
                    stage = ResolutionStage.DOMAIN
                else:
                    code = FailureCode.PRODUCER_FAILED
                    stage = ResolutionStage.PRODUCTION
                return StoreResult(
                    StoreStatus.FAILED,
                    failure=_failure(
                        stage,
                        code,
                        f"artifact production failed: {exc}",
                        retryable=stage is ResolutionStage.CAPACITY,
                    ),
                )
            except Exception as exc:
                if writer is not None:
                    writer.abort()
                self._discard_temp(identity.key, staging)
                return StoreResult(
                    StoreStatus.FAILED,
                    failure=_failure(
                        ResolutionStage.PRODUCTION,
                        FailureCode.PRODUCER_FAILED,
                        f"artifact producer failed: {exc}",
                    ),
                )

            assert manifest is not None
            try:
                with FileLock(self._artifact_lock_path(identity.key), exclusive=True, deadline=None):
                    entry_path = self._entry_path(identity.key)
                    if entry_path.exists():
                        existing = WeightManifest.from_dict(_read_json_file(entry_path / _MANIFEST_FILE))
                        same_semantic_content = (
                            existing.identity.canonical_bytes == manifest.identity.canonical_bytes
                            and existing.artifact_content_sha256 == manifest.artifact_content_sha256
                            and existing.producer_schema == manifest.producer_schema
                            and existing.restorer_schema == manifest.restorer_schema
                            and existing.format_metadata == manifest.format_metadata
                        )
                        if not same_semantic_content:
                            collision = self.quarantine_dir / f"{identity.key}.identity-collision.{uuid.uuid4().hex}"
                            os.replace(staging, collision)
                            _fsync_directory(self.quarantine_dir)
                            _fsync_directory(self.tmp_dir)
                            return StoreResult(
                                StoreStatus.FAILED,
                                failure=_failure(
                                    ResolutionStage.VALIDATION,
                                    FailureCode.IDENTITY_COLLISION,
                                    "the same semantic identity produced different artifact content",
                                ),
                            )
                        self._remove_tree(staging)
                    else:
                        os.replace(staging, entry_path)
                        _fsync_directory(self.artifacts_dir)
                        _fsync_directory(self.tmp_dir)
                    self._remove_deny_locked(identity.key)
            except (OSError, ValueError, ManifestValidationError) as exc:
                self._discard_temp(identity.key, staging)
                return StoreResult(
                    StoreStatus.FAILED,
                    failure=_failure(
                        ResolutionStage.LIFECYCLE,
                        FailureCode.PUBLICATION_FAILED,
                        f"atomic artifact publication failed: {exc}",
                    ),
                )

            opened = self.lookup(identity, validation=validation, deadline=deadline)
            if opened.status is not StoreStatus.HIT:
                return opened
            return StoreResult(StoreStatus.BUILT, lease=opened.lease)

    def _check_capacity(self, projected_artifact_bytes: int, additional_bytes: int) -> None:
        policy = self.capacity_policy
        if policy.max_artifact_bytes is not None and projected_artifact_bytes > policy.max_artifact_bytes:
            raise HostWeightError(
                _failure(
                    ResolutionStage.CAPACITY,
                    FailureCode.ARTIFACT_TOO_LARGE,
                    f"projected artifact size {projected_artifact_bytes} exceeds {policy.max_artifact_bytes}",
                    retryable=True,
                )
            )
        store_bytes = self._tree_bytes(self.root)
        if policy.max_store_bytes is not None and store_bytes + additional_bytes > policy.max_store_bytes:
            raise HostWeightError(
                _failure(
                    ResolutionStage.CAPACITY,
                    FailureCode.STORE_LIMIT_EXCEEDED,
                    f"store size plus allocation would exceed {policy.max_store_bytes}",
                    retryable=True,
                )
            )
        free_bytes = shutil.disk_usage(self.root).free
        if free_bytes - additional_bytes < policy.min_free_bytes:
            raise HostWeightError(
                _failure(
                    ResolutionStage.CAPACITY,
                    FailureCode.INSUFFICIENT_CAPACITY,
                    f"allocation would leave less than {policy.min_free_bytes} free bytes",
                    retryable=True,
                )
            )

    def _write_deny(self, key: str, failure: HostWeightFailure) -> None:
        path = self._deny_path(key)
        document = {
            "schema_version": 1,
            "stage": failure.stage.value,
            "reason": failure.code.value,
            "message": failure.message,
        }
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_CLOEXEC, 0o444)
        except FileExistsError:
            return
        try:
            data = canonical_json(document)
            offset = 0
            while offset < len(data):
                written = os.write(fd, data[offset:])
                if written == 0:
                    raise OSError("short write while publishing deny marker")
                offset += written
            os.fsync(fd)
        finally:
            os.close(fd)
        _fsync_directory(self.deny_dir)

    def _remove_deny_locked(self, key: str) -> None:
        path = self._deny_path(key)
        if path.exists():
            path.unlink()
            _fsync_directory(self.deny_dir)

    def _quarantine_locked(self, key: str, *, reason: str) -> Path | None:
        entry = self._entry_path(key)
        if not entry.exists():
            return None
        destination = self.quarantine_dir / f"{key}.{reason}.{uuid.uuid4().hex}"
        os.replace(entry, destination)
        _fsync_directory(self.artifacts_dir)
        _fsync_directory(self.quarantine_dir)
        return destination

    def _remove_stale_temps_locked(self, key: str) -> None:
        for path in self.tmp_dir.glob(f"{key}.*"):
            self._remove_tree(path)
        _fsync_directory(self.tmp_dir)

    def _discard_temp(self, key: str, path: Path) -> None:
        with contextlib.suppress(OSError, FileLockTimeoutError):
            with FileLock(self._artifact_lock_path(key), exclusive=True, deadline=None):
                self._remove_tree(path)
                _fsync_directory(self.tmp_dir)

    @staticmethod
    def _remove_tree(path: Path) -> None:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.exists():
            for directory, subdirectories, _ in os.walk(path, topdown=False, followlinks=False):
                for name in subdirectories:
                    child = Path(directory) / name
                    if not child.is_symlink():
                        os.chmod(child, 0o700)
                os.chmod(directory, 0o700)
            shutil.rmtree(path)

    @staticmethod
    def _tree_bytes(root: Path) -> int:
        with contextlib.suppress(OSError):
            root_stat = root.lstat()
            if stat.S_ISREG(root_stat.st_mode):
                return root_stat.st_size
        total = 0
        for directory, _, files in os.walk(root, followlinks=False):
            for name in files:
                path = Path(directory) / name
                with contextlib.suppress(OSError):
                    file_stat = path.lstat()
                    if stat.S_ISREG(file_stat.st_mode):
                        total += file_stat.st_size
        return total

    def cleanup(self, identity: WeightArtifactIdentity) -> HostWeightFailure | None:
        """Explicitly remove an inactive artifact without waiting for leases."""
        try:
            build_lock = FileLock(
                self._build_lock_path(identity.key),
                exclusive=True,
                deadline=None,
                nonblocking=True,
            )
        except FileLockTimeoutError:
            return _failure(
                ResolutionStage.LIFECYCLE,
                FailureCode.ACTIVE_BUILD_TIMEOUT,
                f"artifact {identity.key} has an active builder",
                retryable=True,
            )
        with build_lock:
            try:
                artifact_lock = FileLock(
                    self._artifact_lock_path(identity.key),
                    exclusive=True,
                    deadline=None,
                    nonblocking=True,
                )
            except FileLockTimeoutError:
                return _failure(
                    ResolutionStage.LIFECYCLE,
                    FailureCode.ACTIVE_LEASE,
                    f"artifact {identity.key} has an active lease",
                    retryable=True,
                )
            with artifact_lock:
                entry = self._entry_path(identity.key)
                if entry.exists():
                    removal = self.quarantine_dir / f"{identity.key}.cleanup.{uuid.uuid4().hex}"
                    os.replace(entry, removal)
                    _fsync_directory(self.artifacts_dir)
                    self._remove_tree(removal)
                    _fsync_directory(self.quarantine_dir)
                self._remove_deny_locked(identity.key)
        return None

    def inspect_domain(self) -> DomainInspection:
        usage = shutil.disk_usage(self.root)
        inventory: list[ArtifactInventoryEntry] = []
        for state, directory in (
            (ArtifactInventoryState.READY, self.artifacts_dir),
            (ArtifactInventoryState.DENIED, self.deny_dir),
            (ArtifactInventoryState.QUARANTINED, self.quarantine_dir),
            (ArtifactInventoryState.TEMPORARY, self.tmp_dir),
        ):
            for path in directory.iterdir():
                artifact_key = path.name[:64]
                if _ARTIFACT_KEY_RE.fullmatch(artifact_key) is None:
                    continue
                reason = None
                if state is ArtifactInventoryState.DENIED:
                    with contextlib.suppress(OSError, ValueError, ManifestValidationError):
                        reason_value = _read_json_file(path).get("reason")
                        if isinstance(reason_value, str):
                            reason = reason_value
                inventory.append(
                    ArtifactInventoryEntry(
                        artifact_key=artifact_key,
                        state=state,
                        storage_name=path.name,
                        size_bytes=self._tree_bytes(path),
                        reason=reason,
                    )
                )
        inventory.sort(key=lambda item: (item.artifact_key, item.state.value, item.storage_name))
        return DomainInspection(
            domain_uuid=self.domain_uuid,
            scope=self.domain_policy.scope.value,
            domain_id=self.domain_policy.domain_id,
            storage_class=self.domain_policy.storage_class.value,
            filesystem_type=self.filesystem_type,
            policy_version=self.policy_version,
            store_bytes=self._tree_bytes(self.root),
            filesystem_free_bytes=usage.free,
            max_artifact_bytes=self.capacity_policy.max_artifact_bytes,
            max_store_bytes=self.capacity_policy.max_store_bytes,
            min_free_bytes=self.capacity_policy.min_free_bytes,
            inventory=tuple(inventory),
        )

    def inspect_artifact(
        self,
        identity: WeightArtifactIdentity,
        *,
        validation: ValidationLevel,
    ) -> ArtifactInspection:
        result = self._lookup(
            identity,
            validation=validation,
            deadline=time.monotonic(),
            record_invalid=False,
        )
        manifest = None
        publication_id = None
        backing_fingerprint = None
        if result.lease is not None:
            manifest = result.lease.manifest
            publication_id = result.lease.provenance.publication_id
            backing_fingerprint = result.lease.provenance.backing_fingerprint
            result.lease.close()
        entry = self._entry_path(identity.key)
        return ArtifactInspection(
            artifact_key=identity.key,
            status=result.status,
            size_bytes=self._tree_bytes(entry),
            manifest=manifest,
            publication_id=publication_id,
            backing_fingerprint=backing_fingerprint,
            failure=result.failure,
            build_lock_active=lock_is_active(self._build_lock_path(identity.key)),
            artifact_lock_active=lock_is_active(self._artifact_lock_path(identity.key)),
        )


__all__ = ["FilesystemHostWeightStore", "detect_storage_class"]
