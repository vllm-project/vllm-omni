# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Store-scoped streaming safetensors publication."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import struct
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import TracebackType

import torch

from ..identity import WeightArtifactIdentity, canonical_json
from ..manifest import FileManifestEntry, PublicationMarker, TensorManifestEntry, WeightManifest
from ..protocols import ProductionMetadata, TensorWriteSpec

_HASH_CHUNK_BYTES = 8 * 1024**2
_MAX_HASH_WORKERS = 8
_MANIFEST_FILE = "manifest.json"
_READY_FILE = "READY.json"

_SAFETENSORS_DTYPES: dict[torch.dtype, str] = {
    torch.bool: "BOOL",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
}
for _torch_name, _safe_name in (
    ("float8_e4m3fn", "F8_E4M3"),
    ("float8_e5m2", "F8_E5M2"),
    ("float8_e4m3fnuz", "F8_E4M3FNUZ"),
    ("float8_e5m2fnuz", "F8_E5M2FNUZ"),
):
    if (_dtype := getattr(torch, _torch_name, None)) is not None:
        _SAFETENSORS_DTYPES[_dtype] = _safe_name


class ArtifactWriterError(RuntimeError):
    pass


def _safe_relative_name(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or len(path.parts) != 1 or path.name != value or value in {".", ".."}:
        raise ArtifactWriterError(f"artifact payload name must be one safe relative basename: {value!r}")
    if not value.endswith(".safetensors"):
        raise ArtifactWriterError(f"artifact payload must use the .safetensors format: {value!r}")
    return value


def _pwrite_all(fd: int, data: bytes | bytearray | memoryview, offset: int) -> None:
    view = memoryview(data).cast("B")
    written = 0
    while written < len(view):
        count = os.pwrite(fd, view[written:], offset + written)
        if count <= 0:
            raise OSError(errno.EIO, "short pwrite while publishing host weights")
        written += count


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_payload(fd: int) -> None:
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class _PayloadDigests:
    file: str
    tensors: dict[str, str]


def _scan_payload(path: Path, specs: tuple[TensorWriteSpec, ...]) -> _PayloadDigests:
    """Calculate file and tensor digests in one ordered read."""
    file_digest = hashlib.sha256()
    tensor_digests: dict[str, str] = {}
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ArtifactWriterError(f"truncated safetensors header in {path.name!r}")
        header_size = struct.unpack("<Q", prefix)[0]
        header = handle.read(header_size)
        if len(header) != header_size:
            raise ArtifactWriterError(f"truncated safetensors header in {path.name!r}")
        file_digest.update(prefix)
        file_digest.update(header)
        for spec in specs:
            digest = hashlib.sha256()
            remaining = spec.nbytes
            while remaining:
                chunk = handle.read(min(remaining, _HASH_CHUNK_BYTES))
                if not chunk:
                    raise ArtifactWriterError(f"truncated tensor {spec.name!r} in {path.name!r}")
                file_digest.update(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
            tensor_digests[spec.name] = digest.hexdigest()
        if handle.read(1):
            raise ArtifactWriterError(f"unexpected trailing bytes in {path.name!r}")
    return _PayloadDigests(file_digest.hexdigest(), tensor_digests)


class _Coverage:
    def __init__(self, size: int) -> None:
        self._size = size
        self._ranges: list[tuple[int, int]] = []

    def add(self, start: int, end: int) -> None:
        if start < 0 or end < start or end > self._size:
            raise ArtifactWriterError(f"write range [{start}, {end}) exceeds tensor storage {self._size}")
        if start == end:
            return
        for existing_start, existing_end in self._ranges:
            if start < existing_end and existing_start < end:
                raise ArtifactWriterError(f"overlapping tensor write ranges [{start}, {end}) and {self._ranges}")
        self._ranges.append((start, end))

    def complete(self) -> bool:
        if self._size == 0:
            return True
        cursor = 0
        for start, end in sorted(self._ranges):
            if start != cursor:
                return False
            cursor = end
        return cursor == self._size


class StreamingSafeTensorFile:
    """One preallocated safetensors file filled through bounded CPU writes."""

    def __init__(
        self,
        owner: FilesystemArtifactWriter,
        relative_name: str,
        specs: tuple[TensorWriteSpec, ...],
        *,
        ordered: bool,
    ) -> None:
        self._owner = owner
        self.relative_name = _safe_relative_name(relative_name)
        if not specs:
            raise ArtifactWriterError("a tensor payload file must contain at least one tensor")
        by_name = {spec.name: spec for spec in specs}
        if len(by_name) != len(specs):
            raise ArtifactWriterError("tensor payload contains duplicate storage keys")
        unsupported = [spec.dtype for spec in specs if spec.dtype not in _SAFETENSORS_DTYPES]
        if unsupported:
            raise ArtifactWriterError(f"unsupported safetensors dtypes: {unsupported}")
        self.specs = tuple(sorted(specs, key=lambda spec: spec.name))
        self._specs = {spec.name: spec for spec in self.specs}
        self._coverage = {spec.name: _Coverage(spec.nbytes) for spec in self.specs}
        self._ordered = ordered
        self._closed = False

        offsets: dict[str, int] = {}
        cursor = 0
        for spec in self.specs:
            offsets[spec.name] = cursor
            cursor += spec.nbytes
        header: dict[str, object] = {"__metadata__": {"format": "pt", "host_weight_runtime": "1"}}
        for spec in self.specs:
            start = offsets[spec.name]
            header[spec.name] = {
                "dtype": _SAFETENSORS_DTYPES[spec.dtype],
                "shape": list(spec.shape),
                "data_offsets": [start, start + spec.nbytes],
            }
        encoded_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
        encoded_header += b" " * (-len(encoded_header) % 8)
        self._payload_offset = 8 + len(encoded_header)
        self._offsets = offsets
        self._path = owner.staging_dir / self.relative_name
        total_size = self._payload_offset + cursor
        owner._reserve_file(total_size)
        self._fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_CLOEXEC, 0o600)
        prefix = struct.pack("<Q", len(encoded_header))
        try:
            if hasattr(os, "posix_fallocate"):
                try:
                    os.posix_fallocate(self._fd, 0, total_size)
                except OSError as exc:
                    if exc.errno not in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
                        raise
                    os.ftruncate(self._fd, total_size)
            else:  # pragma: no cover - Linux production path has posix_fallocate
                os.ftruncate(self._fd, total_size)
            _pwrite_all(self._fd, prefix, 0)
            _pwrite_all(self._fd, encoded_header, 8)
        except BaseException:
            os.close(self._fd)
            self._path.unlink(missing_ok=True)
            owner._forget_reserved_file(total_size)
            raise
        self._file_digest = hashlib.sha256(prefix + encoded_header) if ordered else None
        self._tensor_digests = {spec.name: hashlib.sha256() for spec in self.specs} if ordered else {}
        self._write_cursor = self._payload_offset
        owner._opened(self)

    @staticmethod
    def _bytes(tensor: torch.Tensor) -> memoryview:
        if tensor.device.type != "cpu" or tensor.is_meta:
            raise ArtifactWriterError("artifact producers must provide materialized CPU tensor chunks")
        contiguous = tensor.detach().contiguous()
        return memoryview(contiguous.view(torch.uint8).numpy()).cast("B")

    def write_tensor(self, name: str, tensor: torch.Tensor) -> None:
        spec = self._get_spec(name)
        if tuple(tensor.shape) != spec.shape or tensor.dtype != spec.dtype:
            raise ArtifactWriterError(
                f"tensor {name!r} differs from its declared shape/dtype: "
                f"got {tuple(tensor.shape)}/{tensor.dtype}, expected {spec.shape}/{spec.dtype}"
            )
        self._write(name, 0, self._bytes(tensor))

    def write_rows(self, name: str, start_row: int, tensor: torch.Tensor) -> None:
        spec = self._get_spec(name)
        if not spec.shape:
            if start_row != 0 or tuple(tensor.shape) != ():
                raise ArtifactWriterError(f"scalar tensor {name!r} only accepts one complete scalar write")
            self.write_tensor(name, tensor)
            return
        if tensor.dtype != spec.dtype or tuple(tensor.shape[1:]) != spec.shape[1:]:
            raise ArtifactWriterError(f"row chunk for {name!r} differs from its declared dtype or trailing shape")
        row_count = tensor.shape[0]
        if start_row < 0 or start_row + row_count > spec.shape[0]:
            raise ArtifactWriterError(f"row chunk for {name!r} exceeds its declared first dimension")
        row_bytes = spec.nbytes // spec.shape[0] if spec.shape[0] else 0
        self._write(name, start_row * row_bytes, self._bytes(tensor))

    def _get_spec(self, name: str) -> TensorWriteSpec:
        if self._closed:
            raise ArtifactWriterError("cannot write through a closed tensor file")
        try:
            return self._specs[name]
        except KeyError as exc:
            raise ArtifactWriterError(f"tensor {name!r} was not declared in this payload file") from exc

    def _write(self, name: str, byte_offset: int, data: memoryview) -> None:
        file_offset = self._payload_offset + self._offsets[name] + byte_offset
        if self._ordered and file_offset != self._write_cursor:
            raise ArtifactWriterError(
                f"ordered tensor payload expected offset {self._write_cursor}, got {file_offset} for {name!r}"
            )
        coverage = self._coverage[name]
        coverage.add(byte_offset, byte_offset + len(data))
        _pwrite_all(self._fd, data, file_offset)
        if self._ordered:
            assert self._file_digest is not None
            self._file_digest.update(data)
            self._tensor_digests[name].update(data)
            self._write_cursor += len(data)

    def close(self) -> None:
        if self._closed:
            return
        missing = [name for name, coverage in self._coverage.items() if not coverage.complete()]
        if missing:
            self.abort()
            raise ArtifactWriterError(f"tensor payload closed with incomplete writes: {missing}")
        self._closed = True
        digests = None
        if self._ordered:
            assert self._file_digest is not None
            digests = _PayloadDigests(
                self._file_digest.hexdigest(),
                {name: digest.hexdigest() for name, digest in self._tensor_digests.items()},
            )
        try:
            os.fchmod(self._fd, 0o444)
            self._owner._closed(self, digests, self._fd)
        except BaseException:
            os.close(self._fd)
            raise

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        os.close(self._fd)
        size = self._path.stat().st_size if self._path.exists() else 0
        self._path.unlink(missing_ok=True)
        self._owner._aborted(self, size)

    def __enter__(self) -> StreamingSafeTensorFile:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


class FilesystemArtifactWriter:
    """Scoped writer that cannot publish outside one invisible staging dir."""

    def __init__(
        self,
        staging_dir: Path,
        *,
        capacity_check: Callable[[int, int], None],
    ) -> None:
        self.staging_dir = staging_dir
        self._capacity_check = capacity_check
        self._active: set[StreamingSafeTensorFile] = set()
        self._files: dict[str, tuple[TensorWriteSpec, ...]] = {}
        self._digests: dict[str, _PayloadDigests] = {}
        self._reserved_bytes = 0
        self._finalized = False
        self._fsync_executor: ThreadPoolExecutor | None = None
        self._fsync_futures: list[Future[None]] = []

    def open_tensor_file(
        self,
        relative_name: str,
        specs: tuple[TensorWriteSpec, ...],
        *,
        ordered: bool = False,
    ) -> StreamingSafeTensorFile:
        if self._finalized:
            raise ArtifactWriterError("cannot add payloads after artifact finalization")
        relative_name = _safe_relative_name(relative_name)
        if relative_name in self._files or any(item.relative_name == relative_name for item in self._active):
            raise ArtifactWriterError(f"artifact payload {relative_name!r} was declared more than once")
        return StreamingSafeTensorFile(self, relative_name, specs, ordered=ordered)

    def _reserve_file(self, size: int) -> None:
        self._capacity_check(self._reserved_bytes + size, size)
        self._reserved_bytes += size

    def _forget_reserved_file(self, size: int) -> None:
        self._reserved_bytes = max(0, self._reserved_bytes - size)

    def _opened(self, writer: StreamingSafeTensorFile) -> None:
        self._active.add(writer)

    def _closed(self, writer: StreamingSafeTensorFile, digests: _PayloadDigests | None, fd: int) -> None:
        self._active.remove(writer)
        self._files[writer.relative_name] = writer.specs
        if digests is not None:
            self._digests[writer.relative_name] = digests
        if self._fsync_executor is None:
            self._fsync_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hwr-fsync")
        self._fsync_futures.append(self._fsync_executor.submit(_fsync_payload, fd))

    def _aborted(self, writer: StreamingSafeTensorFile, size: int) -> None:
        self._active.discard(writer)
        self._forget_reserved_file(size)

    def abort(self) -> None:
        for writer in tuple(self._active):
            with contextlib.suppress(Exception):
                writer.abort()
        with contextlib.suppress(Exception):
            self._wait_for_payload_fsync()

    def _wait_for_payload_fsync(self) -> None:
        if self._fsync_executor is None:
            return
        executor = self._fsync_executor
        futures = tuple(self._fsync_futures)
        self._fsync_executor = None
        self._fsync_futures.clear()
        executor.shutdown(wait=True)
        for future in futures:
            future.result()

    def finalize(
        self,
        identity: WeightArtifactIdentity,
        metadata: ProductionMetadata,
    ) -> tuple[WeightManifest, PublicationMarker]:
        if self._finalized:
            raise ArtifactWriterError("artifact writer was finalized more than once")
        if self._active:
            raise ArtifactWriterError("artifact producer left tensor payload files open")
        if not self._files:
            raise ArtifactWriterError("artifact producer emitted no tensor payload files")
        if metadata.producer_schema != identity.producer.manifest_schema:
            raise ArtifactWriterError("production metadata does not match the identity's manifest schema")
        if metadata.restorer_schema != identity.producer.restorer_schema:
            raise ArtifactWriterError("production metadata does not match the identity's restorer schema")

        self._wait_for_payload_fsync()

        tensor_entries: list[TensorManifestEntry] = []
        file_entries: list[FileManifestEntry] = []
        seen_tensors: set[str] = set()
        file_names = sorted(self._files)
        unhashed = [file_name for file_name in file_names if file_name not in self._digests]
        if unhashed:
            with ThreadPoolExecutor(max_workers=min(_MAX_HASH_WORKERS, len(unhashed))) as executor:
                scans = executor.map(
                    lambda file_name: _scan_payload(self.staging_dir / file_name, self._files[file_name]),
                    unhashed,
                )
                self._digests.update(zip(unhashed, scans, strict=True))

        for file_name in file_names:
            path = self.staging_dir / file_name
            specs = self._files[file_name]
            digests = self._digests[file_name]
            file_entries.append(
                FileManifestEntry(
                    file_name=file_name,
                    size=path.stat().st_size,
                    sha256=digests.file,
                )
            )
            for spec in specs:
                if spec.name in seen_tensors:
                    raise ArtifactWriterError(f"artifact tensor {spec.name!r} occurs in multiple payload files")
                seen_tensors.add(spec.name)
                tensor_entries.append(
                    TensorManifestEntry(
                        name=spec.name,
                        kind=spec.kind,
                        role=spec.role,
                        file_name=file_name,
                        storage_key=spec.name,
                        shape=spec.shape,
                        dtype=str(spec.dtype),
                        stride=tuple(torch.empty(spec.shape, dtype=spec.dtype, device="meta").stride()),
                        nbytes=spec.nbytes,
                        sha256=digests.tensors[spec.name],
                    )
                )

        ordered_tensor_entries = tuple(sorted(tensor_entries, key=lambda entry: entry.name))
        content_digest = hashlib.sha256()
        for entry in ordered_tensor_entries:
            content_digest.update(canonical_json(entry.semantic_dict()))

        manifest = WeightManifest(
            schema_version=1,
            artifact_key=identity.key,
            identity=identity,
            producer_schema=metadata.producer_schema,
            restorer_schema=metadata.restorer_schema,
            tensors=ordered_tensor_entries,
            files=tuple(file_entries),
            artifact_content_sha256=content_digest.hexdigest(),
            format_metadata=metadata.format_metadata,
        )
        manifest_bytes = manifest.canonical_bytes
        self._write_metadata_file(_MANIFEST_FILE, manifest_bytes)
        marker = PublicationMarker(
            schema_version=1,
            publication_id=uuid.uuid4().hex,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            artifact_content_sha256=manifest.artifact_content_sha256,
            integrity_mode="full_checksum",
        )
        self._write_metadata_file(_READY_FILE, marker.canonical_bytes)
        _fsync_directory(self.staging_dir)
        self._finalized = True
        return manifest, marker

    def _write_metadata_file(self, name: str, data: bytes) -> None:
        path = self.staging_dir / name
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_CLOEXEC, 0o600)
        try:
            _pwrite_all(fd, data, 0)
            os.fsync(fd)
            os.fchmod(fd, 0o444)
            os.fsync(fd)
        finally:
            os.close(fd)


__all__ = [
    "ArtifactWriterError",
    "FilesystemArtifactWriter",
    "StreamingSafeTensorFile",
]
