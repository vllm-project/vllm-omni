# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Canonical diffusion weight-source identity and one-load replacement guard."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import regex as re

from vllm_omni.host_weight_runtime import CanonicalJson, WeightSourceIdentity
from vllm_omni.host_weight_runtime.identity import canonical_json

from .contracts import (
    FinalLayoutContractCode,
    FinalLayoutContractError,
    ImplementationIdentity,
)

_HASH_CHUNK_BYTES = 8 * 1024**2
_IMMUTABLE_REVISION_RE = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})\Z")


class WeightSourceKind(str, Enum):
    LOCAL_PATH = "local_path"
    HUGGING_FACE_HUB = "hugging_face_hub"


@dataclass(frozen=True)
class PreparedWeightSource:
    """Canonical source files and adapter ABI selected by the ordinary loader."""

    model_or_path: str
    subfolder: str | None
    requested_revision: str | None
    prefix: str
    resolved_root: Path
    weight_files: tuple[Path, ...]
    use_safetensors: bool
    checkpoint_adapter: ImplementationIdentity | None = None
    source_kind: WeightSourceKind = WeightSourceKind.LOCAL_PATH

    def __post_init__(self) -> None:
        if not self.model_or_path or not isinstance(self.prefix, str):
            raise ValueError("prepared source model and prefix must be valid strings")
        if not isinstance(self.resolved_root, Path):
            raise ValueError("prepared source root must use pathlib.Path")
        if not isinstance(self.weight_files, tuple) or not self.weight_files:
            raise ValueError("prepared source must contain an immutable tuple of weight files")
        if any(not isinstance(path, Path) for path in self.weight_files):
            raise ValueError("prepared source weight files must use pathlib.Path")
        if not isinstance(self.use_safetensors, bool):
            raise ValueError("prepared source safetensors mode must be a boolean")
        if self.checkpoint_adapter is not None and not isinstance(
            self.checkpoint_adapter,
            ImplementationIdentity,
        ):
            raise ValueError("checkpoint adapter must use ImplementationIdentity")
        if not isinstance(self.source_kind, WeightSourceKind):
            raise ValueError("prepared source kind must use WeightSourceKind")


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    relative_name: str
    size: int
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int
    symlink_target: str | None
    content_id: str

    def semantic_dict(self) -> dict[str, object]:
        return {
            "relative_name": self.relative_name,
            "size": self.size,
            "content_id": self.content_id,
        }

    def unchanged(self) -> bool:
        try:
            current = self.path.stat()
            symlink_target = os.readlink(self.path) if self.path.is_symlink() else None
        except OSError:
            return False
        return (
            current.st_size == self.size
            and current.st_dev == self.device
            and current.st_ino == self.inode
            and current.st_mtime_ns == self.mtime_ns
            and current.st_ctime_ns == self.ctime_ns
            and symlink_target == self.symlink_target
        )


@dataclass(frozen=True)
class _SourceSnapshot:
    semantic: CanonicalJson
    revision: str
    files: tuple[_FileSnapshot, ...]

    def unchanged(self) -> bool:
        return all(file.unchanged() for file in self.files)


@dataclass(frozen=True)
class FinalLayoutSourceContext:
    """Resolved source identity plus process-local replacement observations."""

    identity: WeightSourceIdentity
    snapshots: tuple[_SourceSnapshot, ...]

    def sources_unchanged(self) -> bool:
        return all(snapshot.unchanged() for snapshot in self.snapshots)

    def ensure_sources_unchanged(self) -> None:
        if not self.sources_unchanged():
            raise FinalLayoutContractError(
                FinalLayoutContractCode.SOURCE_CHANGED,
                "canonical source changed after final-layout identity resolution",
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _hf_blob_content_id(
    path: Path,
    *,
    source: PreparedWeightSource,
    source_root: Path,
) -> str | None:
    if source.source_kind is not WeightSourceKind.HUGGING_FACE_HUB or not path.is_symlink():
        return None
    if Path(source.model_or_path).expanduser().exists():
        return None

    parts = source_root.parts
    snapshot_indexes = [index for index, part in enumerate(parts[:-1]) if part == "snapshots"]
    if not snapshot_indexes:
        return None
    snapshot_index = snapshot_indexes[-1]
    revision = parts[snapshot_index + 1]
    if _IMMUTABLE_REVISION_RE.fullmatch(revision) is None:
        return None
    repo_root = Path(*parts[:snapshot_index])
    expected_repo_name = f"models--{source.model_or_path.replace('/', '--')}"
    if repo_root.name != expected_repo_name:
        return None
    snapshot_root = repo_root / "snapshots" / revision
    try:
        path.relative_to(snapshot_root)
    except ValueError:
        return None
    blobs_root = repo_root / "blobs"
    try:
        resolved_target = path.resolve(strict=True)
        resolved_blobs_root = blobs_root.resolve(strict=True)
    except OSError:
        return None
    if resolved_target.parent != resolved_blobs_root:
        return None
    blob_name = resolved_target.name
    if _IMMUTABLE_REVISION_RE.fullmatch(blob_name) is None:
        return None
    return f"immutable-blob:{blob_name.lower()}"


def _snapshot_revision(path: Path) -> str | None:
    parts = path.absolute().parts
    for index, part in enumerate(parts[:-1]):
        if part != "snapshots":
            continue
        candidate = parts[index + 1]
        if _IMMUTABLE_REVISION_RE.fullmatch(candidate) is not None:
            return candidate.lower()
    return None


def _logical_model_id(value: str) -> str:
    candidate = Path(value).expanduser()
    try:
        if candidate.exists():
            return str(candidate.resolve())
    except OSError:
        pass
    return value


def _snapshot_source(source: PreparedWeightSource) -> _SourceSnapshot:
    root = source.resolved_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"canonical source root is not a directory: {root}")
    immutable_revision = _snapshot_revision(root)
    if immutable_revision is None and source.requested_revision is not None:
        requested = source.requested_revision.strip()
        if _IMMUTABLE_REVISION_RE.fullmatch(requested) is not None:
            immutable_revision = requested.lower()

    files: list[_FileSnapshot] = []
    for path in sorted(source.weight_files, key=lambda item: str(item)):
        candidate = path.expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.absolute()
        try:
            relative_name = candidate.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"weight file {candidate} is outside its canonical source root {root}") from exc
        try:
            current = candidate.stat()
        except OSError as exc:
            raise ValueError(f"cannot stat canonical weight file {candidate}") from exc
        if not stat.S_ISREG(current.st_mode):
            raise ValueError(f"canonical weight source is not a regular file: {candidate}")
        symlink_target = os.readlink(candidate) if candidate.is_symlink() else None
        content_id = _hf_blob_content_id(
            candidate,
            source=source,
            source_root=root,
        )
        if content_id is None:
            content_id = f"sha256:{_sha256_file(candidate)}"
        files.append(
            _FileSnapshot(
                path=candidate,
                relative_name=relative_name,
                size=current.st_size,
                device=current.st_dev,
                inode=current.st_ino,
                mtime_ns=current.st_mtime_ns,
                ctime_ns=current.st_ctime_ns,
                symlink_target=symlink_target,
                content_id=content_id,
            )
        )

    file_semantics = [file.semantic_dict() for file in files]
    file_fingerprint = hashlib.sha256(canonical_json(file_semantics)).hexdigest()
    revision = immutable_revision or f"content-{file_fingerprint}"
    semantic = CanonicalJson.from_value(
        {
            "checkpoint_adapter": (
                source.checkpoint_adapter.to_dict() if source.checkpoint_adapter is not None else None
            ),
            "files": file_semantics,
            "model_or_path": _logical_model_id(source.model_or_path),
            "prefix": source.prefix,
            "resolved_revision": revision,
            "source_kind": source.source_kind.value,
            "subfolder": source.subfolder,
            "use_safetensors": source.use_safetensors,
        }
    )
    return _SourceSnapshot(semantic=semantic, revision=revision, files=tuple(files))


def _resolve_target_sources(
    prepared_sources: Sequence[PreparedWeightSource],
    target_names: frozenset[str],
) -> dict[str, PreparedWeightSource]:
    """Bind every target to one source using deterministic longest-prefix wins."""
    if not target_names:
        raise FinalLayoutContractError(
            FinalLayoutContractCode.SOURCE_COVERAGE_INVALID,
            "final-layout source identity requires at least one owned tensor",
        )

    bindings: dict[str, PreparedWeightSource] = {}
    for target_name in sorted(target_names):
        matches = [source for source in prepared_sources if target_name.startswith(source.prefix)]
        if not matches:
            raise FinalLayoutContractError(
                FinalLayoutContractCode.SOURCE_COVERAGE_INVALID,
                f"no canonical weight source covers final-layout tensor {target_name!r}",
            )
        longest_prefix = max(len(source.prefix) for source in matches)
        winners = [source for source in matches if len(source.prefix) == longest_prefix]
        if len(winners) != 1:
            candidates = sorted(
                f"{source.model_or_path}:{source.subfolder or ''}:{source.prefix}" for source in winners
            )
            raise FinalLayoutContractError(
                FinalLayoutContractCode.SOURCE_COVERAGE_INVALID,
                f"multiple equally specific canonical sources cover {target_name!r}: {candidates}",
            )
        bindings[target_name] = winners[0]
    return bindings


def resolve_final_layout_source_identity(
    prepared_sources: Sequence[PreparedWeightSource],
    *,
    model_id: str,
    target_names: frozenset[str],
) -> FinalLayoutSourceContext:
    """Resolve exact source identity for files covering the owned tensors."""
    bindings = _resolve_target_sources(prepared_sources, target_names)
    selected_sources = frozenset(bindings.values())
    snapshot_by_source = {source: _snapshot_source(source) for source in selected_sources}
    snapshots = tuple(sorted(snapshot_by_source.values(), key=lambda snapshot: snapshot.semantic.encoded))
    source_semantics = [snapshot.semantic.to_value() for snapshot in snapshots]
    target_bindings = [
        {
            "source_fingerprint": hashlib.sha256(snapshot_by_source[source].semantic.encoded).hexdigest(),
            "source_prefix": source.prefix,
            "target_name": target_name,
        }
        for target_name, source in sorted(bindings.items())
    ]
    source_document = {
        "sources": source_semantics,
        "target_bindings": target_bindings,
    }
    source_fingerprint = hashlib.sha256(canonical_json(source_document)).hexdigest()
    unique_revisions = tuple(sorted({snapshot.revision for snapshot in snapshots}))
    aggregate_revision = (
        unique_revisions[0]
        if len(unique_revisions) == 1
        else f"aggregate-{hashlib.sha256(canonical_json(unique_revisions)).hexdigest()}"
    )
    return FinalLayoutSourceContext(
        identity=WeightSourceIdentity(
            model_id=_logical_model_id(model_id),
            revision=aggregate_revision,
            fingerprint=source_fingerprint,
            metadata=CanonicalJson.from_value(source_document),
        ),
        snapshots=snapshots,
    )


__all__ = [
    "FinalLayoutSourceContext",
    "PreparedWeightSource",
    "WeightSourceKind",
    "resolve_final_layout_source_identity",
]
