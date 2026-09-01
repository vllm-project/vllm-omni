# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""OmniInteract dataset discovery for the standard serving benchmark."""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import tarfile
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vllm.benchmarks.datasets import BenchmarkDataset, SampleRequest
from vllm.tokenizers import TokenizerLike
from vllm.transformers_utils.repo_utils import hf_fs

OMNIINTERACT_SUBSETS = ("1q1a", "1q1a_math", "1qna")
DEFAULT_OMNIINTERACT_REPO = "lucky-lance/OmniInteract"
DEFAULT_MAX_VIDEO_DURATION_S = 3600.0
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OmniInteractCase:
    subset: str
    video_rel: str
    video_path: Path
    annotation_path: Path
    scene_type: str


@dataclass(frozen=True)
class OmniInteractSessionOptions:
    output_root: Path
    timeout_s: float
    media_timeout_s: float
    ref_audio: str
    require_response: bool = False
    max_video_duration_s: float = DEFAULT_MAX_VIDEO_DURATION_S


@dataclass(frozen=True)
class OmniInteractPreparedInput:
    duration_s: float
    pcm16: bytes = field(repr=False)
    video_frames: tuple[str | None, ...] = field(repr=False)
    ref_audio_data_url: str = field(repr=False)


@dataclass
class OmniInteractSampleRequest(SampleRequest):
    omniinteract_case: OmniInteractCase | None = None
    omniinteract_options: OmniInteractSessionOptions | None = None
    omniinteract_prepared_input: OmniInteractPreparedInput | None = field(default=None, repr=False)


class OmniInteractDataset(BenchmarkDataset):
    def __init__(
        self,
        *,
        data_root: str | None,
        dataset_repo: str = DEFAULT_OMNIINTERACT_REPO,
        subsets: Sequence[str] = OMNIINTERACT_SUBSETS,
        random_seed: int = 0,
        disable_shuffle: bool = False,
    ) -> None:
        super().__init__(
            dataset_path=data_root or dataset_repo,
            random_seed=random_seed,
            disable_shuffle=disable_shuffle,
        )
        self.root = resolve_omniinteract_root(data_root, dataset_repo)
        self.subsets = tuple(subsets)

    def sample(
        self,
        tokenizer: TokenizerLike | None,
        num_requests: int,
        *,
        request_id_prefix: str = "",
        options: OmniInteractSessionOptions,
        **_: Any,
    ) -> list[SampleRequest]:
        del tokenizer
        cases = discover_omniinteract_cases(
            self.root,
            self.subsets,
            num_prompts=num_requests,
            seed=self.random_seed,
            disable_shuffle=self.disable_shuffle,
        )
        return [
            OmniInteractSampleRequest(
                prompt="",
                prompt_len=0,
                expected_output_len=0,
                multi_modal_data=None,
                request_id=f"{request_id_prefix}{index}",
                omniinteract_case=case,
                omniinteract_options=options,
            )
            for index, case in enumerate(cases)
        ]


def _data_dir(root: Path) -> Path:
    for candidate in (root, root / "data"):
        if any((candidate / subset).is_dir() for subset in OMNIINTERACT_SUBSETS):
            return candidate
    raise FileNotFoundError(f"OmniInteract data not found under {root}")


def _confined_path(root: Path, relative: object, *, field: str) -> Path:
    text, resolved_root = str(relative or ""), root.resolve()
    path = Path(text)
    destination = (root / path).resolve()
    if not text or path.is_absolute() or ".." in path.parts or not destination.is_relative_to(resolved_root):
        raise ValueError(f"Unsafe OmniInteract {field} path: {text!r}")
    return destination


def _safe_extract(archive: tarfile.TarFile, target: Path) -> None:
    root = target.resolve()
    members = archive.getmembers()
    for member in members:
        path = Path(member.name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not (root / path).resolve().is_relative_to(root)
            or not (member.isdir() or member.isfile())
        ):
            raise ValueError(f"Unsafe path in OmniInteract archive: {member.name!r}")
    archive.extractall(target, members=members, filter="data")


def _extract_archive(archive: Path, target: Path) -> Path:
    if target.is_symlink():
        raise ValueError(f"Refusing to extract through symlink: {target}")
    stat = archive.stat()
    published = target / f"{stat.st_size}-{stat.st_mtime_ns}"
    if published.is_symlink():
        raise ValueError(f"Refusing to use symlinked extraction: {published}")
    if published.is_dir():
        try:
            return _data_dir(published)
        except FileNotFoundError:
            pass
    target.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".tmp-", dir=target))
    try:
        with tarfile.open(archive, "r:*") as handle:
            _safe_extract(handle, staging)
        root = _data_dir(staging)
        try:
            staging.replace(published)
            return published / root.relative_to(staging)
        except OSError:
            if not published.is_dir():
                raise
            return _data_dir(published)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def resolve_omniinteract_root(data_root: str | None, dataset_repo: str = DEFAULT_OMNIINTERACT_REPO) -> Path:
    """Resolve a local tree/archive or download the official archive."""

    if data_root:
        local = Path(data_root).expanduser().resolve()
        if local.is_file():
            return _extract_archive(local, local.parent / f".{local.stem}.vllm_omni_extracted")
        if not local.is_dir():
            raise FileNotFoundError(f"--dataset-path does not exist: {local}")
        try:
            return _data_dir(local)
        except FileNotFoundError:
            for name in ("data.tar.gz", "data.tar"):
                if (archive := local / name).is_file():
                    return _extract_archive(archive, local / ".vllm_omni_extracted")
            raise

    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    target = cache / "vllm_omni" / "omniinteract" / dataset_repo.replace("/", "__")
    downloads = target / ".downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    filesystem, errors = hf_fs(), []
    for name in ("data.tar.gz", "data.tar"):
        archive = downloads / name
        try:
            if not archive.is_file():
                descriptor, temporary_name = tempfile.mkstemp(prefix=f".{name}.", dir=downloads)
                os.close(descriptor)
                temporary = Path(temporary_name)
                try:
                    filesystem.get_file(f"datasets/{dataset_repo}/{name}", str(temporary))
                    if not temporary.stat().st_size:
                        raise ValueError("downloaded archive is empty")
                    temporary.replace(archive)
                finally:
                    temporary.unlink(missing_ok=True)
            return _extract_archive(archive, target)
        except Exception as exc:  # noqa: BLE001, PERF203 - Hub errors vary by version
            errors.append(f"{name}: {exc}")
    raise FileNotFoundError(f"Could not download OmniInteract from {dataset_repo!r}: {'; '.join(errors)}")


def _mapping_cases(root: Path, subset: str) -> list[OmniInteractCase]:
    mapping_path = root / "video_json_map.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Missing OmniInteract mapping: {mapping_path}")
    mapping = json.loads(mapping_path.read_text())
    entries = mapping.get("entries") if isinstance(mapping, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"Invalid OmniInteract mapping: {mapping_path}")
    cases = []
    for row in entries:
        if not isinstance(row, dict):
            raise ValueError(f"Invalid OmniInteract mapping row in {mapping_path}")
        video_rel, annotation_rel = str(row.get("video") or ""), str(row.get("annotation") or "")
        video = _confined_path(root, video_rel, field="video")
        annotation = _confined_path(root, annotation_rel, field="annotation")
        if not video.is_file() or not annotation.is_file():
            raise FileNotFoundError(f"OmniInteract mapping references missing files: {video_rel!r}, {annotation_rel!r}")
        cases.append(
            OmniInteractCase(
                subset,
                video_rel,
                video,
                annotation,
                str(row.get("scene_type") or "multi_turn").lower(),
            )
        )
    return cases


def _one_to_many_cases(root: Path) -> list[OmniInteractCase]:
    videos, annotations = root / "videos_bench", root / "annotations"
    if not videos.is_dir() or not annotations.is_dir():
        raise FileNotFoundError(f"Invalid OmniInteract 1qna layout under {root}")
    cases = []
    for video in sorted(videos.rglob("*.mp4")):
        if not video.resolve().is_relative_to(videos.resolve()):
            raise ValueError(f"Unsafe OmniInteract video path: {video}")
        relative = video.relative_to(videos)
        annotation = (annotations / relative).with_suffix(".json").resolve()
        if not annotation.is_relative_to(annotations.resolve()):
            raise ValueError(f"Unsafe OmniInteract annotation path: {relative}")
        if not annotation.is_file():
            raise FileNotFoundError(f"Missing OmniInteract annotation: {annotation}")
        cases.append(OmniInteractCase("1qna", f"videos_bench/{relative.as_posix()}", video, annotation, "1qna"))
    return cases


def discover_omniinteract_cases(
    root: Path,
    subsets: Sequence[str],
    *,
    num_prompts: int,
    seed: int = 0,
    disable_shuffle: bool = False,
) -> list[OmniInteractCase]:
    invalid = set(subsets) - set(OMNIINTERACT_SUBSETS)
    if invalid:
        raise ValueError(f"Unsupported OmniInteract subsets: {sorted(invalid)}")
    if not subsets:
        raise ValueError("At least one OmniInteract subset is required")
    if len(set(subsets)) != len(subsets):
        raise ValueError("OmniInteract subsets must not contain duplicates")
    if num_prompts < 0:
        raise ValueError("num_prompts must be non-negative")
    data_root, cases = _data_dir(root.resolve()), []
    for subset in subsets:
        subset_root = data_root / subset
        selected = _one_to_many_cases(subset_root) if subset == "1qna" else _mapping_cases(subset_root, subset)
        if not selected:
            raise ValueError(f"No OmniInteract sessions found for requested subset {subset!r}")
        cases.extend(selected)
    if len({case.video_path for case in cases}) != len(cases):
        raise ValueError("OmniInteract dataset contains duplicate video paths")
    if not disable_shuffle:
        random.Random(seed).shuffle(cases)
    if num_prompts:
        if num_prompts > len(cases):
            logger.warning(
                "Requested %d OmniInteract prompts but only %d are available; using all cases", num_prompts, len(cases)
            )
        cases = cases[:num_prompts]
    return cases


def case_manifest(case: OmniInteractCase, output_dir: Path) -> dict[str, Any]:
    return {
        "sample_id": f"{case.subset}__{output_dir.name}",
        "video": str(case.video_path),
        "gt_json": str(case.annotation_path.resolve()),
        "model_json": str((output_dir / "wav_transcript.json").resolve()),
        "scene_type": "1QnA" if case.scene_type == "1qna" else case.scene_type,
    }
