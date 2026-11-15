# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Resolve Hub datasets and local mirrors for continuous visual-session QA."""

from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import BinaryIO, cast

from filelock import FileLock
from huggingface_hub import constants as hf_constants

from vllm_omni.benchmarks.data_modules.videomme_dataset import (
    ensure_videomme_videos_extracted,
    resolve_videomme_root,
    videomme_local_parquet,
)
from vllm_omni.transformers_utils.repo_utils import hf_api

OVO_REPO = "JoeLeelyf/OVO-Bench"
OVO_ANNOTATION_REVISION = "f3911d72411e6d19e358e8e044ec733051e86e6d"


def _ovo_annotation() -> Path:
    import requests

    root = Path(hf_constants.HF_HOME) / "vllm_omni" / "ovo-bench" / OVO_ANNOTATION_REVISION
    root.mkdir(parents=True, exist_ok=True)
    path = root / "ovo_bench_new.json"
    with FileLock(str(root / ".annotation.lock")):
        if path.is_file():
            return path
        if hf_constants.HF_HUB_OFFLINE:
            raise FileNotFoundError(f"OVO annotations are not cached at {path}; provide a local --dataset-path")
        url = f"https://raw.githubusercontent.com/JoeLeelyf/OVO-Bench/{OVO_ANNOTATION_REVISION}/data/ovo_bench_new.json"
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            downloaded = Path(temporary) / path.name
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with downloaded.open("wb") as out:
                    for block in response.iter_content(1024 * 1024):
                        out.write(block)
            if not isinstance(json.loads(downloaded.read_text()), list):
                raise ValueError("Expected an OVO annotation array")
            downloaded.replace(path)
    return path


class _ArchiveParts:
    """Read a split tar sequentially without a second full-size archive copy."""

    def __init__(self, files):
        self.files = iter(files)
        self.current = next(self.files, None)

    def read(self, size: int) -> bytes:
        data = bytearray()
        while self.current is not None and len(data) < size:
            block = self.current.read(size - len(data))
            if block:
                data.extend(block)
            else:
                self.current = next(self.files, None)
        return bytes(data)


def _ovo_media(root: Path) -> Path:
    for path in (root / "src_videos", root / "data" / "src_videos", root):
        if any(
            (path / name).is_dir()
            for name in (
                "AutoEvalMetaData",
                "Ego4D",
                "OpenEQA",
                "YouTube_Games",
                "COIN",
                "youcook2",
                "MovieNet",
                "star",
                "thumos",
                "perception_test",
                "cross_task",
                "hirest",
            )
        ):
            return path
    raise FileNotFoundError(f"OVO source videos not found under {root}")


def _extract_ovo(root: Path) -> Path:
    target = root / ".vllm_omni_src_videos"
    with FileLock(str(root / ".ovo-extract.lock")):
        if target.is_dir():
            return _ovo_media(target)
        parts = sorted(root.glob("src_videos.tar.part*"))
        if not parts:
            parts = [root / "src_videos.tar"]
        if not all(part.is_file() for part in parts):
            raise FileNotFoundError(f"No src_videos.tar or split parts under {root}")
        with tempfile.TemporaryDirectory(prefix=".ovo-", dir=root) as temporary, ExitStack() as stack:
            staging = Path(temporary).resolve()
            stream = _ArchiveParts([stack.enter_context(part.open("rb")) for part in parts])
            with tarfile.open(fileobj=cast(BinaryIO, stream), mode="r|*") as archive:
                for member in archive:
                    destination = (staging / member.name).resolve()
                    if not destination.is_relative_to(staging) or not (member.isdir() or member.isfile()):
                        raise ValueError(f"Unsafe OVO archive member: {member.name}")
                    if member.isdir():
                        destination.mkdir(parents=True, exist_ok=True)
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        source = archive.extractfile(member)
                        assert source is not None
                        with source, destination.open("wb") as out:
                            shutil.copyfileobj(source, out)
            _ovo_media(staging)
            staging.rename(target)
    return _ovo_media(target)


def resolve_long_video_source(
    dataset_path: str | None, *, dataset: str, media_root: Path | None = None
) -> tuple[Path, Path]:
    """Accept a Hub ID, a local dataset root, or an explicit annotation file."""
    local = Path(dataset_path).expanduser() if dataset_path else None
    if (
        local is not None
        and not local.exists()
        and (local.is_absolute() or str(dataset_path).startswith((".", "~")) or local.suffix in {".json", ".parquet"})
    ):
        raise FileNotFoundError(f"Local dataset path does not exist: {local}")
    if local is not None and local.is_file():
        if media_root is not None:
            return local, media_root.expanduser()
        root = local.parent.parent if local.parent.name in {"data", "videomme"} else local.parent
        if dataset == "ovo-bench":
            try:
                media = _ovo_media(root)
            except FileNotFoundError:
                media = _extract_ovo(root)
            return local, media
        return local, ensure_videomme_videos_extracted(root)
    if dataset == "videomme-streaming":
        root = resolve_videomme_root(dataset_path or "lmms-lab/Video-MME")
        annotation = videomme_local_parquet(root)
        if annotation is None:
            raise FileNotFoundError(f"Video-MME parquet not found under {root}")
        return annotation, media_root.expanduser() if media_root else ensure_videomme_videos_extracted(root)
    if local is not None and local.is_dir():
        root = local.resolve()
        annotation = next(
            (p for p in (root / "ovo_bench_new.json", root / "data" / "ovo_bench_new.json") if p.is_file()), None
        )
        if annotation is None:
            raise FileNotFoundError(f"OVO ovo_bench_new.json not found under {root}")
    else:
        annotation = _ovo_annotation()
        if media_root is not None:
            return annotation, media_root.expanduser()
        repo, _, revision = (dataset_path or OVO_REPO).partition("@")
        root = Path(
            hf_api().snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                revision=revision or None,
                allow_patterns=["src_videos.tar.part*"],
                local_files_only=hf_constants.HF_HUB_OFFLINE,
            )
        )
    if media_root is not None:
        return annotation, media_root.expanduser()
    try:
        media = _ovo_media(root)
    except FileNotFoundError:
        media = _extract_ovo(root)
    return annotation, media
