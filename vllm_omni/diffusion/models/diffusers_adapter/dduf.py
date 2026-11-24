# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Keep DDUF loading through Diffusers' directory loader after 0.42 removes it.

A live pipeline owns one temporary directory of uncompressed archive size.
Copying uses a fixed buffer; multiple workers require separate disk space.
Normal collection/exit cleans it up; SIGKILL cannot run Python finalizers.
"""

from __future__ import annotations

import errno
import shutil
import stat
import tempfile
import weakref
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from diffusers import __version__ as diffusers_version
from huggingface_hub import read_dduf_file
from packaging.version import Version

from vllm_omni.transformers_utils.repo_utils import hf_api

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline


def load_diffusers_pipeline(
    loader_class: type[DiffusionPipeline], model_id: str, load_kwargs: dict[str, Any]
) -> DiffusionPipeline:
    """Preserve old loading and materialize DDUF only for the new directory API."""
    if Version(diffusers_version) < Version("0.42.0"):
        return loader_class.from_pretrained(model_id, **load_kwargs)
    kwargs = dict(load_kwargs)
    dduf_file = kwargs.pop("dduf_file", None)
    if not dduf_file:
        return loader_class.from_pretrained(model_id, **kwargs)
    if kwargs.get("custom_pipeline") or kwargs.get("load_connected_pipeline"):
        raise NotImplementedError("Custom and connected pipelines are not supported with DDUF.")

    if Path(model_id).is_dir():
        archive_path = Path(model_id) / dduf_file
    else:
        download_keys = ("cache_dir", "revision", "token", "local_files_only", "force_download", "subfolder")
        archive_path = Path(
            hf_api().hf_hub_download(
                repo_id=model_id,
                filename=dduf_file,
                **{key: kwargs[key] for key in download_keys if key in kwargs},
            )
        )
    read_dduf_file(archive_path)
    temporary = tempfile.TemporaryDirectory(prefix="vllm-omni-dduf-")
    try:
        directory = Path(temporary.name)
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            seen = set()
            for info in members:
                name = info.filename
                parts = name.split("/")
                mode = stat.S_IFMT(info.external_attr >> 16)
                if (
                    name != info.orig_filename
                    or any(part in ("", ".", "..") or ":" in part or "\\" in part for part in parts)
                    or name in seen
                    or info.is_dir()
                    or mode not in (0, stat.S_IFREG)
                    or info.external_attr & 0x10
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.flag_bits & 1
                    or not (directory / name).resolve().is_relative_to(directory)
                ):
                    raise ValueError(f"Unsupported DDUF archive entry: {name!r}")
                seen.add(name)
            required = sum(info.file_size for info in members)
            if required > shutil.disk_usage(directory).free:
                raise OSError(errno.ENOSPC, f"DDUF materialization requires {required} free temporary bytes")
            for info in members:
                destination = directory / info.filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, destination.open("xb") as target:
                    # ZipExtFile verifies CRC as each bounded stream reaches EOF.
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        kwargs.pop("dduf_entries", None)
        kwargs["local_files_only"] = True
        pipeline = loader_class.from_pretrained(str(directory), **kwargs)
        pipeline._omni_dduf_cleanup = weakref.finalize(pipeline, temporary.cleanup)
        return pipeline
    except BaseException:
        temporary.cleanup()
        raise


def cleanup_dduf_pipeline(pipeline: DiffusionPipeline | None) -> None:
    """Roll back a materialized directory when later adapter setup fails."""
    cleanup = getattr(pipeline, "_omni_dduf_cleanup", None)
    if cleanup is not None:
        cleanup()
