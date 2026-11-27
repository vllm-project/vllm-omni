# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Dataset sources shared by reproducible benchmark scenarios."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSource:
    repo: str
    revision: str
    manifest: str
    sha256: str
    count: int
    root_env: str = ""

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> DatasetSource:
        fields: dict[str, str] = {}
        for key in ("repo", "revision", "manifest", "sha256"):
            value = config.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"dataset_source.{key} must be a non-empty string")
            fields[key] = value
        count = config.get("count")
        if type(count) is not int or count <= 0:
            raise ValueError("dataset_source.count must be a positive integer")
        root_env = config.get("root_env", "")
        if not isinstance(root_env, str):
            raise ValueError("dataset_source.root_env must be a string")
        manifest = Path(fields["manifest"])
        if manifest.is_absolute() or ".." in manifest.parts:
            raise ValueError("dataset_source.manifest must stay within the dataset root")
        return cls(fields["repo"], fields["revision"], fields["manifest"], fields["sha256"], count, root_env)

    def resolve(self, allow_patterns: list[str]) -> Path:
        local = os.environ.get(self.root_env) if self.root_env else None
        if local:
            root = Path(local).expanduser().resolve()
        else:
            from vllm_omni.transformers_utils.repo_utils import hf_api

            root = Path(
                hf_api().snapshot_download(
                    repo_id=self.repo, repo_type="dataset", revision=self.revision, allow_patterns=allow_patterns
                )
            )
        self.verify(root)
        return root

    def verify(self, root: Path) -> None:
        path = root / self.manifest
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != self.sha256:
            raise ValueError(f"{path}: manifest differs from pinned {self.revision}; got SHA256 {actual}")
