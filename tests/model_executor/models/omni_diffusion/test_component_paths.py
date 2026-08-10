# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Omni-Diffusion side-component path resolution."""

from pathlib import Path
from unittest.mock import patch

import pytest

from vllm_omni.model_executor.models.omni_diffusion.component_paths import (
    resolve_omni_diffusion_component_path,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_resolve_omni_diffusion_component_path_prefers_local_directory(tmp_path: Path) -> None:
    component_dir = tmp_path / "component"
    component_dir.mkdir()

    with patch("vllm_omni.model_executor.models.omni_diffusion.component_paths.snapshot_download") as download:
        result = resolve_omni_diffusion_component_path(
            str(component_dir),
            config_key="image_tokenizer_path",
            default_repo_id="org/default-component",
        )

    assert result == str(component_dir.resolve())
    download.assert_not_called()


@pytest.mark.parametrize("configured_path", [None, "", "   "])
def test_resolve_omni_diffusion_component_path_downloads_default_snapshot(configured_path: str | None) -> None:
    with patch(
        "vllm_omni.model_executor.models.omni_diffusion.component_paths.snapshot_download",
        return_value="/cache/component",
    ) as download:
        result = resolve_omni_diffusion_component_path(
            configured_path,
            config_key="sensevoice_path",
            default_repo_id="org/default-component",
        )

    assert result == "/cache/component"
    download.assert_called_once_with(repo_id="org/default-component")


def test_resolve_omni_diffusion_component_path_rejects_missing_local_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="existing local directory"):
        resolve_omni_diffusion_component_path(
            str(tmp_path / "missing"),
            config_key="flow_path",
            default_repo_id="org/default-component",
        )


def test_resolve_omni_diffusion_component_path_rejects_invalid_type() -> None:
    with pytest.raises(TypeError, match="local directory path or null"):
        resolve_omni_diffusion_component_path(
            42,
            config_key="flow_path",
            default_repo_id="org/default-component",
        )
