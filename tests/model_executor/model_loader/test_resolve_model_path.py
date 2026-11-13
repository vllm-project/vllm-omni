# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for resolve_model_to_local_path."""

from __future__ import annotations

from types import SimpleNamespace

import huggingface_hub
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from vllm_omni.model_executor.model_loader import weight_utils
from vllm_omni.model_executor.model_loader.weight_utils import resolve_model_to_local_path

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Both resolution paths are patched in every test, so nothing is ever fetched
REPO_ID = "test-org/test-model"
RESOLVED = "/cache/snapshots/deadbeef"


@pytest.fixture
def paths(mocker):
    """The resolver's two ways out, both patched."""
    return SimpleNamespace(
        cache_only=mocker.patch.object(huggingface_hub, "snapshot_download", return_value=RESOLVED),
        download=mocker.patch.object(weight_utils, "download_weights_from_hf_specific", return_value=RESOLVED),
    )


@pytest.mark.parametrize("allow_download", [False, True])
def test_local_directory_short_circuits(tmp_path, paths, allow_download):
    """A local dir resolves to itself without either path running."""
    assert resolve_model_to_local_path(str(tmp_path), allow_download=allow_download) == str(tmp_path)

    paths.cache_only.assert_not_called()
    paths.download.assert_not_called()


@pytest.mark.parametrize(
    ("kwargs", "allow_patterns", "cache_dir"),
    [
        ({}, None, None),
        ({"allow_patterns": ["subdir/*"], "cache_dir": "/cache"}, ["subdir/*"], "/cache"),
    ],
)
def test_repo_id_is_cache_only_by_default(paths, kwargs, allow_patterns, cache_dir):
    """Without `allow_download` nothing is fetched, and narrowing is forwarded."""
    assert resolve_model_to_local_path(REPO_ID, **kwargs) == RESOLVED

    paths.cache_only.assert_called_once_with(
        REPO_ID,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
        local_files_only=True,
    )
    paths.download.assert_not_called()


def test_cache_miss_propagates(paths):
    """A failed resolution raises instead of degrading into the raw repo id."""
    paths.cache_only.side_effect = LocalEntryNotFoundError("cold cache")

    with pytest.raises(LocalEntryNotFoundError):
        resolve_model_to_local_path(REPO_ID)

    paths.download.assert_not_called()


@pytest.mark.parametrize(
    ("kwargs", "allow_patterns"),
    [
        ({}, ["*"]),
        ({"allow_patterns": ["subdir/model*.safetensors"]}, ["subdir/model*.safetensors"]),
    ],
)
def test_download_routes_through_the_locked_helper(paths, kwargs, allow_patterns):
    """Downloads go through the vLLM helper holding the node-wide lock.

    That helper is also what honors `VLLM_USE_MODELSCOPE`;
    calling `snapshot_download` directly would drop both.
    """
    assert resolve_model_to_local_path(REPO_ID, allow_download=True, cache_dir="/cache", **kwargs) == RESOLVED

    paths.download.assert_called_once_with(
        model_name_or_path=REPO_ID,
        cache_dir="/cache",
        allow_patterns=allow_patterns,
        require_all=True,
    )
    paths.cache_only.assert_not_called()
