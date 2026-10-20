# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LongCat-Next checkpoint snapshot resolution.

vllm_config.model_config.model can be a Hub ID rather than a local
directory, and every raw file access in the decoder stages (generation_config.json,
model.safetensors.index.json shards, WEIGHT_PATH placeholder resolution) needs
the actual local snapshot path. These tests cover resolve_checkpoint_snapshot
without any network access: a local directory must pass through untouched, and
a Hub ID must go through huggingface_hub's snapshot_download with the requested
revision.
"""

import pytest

from vllm_omni.model_executor.models.longcat_next.longcat_next_utils import (
    resolve_checkpoint_snapshot,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_HUB_ID = "meituan-longcat/LongCat-Next"


class _StubApi:
    def __init__(self):
        self.calls = []

    def snapshot_download(self, repo_id, **kwargs):
        self.calls.append((repo_id, kwargs))
        return f"/hf-cache/hub/models--{repo_id.replace('/', '--')}/snapshots/abc123"


def _install_stub(monkeypatch):
    stub = _StubApi()
    monkeypatch.setattr("vllm_omni.transformers_utils.repo_utils.hf_api", lambda: stub)
    return stub


def test_local_directory_passes_through_unchanged(tmp_path, monkeypatch):
    stub = _install_stub(monkeypatch)

    resolved = resolve_checkpoint_snapshot(str(tmp_path))

    assert resolved == str(tmp_path)
    assert stub.calls == []


def test_local_directory_passes_through_even_with_revision(tmp_path, monkeypatch):
    stub = _install_stub(monkeypatch)

    resolved = resolve_checkpoint_snapshot(str(tmp_path), revision="v1.0")

    assert resolved == str(tmp_path)
    assert stub.calls == []


def test_hub_id_resolves_through_snapshot_download(monkeypatch):
    stub = _install_stub(monkeypatch)

    resolved = resolve_checkpoint_snapshot(_HUB_ID, revision="v1.0")

    assert resolved == f"/hf-cache/hub/models--{_HUB_ID.replace('/', '--')}/snapshots/abc123"
    assert stub.calls == [(_HUB_ID, {"revision": "v1.0"})]


def test_hub_id_without_revision_passes_none(monkeypatch):
    stub = _install_stub(monkeypatch)

    resolved = resolve_checkpoint_snapshot(_HUB_ID)

    assert resolved == f"/hf-cache/hub/models--{_HUB_ID.replace('/', '--')}/snapshots/abc123"
    assert stub.calls == [(_HUB_ID, {"revision": None})]
