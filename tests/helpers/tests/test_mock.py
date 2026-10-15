# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
from pathlib import Path

import pytest

from tests.helpers.mock import patch_hf_snapshot_download

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_patch_hf_snapshot_download_binds_instance_without_self(monkeypatch):
    from vllm_omni.transformers_utils.repo_utils import hf_api

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return "ok"

    patch_hf_snapshot_download(monkeypatch, fake)
    assert hf_api().snapshot_download("org/model", revision="abc") == "ok"
    assert calls == [(("org/model",), {"revision": "abc"})]


def test_patch_hf_snapshot_download_sets_hf_home(monkeypatch, tmp_path):
    hf_home = tmp_path / "hf-home"
    patch_hf_snapshot_download(monkeypatch, lambda *args, **kwargs: "ok", hf_home=hf_home)
    assert os.environ["HF_HOME"] == str(hf_home)
    from huggingface_hub import constants

    assert constants.HF_HOME == str(hf_home)
    assert constants.HF_HUB_CACHE == str(hf_home / "hub")
    assert constants.HUGGINGFACE_HUB_CACHE == str(hf_home / "hub")
    assert constants.default_cache_path == str(hf_home / "hub")


def test_patch_hf_snapshot_download_leaves_hf_home_when_omitted(monkeypatch):
    from huggingface_hub import constants

    monkeypatch.setenv("HF_HOME", "/original-hf-home")
    original_cache = constants.HF_HUB_CACHE
    original_default = constants.default_cache_path
    patch_hf_snapshot_download(monkeypatch, lambda *args, **kwargs: "ok")
    assert os.environ["HF_HOME"] == "/original-hf-home"
    assert constants.HF_HUB_CACHE == original_cache
    assert constants.default_cache_path == original_default


def test_patch_hf_snapshot_download_overrides_preimported_hub_cache(monkeypatch, tmp_path):
    """Hub is imported before the helper; ``HF_HOME`` env alone does not move the cache.

    ``repo_utils`` / ``huggingface_hub`` compute ``constants.HF_HUB_CACHE`` at
    import time. Weekly CPU already imported them with ``HF_HOME=/fsx/hf_cache``.
    A download that bypasses the instance mock uses that constant, not the env.
    """
    from huggingface_hub import constants, snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    from vllm_omni.transformers_utils.repo_utils import hf_api

    hf_api()

    poisoned_home = tmp_path / "fsx-hf-cache"
    poisoned_cache = poisoned_home / "hub"
    repo_dir = poisoned_cache / "models--org--model"
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text("deadbeef", encoding="utf-8")
    snapshot = repo_dir / "snapshots" / "deadbeef"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setenv("HF_HOME", str(poisoned_home))
    monkeypatch.setattr(constants, "HF_HOME", str(poisoned_home))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(poisoned_cache))
    monkeypatch.setattr(constants, "HUGGINGFACE_HUB_CACHE", str(poisoned_cache))

    found = snapshot_download("org/model", local_files_only=True)
    assert Path(found).resolve() == snapshot.resolve()

    isolated = tmp_path / "isolated-hf-home"
    patch_hf_snapshot_download(monkeypatch, lambda *args, **kwargs: "mocked", hf_home=isolated)

    assert os.environ["HF_HOME"] == str(isolated)
    assert os.environ["HF_HUB_CACHE"] == str(isolated / "hub")
    assert constants.HF_HOME == str(isolated)
    assert constants.HF_HUB_CACHE == str(isolated / "hub")
    assert constants.HUGGINGFACE_HUB_CACHE == str(isolated / "hub")
    assert constants.default_cache_path == str(isolated / "hub")
    with pytest.raises(LocalEntryNotFoundError):
        snapshot_download("org/model", local_files_only=True)
