# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os

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


def test_patch_hf_snapshot_download_leaves_hf_home_when_omitted(monkeypatch):
    monkeypatch.setenv("HF_HOME", "/original-hf-home")
    patch_hf_snapshot_download(monkeypatch, lambda *args, **kwargs: "ok")
    assert os.environ["HF_HOME"] == "/original-hf-home"
