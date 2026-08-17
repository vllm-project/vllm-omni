# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib

import huggingface_hub
import pytest

from vllm_omni.diffusion.model_loader import hub_prefetch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_prefetch_subfolders_propagates_revision(monkeypatch):
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(hub_prefetch, "_repo_prefetch_lock", lambda _model: contextlib.nullcontext())

    hub_prefetch.prefetch_subfolders(
        "org/model",
        ["transformer"],
        revision="pinned-revision",
    )

    assert calls == [
        {
            "repo_id": "org/model",
            "revision": "pinned-revision",
            "allow_patterns": ["transformer/*", "transformer/**", "*.json", "*.txt"],
        }
    ]


def test_from_pretrained_retry_preserves_revision(monkeypatch):
    factory_calls = []
    prefetch_calls = []

    def fake_factory(model, **kwargs):
        factory_calls.append((model, kwargs))
        if len(factory_calls) == 1:
            raise OSError("partial cache")
        return "loaded"

    def fake_prefetch(model, subfolders, **kwargs):
        prefetch_calls.append((model, tuple(subfolders), kwargs))

    monkeypatch.setattr(hub_prefetch, "prefetch_subfolders", fake_prefetch)
    monkeypatch.setattr(hub_prefetch.time, "sleep", lambda _seconds: None)

    result = hub_prefetch.from_pretrained_with_prefetch(
        fake_factory,
        "org/model",
        subfolder="transformer",
        prefetch_list=["transformer", "vae"],
        revision="pinned-revision",
        max_attempts=2,
    )

    assert result == "loaded"
    assert len(factory_calls) == 2
    assert all(call[1]["revision"] == "pinned-revision" for call in factory_calls)
    assert prefetch_calls == [
        (
            "org/model",
            ("transformer", "vae"),
            {"local_files_only": False, "revision": "pinned-revision"},
        )
    ]
