# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import contextlib

import pytest
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

from tests.helpers.mock import patch_hf_snapshot_download
from vllm_omni.diffusion.model_loader import hub_prefetch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def prefetch_io(mocker, monkeypatch, tmp_path):
    download = mocker.Mock()
    patch_hf_snapshot_download(monkeypatch, download, hf_home=tmp_path)
    mocker.patch.object(hub_prefetch, "_repo_prefetch_lock", side_effect=lambda _: contextlib.nullcontext())
    sleep = mocker.patch.object(hub_prefetch.time, "sleep")
    return download, sleep


@pytest.mark.parametrize(
    "model,local_files_only",
    [("org/model", True), ("", None), ("local", None), ("local", False)],
    ids=["offline", "empty-model", "local-auto", "local-explicit"],
)
def test_prefetch_skips_local_and_offline_models(prefetch_io, tmp_path, model, local_files_only):
    download, sleep = prefetch_io
    if model == "local":
        model = str(tmp_path)

    hub_prefetch.prefetch_subfolders(model, ["transformer"], local_files_only=local_files_only)

    download.assert_not_called()
    sleep.assert_not_called()


def test_prefetch_empty_selection_does_not_download(prefetch_io):
    download, sleep = prefetch_io

    hub_prefetch.prefetch_subfolders("org/model", ["", "/"], include_root_metadata=False)

    download.assert_not_called()
    sleep.assert_not_called()


@pytest.mark.parametrize("succeeds", [True, False], ids=["recovers", "exhausted"])
def test_prefetch_retries_transient_failures(prefetch_io, mocker, caplog, succeeds):
    download, sleep = prefetch_io
    download.side_effect = [
        OSError("partial cache"),
        OSError("partial cache"),
        None if succeeds else OSError("still busy"),
    ]

    # A one-shot iterable must survive retries, without broadening the download.
    hub_prefetch.prefetch_subfolders(
        "org/model",
        iter(["/transformer/", "vae"]),
        revision="pinned-revision",
        include_root_metadata=False,
    )

    assert (
        download.call_args_list
        == [
            mocker.call(
                repo_id="org/model",
                revision="pinned-revision",
                allow_patterns=["transformer/*", "transformer/**", "vae/*", "vae/**"],
            )
        ]
        * 3
    )
    assert sleep.call_args_list == [mocker.call(1.0), mocker.call(2.0)]
    if not succeeds:
        assert "failed after 3 attempts" in caplog.text
        assert "still busy" in caplog.text


@pytest.mark.parametrize(
    "error_type,message",
    [
        (GatedRepoError, "license required"),
        (RepositoryNotFoundError, "private repo"),
        (OSError, "401 Client Error: Unauthorized"),
        (OSError, "403 Client Error: Forbidden"),
    ],
    ids=["gated-repo", "private-repo", "wrapped-401", "wrapped-403"],
)
def test_prefetch_does_not_retry_authentication_errors(prefetch_io, mocker, caplog, error_type, message):
    download, sleep = prefetch_io
    if error_type is OSError:
        error = error_type(message)
    else:
        response = mocker.Mock(headers={}, request=None)
        error = error_type(message, response=response)
    download.side_effect = error

    hub_prefetch.prefetch_subfolders("org/model", ["transformer"])

    download.assert_called_once()
    sleep.assert_not_called()
    assert "authentication / gated" in caplog.text
    assert "org/model" in caplog.text


@pytest.mark.parametrize("status", [401, 403, 503])
def test_prefetch_classifies_http_status(prefetch_io, mocker, status):
    download, sleep = prefetch_io
    error = OSError("Hub request failed")
    setattr(error, "response", mocker.Mock(status_code=status))
    download.side_effect = [error, None]

    hub_prefetch.prefetch_subfolders("org/model", ["transformer"])

    if status == 503:
        assert download.call_count == 2
        sleep.assert_called_once_with(1.0)
    else:
        download.assert_called_once()
        sleep.assert_not_called()


def test_prefetch_subfolders_propagates_revision(monkeypatch, tmp_path):
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)

    patch_hf_snapshot_download(monkeypatch, fake_snapshot_download, hf_home=tmp_path)
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


def test_from_pretrained_success_does_not_prefetch_or_retry(mocker):
    loaded = object()
    factory = mocker.Mock(return_value=loaded)
    prefetch = mocker.patch.object(hub_prefetch, "prefetch_subfolders")
    sleep = mocker.patch.object(hub_prefetch.time, "sleep")

    result = hub_prefetch.from_pretrained_with_prefetch(
        factory, "org/model", subfolder="vae", prefetch_list=["vae"], revision="pinned-revision"
    )

    assert result is loaded
    factory.assert_called_once_with("org/model", subfolder="vae", local_files_only=False, revision="pinned-revision")
    prefetch.assert_not_called()
    sleep.assert_not_called()


@pytest.mark.parametrize("source", ["offline", "local", "empty-model"])
def test_from_pretrained_local_failures_are_not_retried(mocker, tmp_path, source):
    model = {"offline": "org/model", "local": str(tmp_path), "empty-model": ""}[source]
    error = OSError("missing local weights")
    factory = mocker.Mock(side_effect=error)
    prefetch = mocker.patch.object(hub_prefetch, "prefetch_subfolders")
    sleep = mocker.patch.object(hub_prefetch.time, "sleep")

    with pytest.raises(OSError) as exc_info:
        hub_prefetch.from_pretrained_with_prefetch(
            factory, model, subfolder="vae", prefetch_list=["vae"], local_files_only=source == "offline"
        )

    assert exc_info.value is error
    factory.assert_called_once_with(model, subfolder="vae", local_files_only=source == "offline")
    prefetch.assert_not_called()
    sleep.assert_not_called()


@pytest.mark.parametrize("error_type", [OSError, RuntimeError, ValueError])
@pytest.mark.parametrize("max_attempts", [1, 3])
def test_from_pretrained_exhaustion_raises_last_error(mocker, error_type, max_attempts):
    errors = [error_type(f"load failure {attempt}") for attempt in range(max_attempts)]
    factory = mocker.Mock(side_effect=errors)
    prefetch = mocker.patch.object(hub_prefetch, "prefetch_subfolders")
    sleep = mocker.patch.object(hub_prefetch.time, "sleep")

    with pytest.raises(error_type) as exc_info:
        hub_prefetch.from_pretrained_with_prefetch(
            factory,
            "org/model",
            subfolder="transformer",
            prefetch_list=iter(["transformer", "vae"]),
            revision="pinned-revision",
            max_attempts=max_attempts,
        )

    assert exc_info.value is errors[-1]
    assert (
        factory.call_args_list
        == [mocker.call("org/model", subfolder="transformer", local_files_only=False, revision="pinned-revision")]
        * max_attempts
    )
    assert prefetch.call_args_list == [
        mocker.call("org/model", ["transformer", "vae"], local_files_only=False, revision="pinned-revision")
    ] * (max_attempts - 1)
    assert sleep.call_args_list == [mocker.call(float(attempt)) for attempt in range(1, max_attempts)]


def test_from_pretrained_unexpected_error_is_not_retried(mocker):
    error = TypeError("invalid loader argument")
    factory = mocker.Mock(side_effect=error)
    prefetch = mocker.patch.object(hub_prefetch, "prefetch_subfolders")
    sleep = mocker.patch.object(hub_prefetch.time, "sleep")

    with pytest.raises(TypeError) as exc_info:
        hub_prefetch.from_pretrained_with_prefetch(factory, "org/model", subfolder="vae", prefetch_list=["vae"])

    assert exc_info.value is error
    factory.assert_called_once()
    prefetch.assert_not_called()
    sleep.assert_not_called()
