# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``omni_snapshot_download`` must honor vLLM's ``VLLM_USE_MODELSCOPE`` semantics.

vLLM treats the flag as enabled for the literal strings ``"1"`` or ``"true"``
(case-insensitive; see vllm.envs). Reading ``os.environ`` directly made every
non-empty value truthy, so an explicit opt-out such as ``VLLM_USE_MODELSCOPE=0``
still took the ModelScope path.
"""

import sys
import types
from pathlib import Path

import httpx
import huggingface_hub
import pytest
from pytest_mock import MockerFixture
from vllm import envs

from vllm_omni.entrypoints import omni_base

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


@pytest.mark.parametrize(
    "model_uri",
    [
        "s3://bucket/model",
        "gs://bucket/model",
        "az://bucket/model",
    ],
)
def test_omni_snapshot_download_preserves_object_storage_uri(
    model_uri: str,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_USE_MODELSCOPE", raising=False)
    hf_download = mocker.patch.object(omni_base, "download_weights_from_hf_specific")

    assert omni_base.omni_snapshot_download(model_uri) == model_uri
    hf_download.assert_not_called()


def test_omni_snapshot_download_preserves_existing_local_path(
    tmp_path: Path,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_USE_MODELSCOPE", raising=False)
    absolute_model_path = tmp_path / "absolute-model"
    absolute_model_path.mkdir()
    relative_model_path = "relative-model"
    (tmp_path / relative_model_path).mkdir()
    monkeypatch.chdir(tmp_path)
    hf_download = mocker.patch.object(omni_base, "download_weights_from_hf_specific")

    assert omni_base.omni_snapshot_download(str(absolute_model_path)) == str(absolute_model_path)
    assert omni_base.omni_snapshot_download(relative_model_path) == relative_model_path
    hf_download.assert_not_called()


def test_omni_snapshot_download_uses_hf_for_relative_repo_id(
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_USE_MODELSCOPE", raising=False)
    model_id = "org/model"
    # Stub the modular-index probe so the test never reaches huggingface.co.
    hf_file_probe = mocker.patch.object(omni_base, "file_or_path_exists", return_value=False)
    hf_download = mocker.patch.object(omni_base, "download_weights_from_hf_specific")

    assert omni_base.omni_snapshot_download(model_id) == model_id
    hf_file_probe.assert_called_once_with(model_id, "modular_model_index.json", revision=None)
    hf_download.assert_called_once_with(
        model_name_or_path=model_id,
        cache_dir=None,
        allow_patterns=["*"],
        require_all=True,
    )


@pytest.fixture
def download_backend(monkeypatch: pytest.MonkeyPatch):
    """Run ``omni_snapshot_download`` and report which backend it selected.

    ModelScope is not a vLLM-Omni dependency, so the ModelScope branch is stubbed
    into ``sys.modules``; without the stub it would raise ``ModuleNotFoundError``
    instead of being observable.
    """
    # vLLM caches env lookups once a service is initialized; make sure this test
    # reads the values monkeypatch sets rather than a cached snapshot.
    envs.disable_envs_cache()

    picked: list[str] = []

    snapshot_module = types.ModuleType("modelscope.hub.snapshot_download")
    snapshot_module.snapshot_download = lambda model_id: picked.append("modelscope") or model_id
    hub_module = types.ModuleType("modelscope.hub")
    hub_module.snapshot_download = snapshot_module
    root_module = types.ModuleType("modelscope")
    root_module.hub = hub_module
    for name, module in (
        ("modelscope", root_module),
        ("modelscope.hub", hub_module),
        ("modelscope.hub.snapshot_download", snapshot_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setattr(
        omni_base,
        "download_weights_from_hf_specific",
        lambda **_kwargs: picked.append("huggingface"),
    )
    monkeypatch.setattr(
        omni_base,
        "file_or_path_exists",
        lambda *_args, **_kwargs: False,
    )

    def run() -> str:
        picked.clear()
        omni_base.omni_snapshot_download(MODEL_ID)
        return picked[0] if picked else "none"

    return run


@pytest.mark.parametrize("value", ["0", "False", "false", "no", "off"])
def test_non_true_values_do_not_enable_modelscope(monkeypatch, download_backend, value):
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", value)

    assert envs.VLLM_USE_MODELSCOPE is False
    assert download_backend() == "huggingface"


@pytest.mark.parametrize("value", ["1", "true", "True", "TRUE"])
def test_true_values_enable_modelscope(monkeypatch, download_backend, value):
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", value)

    assert envs.VLLM_USE_MODELSCOPE is True
    assert download_backend() == "modelscope"


def test_unset_defaults_to_huggingface(monkeypatch, download_backend):
    monkeypatch.delenv("VLLM_USE_MODELSCOPE", raising=False)

    assert download_backend() == "huggingface"


def test_modular_diffusers_defers_component_download(monkeypatch, download_backend):
    monkeypatch.delenv("VLLM_USE_MODELSCOPE", raising=False)
    monkeypatch.setattr(
        omni_base,
        "file_or_path_exists",
        lambda *_args, **_kwargs: True,
    )

    assert download_backend() == "none"


def _hub_error(
    error_cls: type[huggingface_hub.errors.HfHubHTTPError], status_code: int
) -> huggingface_hub.errors.HfHubHTTPError:
    request = httpx.Request("GET", f"https://huggingface.co/{MODEL_ID}/resolve/main/model_index.json")
    return error_cls(f"{status_code} Client Error", response=httpx.Response(status_code, request=request))


def test_gated_repo_reports_access_instructions(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gated diffusers repos (e.g. FLUX.2-klein-9B, Stable-Audio-Open) must not
    surface as a generic "could not determine model_type" error (#1697)."""
    monkeypatch.delenv("VLLM_USE_MODELSCOPE", raising=False)
    mocker.patch.object(omni_base, "file_or_path_exists", return_value=False)
    mocker.patch.object(
        omni_base,
        "download_weights_from_hf_specific",
        side_effect=_hub_error(huggingface_hub.errors.GatedRepoError, 403),
    )

    with pytest.raises(ValueError, match="is restricted") as exc_info:
        omni_base.omni_snapshot_download(MODEL_ID)

    assert f"https://huggingface.co/{MODEL_ID}" in str(exc_info.value)


def test_gated_modular_index_probe_falls_through_to_access_error(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VLLM_USE_MODELSCOPE", raising=False)
    gated = _hub_error(huggingface_hub.errors.GatedRepoError, 403)
    mocker.patch.object(omni_base, "file_or_path_exists", side_effect=gated)
    hf_download = mocker.patch.object(omni_base, "download_weights_from_hf_specific", side_effect=gated)

    with pytest.raises(ValueError, match="is restricted"):
        omni_base.omni_snapshot_download(MODEL_ID)
    hf_download.assert_called_once()


def test_missing_repo_reports_not_found(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    # GatedRepoError subclasses RepositoryNotFoundError, so this also guards the
    # handler order: a plain 404 must not be reported as an access problem.
    monkeypatch.delenv("VLLM_USE_MODELSCOPE", raising=False)
    mocker.patch.object(omni_base, "file_or_path_exists", return_value=False)
    mocker.patch.object(
        omni_base,
        "download_weights_from_hf_specific",
        side_effect=_hub_error(huggingface_hub.errors.RepositoryNotFoundError, 404),
    )

    with pytest.raises(ValueError, match="Repository not found"):
        omni_base.omni_snapshot_download(MODEL_ID)
