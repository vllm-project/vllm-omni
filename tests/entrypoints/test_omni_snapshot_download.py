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
