# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``omni_snapshot_download`` must honor vLLM's ``VLLM_USE_MODELSCOPE`` semantics.

vLLM treats the flag as enabled only for the literal string ``"true"``
(case-insensitive). Reading ``os.environ`` directly made every non-empty value
truthy, so an explicit opt-out such as ``VLLM_USE_MODELSCOPE=0`` still took the
ModelScope path.
"""

import sys
import types

import pytest
from vllm import envs

from vllm_omni.entrypoints import omni_base

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


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

    def run() -> str:
        picked.clear()
        omni_base.omni_snapshot_download(MODEL_ID)
        return picked[0] if picked else "none"

    return run


@pytest.mark.parametrize("value", ["0", "1", "False", "false", "no", "off"])
def test_non_true_values_do_not_enable_modelscope(monkeypatch, download_backend, value):
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", value)

    assert envs.VLLM_USE_MODELSCOPE is False
    assert download_backend() == "huggingface"


@pytest.mark.parametrize("value", ["true", "True", "TRUE"])
def test_true_values_enable_modelscope(monkeypatch, download_backend, value):
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", value)

    assert envs.VLLM_USE_MODELSCOPE is True
    assert download_backend() == "modelscope"


def test_unset_defaults_to_huggingface(monkeypatch, download_backend):
    monkeypatch.delenv("VLLM_USE_MODELSCOPE", raising=False)

    assert download_backend() == "huggingface"
