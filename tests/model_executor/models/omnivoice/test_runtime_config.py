# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OmniVoice runtime flags: typed additional_config and deprecated env switches (#6232)."""

import pytest

from vllm_omni.diffusion.models.omnivoice import pipeline_omnivoice
from vllm_omni.transformers_utils.configs import omnivoice as omnivoice_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_ENV_NAMES = ("OMNIVOICE_CUDA_GRAPH", "OMNIVOICE_TF32")


@pytest.fixture
def warnings(monkeypatch) -> list[str]:
    messages: list[str] = []
    monkeypatch.setattr(omnivoice_config.logger, "warning_once", lambda msg, *args: messages.append(msg % args))
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    return messages


def test_defaults_without_env_do_not_warn(warnings):
    config = omnivoice_config.OmniVoiceConfig()

    assert config.enable_cuda_graph is True
    assert config.enable_tf32 is False
    assert warnings == []


def test_legacy_env_still_applies_and_warns(warnings, monkeypatch):
    monkeypatch.setenv("OMNIVOICE_CUDA_GRAPH", "0")
    monkeypatch.setenv("OMNIVOICE_TF32", "1")

    config = omnivoice_config.OmniVoiceConfig()

    assert config.enable_cuda_graph is False
    assert config.enable_tf32 is True
    assert len(warnings) == 2
    assert all("is deprecated" in msg and "omnivoice_runtime" in msg for msg in warnings)


def test_config_json_value_wins_over_env_without_warning(warnings, monkeypatch):
    monkeypatch.setenv("OMNIVOICE_CUDA_GRAPH", "0")

    config = omnivoice_config.OmniVoiceConfig(enable_cuda_graph=True)

    assert config.enable_cuda_graph is True
    assert warnings == []


@pytest.mark.parametrize(
    ("additional_config", "expected"),
    [
        (None, {}),
        ({}, {}),
        ({"omnivoice_asr": {"load_asr_on_startup": True}}, {}),
        ({"omnivoice_runtime": {}}, {}),
        ({"omnivoice_runtime": {"enable_cuda_graph": False}}, {"enable_cuda_graph": False}),
        (
            {"omnivoice_runtime": {"enable_cuda_graph": True, "enable_tf32": True}},
            {"enable_cuda_graph": True, "enable_tf32": True},
        ),
    ],
)
def test_runtime_config_values(additional_config, expected):
    assert pipeline_omnivoice._parse_runtime_config(additional_config) == expected


@pytest.mark.parametrize(
    "additional_config",
    [
        [],
        {"omnivoice_runtime": None},
        {"omnivoice_runtime": []},
        {"omnivoice_runtime": {"enable_cuda_graph": "false"}},
        {"omnivoice_runtime": {"enable_tf32": 1}},
        {"omnivoice_runtime": {"enable_cudagraph": False}},
    ],
)
def test_runtime_config_rejects_invalid_values(additional_config):
    with pytest.raises((TypeError, ValueError)):
        pipeline_omnivoice._parse_runtime_config(additional_config)
