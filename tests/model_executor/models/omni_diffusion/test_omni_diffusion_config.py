# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Omni-Diffusion startup and forward configuration."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_omni.model_executor.models.omni_diffusion.omni_diffusion import (
    OmniDiffusionAdditionalConfig,
    OmniDiffusionForwardKwargs,
    _is_dummy_run,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _vllm_config(additional_config: object) -> SimpleNamespace:
    return SimpleNamespace(additional_config=additional_config)


def _component_dirs(tmp_path: Path) -> dict[str, str]:
    paths = {}
    for key in ("image_tokenizer_path", "flow_path", "sensevoice_path"):
        path = tmp_path / key
        path.mkdir()
        paths[key] = str(path)
    return paths


def _task_config(task: str, paths: dict[str, str]) -> dict[str, object]:
    normalized_task = task.upper()
    config: dict[str, object] = {
        "task": task,
        "steps": 64,
        "max_new_tokens": 64,
        "alg": "entropy",
        "cfg": 0.0,
        "temperature": 0.0,
        "top_p": 0.9,
        "add_boa_token": 0,
        "max_position_penalty": 2.0,
        "repeat_penalty": 1.0,
    }
    if normalized_task in {"T2I", "VQA", "SVQA"}:
        config["image_tokenizer_path"] = paths["image_tokenizer_path"]
    if normalized_task in {"ASR", "SVQA"}:
        config["sensevoice_path"] = paths["sensevoice_path"]
    if normalized_task == "TTS":
        config["flow_path"] = paths["flow_path"]
    return config


@pytest.mark.parametrize(
    ("task", "expected_paths"),
    [
        ("T2I", {"image_tokenizer_path"}),
        ("VQA", {"image_tokenizer_path"}),
        ("ASR", {"sensevoice_path"}),
        ("TTS", {"flow_path"}),
        ("SVQA", {"image_tokenizer_path", "sensevoice_path"}),
    ],
)
def test_additional_config_loads_only_task_components(
    tmp_path: Path,
    task: str,
    expected_paths: set[str],
) -> None:
    paths = _component_dirs(tmp_path)
    config = OmniDiffusionAdditionalConfig.from_vllm_config(_vllm_config(_task_config(task.lower(), paths)))

    assert config.task == task
    for key in ("image_tokenizer_path", "flow_path", "sensevoice_path"):
        expected = paths[key] if key in expected_paths else None
        assert getattr(config, key) == expected


@pytest.mark.parametrize(
    ("task", "config_key", "attribute"),
    [
        ("T2I", "additional_config.image_tokenizer_path", "image_tokenizer_path"),
        ("ASR", "additional_config.sensevoice_path", "sensevoice_path"),
        ("TTS", "additional_config.flow_path", "flow_path"),
    ],
)
def test_additional_config_resolves_default_task_component(
    task: str,
    config_key: str,
    attribute: str,
) -> None:
    with patch(
        "vllm_omni.model_executor.models.omni_diffusion.omni_diffusion.resolve_omni_diffusion_component_path",
        return_value="/cache/component",
    ) as resolve:
        config = OmniDiffusionAdditionalConfig.from_vllm_config(_vllm_config({"task": task}))

    assert getattr(config, attribute) == "/cache/component"
    assert resolve.call_args.kwargs["config_key"] == config_key


def test_additional_config_parses_generation_settings(tmp_path: Path) -> None:
    paths = _component_dirs(tmp_path)
    values = _task_config("T2I", paths) | {
        "attn_implementation": "eager",
        "output_text_only": True,
        "seed": 42,
        "steps": 260,
        "max_new_tokens": 512,
        "alg": "entropy-penalty",
        "cfg": 2,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 50,
        "add_boa_token": 1,
        "max_position_penalty": 3,
        "repeat_penalty": 1.2,
    }

    config = OmniDiffusionAdditionalConfig.from_vllm_config(_vllm_config(values))

    assert config.attn_implementation == "eager"
    assert config.output_text_only is True
    assert config.seed == 42
    assert config.steps == 260
    assert config.max_new_tokens == 512
    assert config.alg == "entropy-penalty"
    assert config.cfg == 2.0
    assert config.temperature == 0.8
    assert config.top_p == 0.95
    assert config.top_k == 50
    assert config.add_boa_token == 1
    assert config.max_position_penalty == 3.0
    assert config.repeat_penalty == 1.2


def test_additional_config_uses_generation_defaults(tmp_path: Path) -> None:
    paths = _component_dirs(tmp_path)
    config = OmniDiffusionAdditionalConfig.from_vllm_config(
        _vllm_config(
            {
                "task": "T2I",
                "image_tokenizer_path": paths["image_tokenizer_path"],
            }
        )
    )

    assert config.steps == 128
    assert config.max_new_tokens == 128
    assert config.alg == "entropy"
    assert config.cfg == 0.0
    assert config.temperature == 0.0
    assert config.top_p == 0.9
    assert config.add_boa_token == 0
    assert config.max_position_penalty == 1.0
    assert config.repeat_penalty == 1.0
    assert config.seed is None
    assert config.top_k is None


@pytest.mark.parametrize("task", [None, "", "S2I", "unknown"])
def test_additional_config_rejects_invalid_task(task: object) -> None:
    with pytest.raises(ValueError, match="task"):
        OmniDiffusionAdditionalConfig.from_vllm_config(_vllm_config({"task": task}))


def test_additional_config_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match="mapping"):
        OmniDiffusionAdditionalConfig.from_vllm_config(_vllm_config("not-a-mapping"))


def test_additional_config_rejects_missing_component_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="image_tokenizer_path"):
        OmniDiffusionAdditionalConfig.from_vllm_config(
            _vllm_config(
                {
                    "task": "VQA",
                    "image_tokenizer_path": str(missing),
                }
            )
        )


def test_additional_config_rejects_unsupported_audio_tokenizer(tmp_path: Path) -> None:
    paths = _component_dirs(tmp_path)
    config = _task_config("ASR", paths)
    config["audio_tokenizer_type"] = "unsupported"
    with pytest.raises(ValueError, match="audio_tokenizer_type"):
        OmniDiffusionAdditionalConfig.from_vllm_config(_vllm_config(config))


def test_forward_kwargs_extract_model_inputs_and_leave_unrelated_values() -> None:
    images = torch.rand(3, 8, 8)
    audios = torch.rand(160)
    kwargs = {
        "omni_images": images,
        "omni_audios": audios,
        "omni_audio_sample_rates": 16000,
        "runtime_additional_information": {"_is_dummy": True},
    }

    parsed = OmniDiffusionForwardKwargs.from_forward_kwargs(kwargs)

    assert parsed.omni_images is images
    assert parsed.omni_audios is audios
    assert parsed.omni_audio_sample_rates == 16000
    assert kwargs == {"runtime_additional_information": {"_is_dummy": True}}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"_is_dummy": True}, True),
        ([{"_is_dummy": True}], True),
        ({"_is_dummy": False}, False),
        ([], False),
        (None, False),
    ],
)
def test_is_dummy_run(value: object, expected: bool) -> None:
    assert _is_dummy_run(value) is expected
