# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cosmos3 sound tokenizer path precedence: custom_pipeline_args > deprecated env (#6232)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3 import sound_tokenizer
from vllm_omni.diffusion.models.cosmos3.sound_tokenizer import Cosmos3SoundTokenizerPaths

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

PATH_ENV = "COSMOS3_SOUND_TOKENIZER_PATH"
CONFIG_PATH_ENV = "COSMOS3_SOUND_TOKENIZER_CONFIG_PATH"


@pytest.fixture
def warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    messages: list[str] = []
    monkeypatch.setattr(sound_tokenizer.logger, "warning_once", lambda msg, *args: messages.append(msg % args))
    monkeypatch.delenv(PATH_ENV, raising=False)
    monkeypatch.delenv(CONFIG_PATH_ENV, raising=False)
    return messages


def _paths(custom_pipeline_args: dict[str, Any] | None) -> Cosmos3SoundTokenizerPaths:
    return Cosmos3SoundTokenizerPaths.from_config(SimpleNamespace(custom_pipeline_args=custom_pipeline_args))


def test_defaults_to_none_without_warning(warnings: list[str]) -> None:
    assert _paths(None) == Cosmos3SoundTokenizerPaths(checkpoint_path=None, config_path=None)
    assert warnings == []


def test_pipeline_args_win_over_env_without_warning(warnings: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PATH_ENV, "/env/avae.safetensors")
    monkeypatch.setenv(CONFIG_PATH_ENV, "/env/config.json")

    paths = _paths(
        {
            "sound_tokenizer_path": Path("/args/avae.safetensors"),
            "sound_tokenizer_config_path": "/args/config.json",
        }
    )

    assert paths == Cosmos3SoundTokenizerPaths("/args/avae.safetensors", "/args/config.json")
    assert warnings == []


def test_legacy_env_still_applies_and_warns(warnings: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PATH_ENV, "/env/avae.safetensors")
    monkeypatch.setenv(CONFIG_PATH_ENV, "/env/config.json")

    paths = _paths({})

    assert paths == Cosmos3SoundTokenizerPaths("/env/avae.safetensors", "/env/config.json")
    assert len(warnings) == 2
    assert f"{PATH_ENV} is deprecated" in warnings[0]
    assert "'sound_tokenizer_path'" in warnings[0]
    assert f"{CONFIG_PATH_ENV} is deprecated" in warnings[1]
    assert "'sound_tokenizer_config_path'" in warnings[1]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"sound_tokenizer_path": "a", "avae_path": "b", "cosmos3_avae_path": "c"}, "a"),
        ({"sound_tokenizer_path": "", "avae_path": "b", "cosmos3_avae_path": "c"}, "b"),
        ({"cosmos3_avae_path": "c"}, "c"),
    ],
)
def test_checkpoint_alias_precedence(warnings: list[str], args: dict[str, Any], expected: str) -> None:
    assert _paths(args).checkpoint_path == expected


@pytest.mark.parametrize("key", ["sound_tokenizer_path", "sound_tokenizer_config_path"])
def test_rejects_non_path_values(warnings: list[str], key: str) -> None:
    with pytest.raises(TypeError, match=key):
        _paths({key: 123})


def test_tokenizer_still_loads_checkpoint_from_legacy_env(
    warnings: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "avae.safetensors"
    checkpoint_path.write_bytes(b"stub")
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(PATH_ENV, str(checkpoint_path))
    monkeypatch.setenv(CONFIG_PATH_ENV, str(config_path))
    created: dict[str, Any] = {}

    def fake_avae(**kwargs: Any) -> SimpleNamespace:
        created.update(kwargs)
        return SimpleNamespace(sample_rate=kwargs["sample_rate"], temporal_compression_factor=kwargs["hop_size"])

    monkeypatch.setattr(sound_tokenizer, "Cosmos3AVAEAudioTokenizer", fake_avae)
    monkeypatch.setattr(sound_tokenizer, "get_local_device", lambda: torch.device("cpu"))

    sound_tokenizer.Cosmos3SoundTokenizer.from_config(
        SimpleNamespace(model=None, custom_pipeline_args={}, dtype=torch.float32)
    )

    assert created["checkpoint_path"] == str(checkpoint_path)
    assert created["config_path"] == str(config_path)
    assert len(warnings) == 2
