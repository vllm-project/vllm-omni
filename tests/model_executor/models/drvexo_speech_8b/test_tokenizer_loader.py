# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""TTS subfolder loading must preserve the registered checkpoint loader."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from vllm.config import LoadConfig
from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader
from vllm.model_executor.models.utils import WeightsMapper

from vllm_omni.model_executor.models.drvexo_speech_8b.checkpoint import FORMAT, VIEW_FORMAT
from vllm_omni.model_executor.models.drvexo_speech_8b.loader import DRVEXOUnifiedModelLoader
from vllm_omni.model_executor.models.qwen3_tts import first_frame_decoder, qwen3_tts_code2wav, qwen3_tts_talker

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class ModelConfig:
    revision: str | None = None


@dataclass
class Config:
    load_config: LoadConfig
    model_config: ModelConfig = field(default_factory=ModelConfig)


class SubfolderReachedError(Exception):
    """Stop before decoder runtime setup, after actual tensor loading."""


class Stage(torch.nn.Module):
    """Only the fields consumed before subfolder loading are needed here."""

    def __init__(self, root: Path, config: Config):
        super().__init__()
        self.model_path = str(root)
        self.vllm_config = config
        self.hf_to_vllm_mapper = WeightsMapper()
        self.tokenizer_subfolder = "speech_tokenizer"


@pytest.mark.parametrize("entry", ["talker", "codec", "first_frame"])
@pytest.mark.parametrize("load_format", ["drvexo_unified", "safetensors", "dummy"])
def test_speech_tokenizer_uses_selected_loader(tmp_path, mocker, entry, load_format):
    expected = torch.arange(4, dtype=torch.float32)
    folder = tmp_path / "speech_tokenizer"
    folder.mkdir()
    if load_format == "drvexo_unified":
        key = "speech_codec.decoder.weight"
        save_file({key: expected}, tmp_path / "model.safetensors")
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"format": FORMAT}, "weight_map": {key: "model.safetensors"}})
        )
        (tmp_path / "components.json").write_text(
            json.dumps(
                {"format": FORMAT, "components": {"speech_codec": {"prefix": "speech_codec.", "tensor_count": 1}}}
            )
        )
        (folder / "weights.json").write_text(
            json.dumps({"format": VIEW_FORMAT, "index": "../model.safetensors.index.json", "component": "speech_codec"})
        )
    else:
        save_file({"decoder.weight": expected}, folder / "model.safetensors")
    config = Config(LoadConfig(load_format=load_format))
    loader = get_model_loader(config.load_config)
    assert isinstance(loader, DRVEXOUnifiedModelLoader) == (load_format == "drvexo_unified")
    iterator = mocker.spy(
        loader if isinstance(loader, DefaultModelLoader) else DefaultModelLoader, "_get_weights_iterator"
    )
    module = {"talker": qwen3_tts_talker, "codec": qwen3_tts_code2wav, "first_frame": first_frame_decoder}[entry]
    factory = mocker.patch.object(module, "get_model_loader", return_value=loader)
    calls = 0

    def consume(weights, **kwargs):
        nonlocal calls
        calls += 1
        values = dict(weights)
        if entry == "talker" and calls == 1:
            assert values == {}
            return set()
        assert set(values) == {"decoder.weight"}
        assert torch.equal(values["decoder.weight"], expected)
        raise SubfolderReachedError

    mocker.patch.object(module.AutoWeightsLoader, "load_weights", side_effect=consume)
    stage = Stage(tmp_path, config)
    if entry == "talker":
        stage._prompt_builder = mocker.Mock()
    with pytest.raises(SubfolderReachedError):
        if entry == "talker":
            module.Qwen3TTSTalkerForConditionalGeneration.load_weights(stage, iter(()))
        elif entry == "codec":
            module.Qwen3TTSCode2Wav.load_weights(stage, iter(()))
        else:
            module.Qwen3TTSFirstFrameDecoder.load(stage, config)
    factory.assert_called_once_with(config.load_config)
    source = iterator.call_args.args[-1]
    assert source.model_or_path == str(tmp_path)
    assert source.subfolder == "speech_tokenizer"
