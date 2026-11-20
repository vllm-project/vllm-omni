# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.entrypoints.openai.tts_adapters.phonon import PhononAdapter
from vllm_omni.model_executor.models.phonon.prompt import build_prompt, load_default_voice

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("name", ["emb", "speaker_wavs"])
@pytest.mark.parametrize("batched", [False, True])
def test_adapter_prompt_length_matches_worker_voice(tmp_path, monkeypatch, name, batched):
    frames, width = 3, 8
    voice = torch.arange(frames * width, dtype=torch.float32).reshape(frames, width)
    stored = voice.T if name == "speaker_wavs" else voice
    if batched:
        stored = stored.unsqueeze(0)
    path = tmp_path / "default-voice.safetensors"
    save_file({name: stored.contiguous()}, str(path))
    monkeypatch.setattr("transformers.utils.hub.cached_file", lambda *args, **kwargs: str(path))
    monkeypatch.setattr("vllm_omni.model_executor.models.phonon.prompt.cached_file", lambda *args, **kwargs: str(path))
    tokenizer = SimpleNamespace(encode=lambda text: [1, 2])
    monkeypatch.setitem(
        sys.modules, "sentencepiece", SimpleNamespace(SentencePieceProcessor=lambda **kwargs: tokenizer)
    )
    config = SimpleNamespace(model="test", revision=None, hf_config=SimpleNamespace(to_dict=lambda: {}))
    adapter = PhononAdapter(SimpleNamespace(engine_client=SimpleNamespace(model_config=config)))
    monkeypatch.setattr(
        "vllm_omni.entrypoints.openai.tts_adapters.phonon.conditioning_cache_salt", lambda request: "test"
    )
    request = adapter._build(SimpleNamespace(input="Hello world")).prompt

    lm = SimpleNamespace(
        bos_emb=torch.zeros(4),
        speaker_projection=lambda inputs: inputs * 2,
        text_embedding=torch.nn.Embedding(4, width),
        input_linear=torch.nn.Linear(4, width, bias=False),
    )
    projected = load_default_voice(config, lm)
    torch.testing.assert_close(projected, voice * 2 if name == "speaker_wavs" else voice)
    prompt = build_prompt(
        lm,
        request["additional_information"]["phonon_text_ids"],
        projected,
        project_voice=False,
        conditioning=torch.zeros(width),
    )
    assert prompt.shape[0] == len(request["prompt_token_ids"]) == frames + 2 + 1
    torch.testing.assert_close(prompt[:frames], projected)
