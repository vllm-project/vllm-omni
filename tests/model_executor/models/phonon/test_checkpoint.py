# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_official_checkpoint_streaming_smoke():
    checkpoint = os.environ.get("PHONON_CHECKPOINT_DIR") or "gradium/phonon-31m-en"
    import sentencepiece
    from safetensors.torch import load_file
    from transformers.utils.hub import cached_file

    root = (
        Path(__file__).resolve().parents[4] / "vllm_omni/model_executor/models/phonon"
    )
    spec = importlib.util.spec_from_file_location(
        "_phonon_checkpoint_test",
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    from _phonon_checkpoint_test.mimi_decoder import MimiStreamingDecoder
    from _phonon_checkpoint_test.phonon_talker import PhononFlowLM
    from _phonon_checkpoint_test.prompt import build_prompt

    config = json.loads(Path(cached_file(checkpoint, "config.json")).read_text())
    weights = load_file(cached_file(checkpoint, "model.safetensors"))
    lm = PhononFlowLM(config).eval()
    lm.load_official_weights(weights)
    decoder = MimiStreamingDecoder(config["mimi"], 1).eval()
    decoder.load_official_weights(weights)
    voice = load_file(cached_file(checkpoint, "default-voice.safetensors"))[
        "speaker_wavs"
    ]
    tokenizer = sentencepiece.SentencePieceProcessor(
        model_file=cached_file(checkpoint, "tokenizer.model")
    )
    slots = torch.tensor([0])
    caches = [
        [torch.empty(0, 8, 64), torch.empty(0, 8, 64)] for _ in lm.transformer.layers
    ]
    torch.manual_seed(53)
    with torch.inference_mode():
        prompt = build_prompt(lm, tokenizer.encode("Hello world."), voice)
        hidden = lm.transformer(prompt, torch.arange(prompt.shape[0]), caches)
        audio = []
        latents = []
        countdown = None
        for step in range(128):
            latent, eos = lm.sample(hidden[-1:], torch.randn(1, 32) * 0.8**0.5)
            denormalized = (latent * lm.emb_std + lm.emb_mean).unsqueeze(-1)
            latents.append(denormalized)
            pcm = decoder(
                denormalized, slots, torch.tensor([step * decoder.upsample_ratio])
            )
            assert pcm.shape == (1, 1, 1920)
            assert torch.isfinite(pcm).all() and torch.isfinite(eos).all()
            audio.append(pcm.flatten())
            if countdown is None and eos.item() > config["eos_threshold"]:
                countdown = 3
            if countdown == 0:
                break
            if countdown is not None:
                countdown -= 1
            feedback = lm.input_linear(latent) + lm.conditioning()
            hidden = lm.transformer(
                feedback, torch.tensor([prompt.shape[0] + step]), caches
            )
        assert countdown == 0, "Smoke request did not finish its EOS tail"
        waveform = torch.cat(audio)
        assert waveform.square().mean() > 0
        decoder.reset_slots(slots)
        complete = decoder(torch.cat(latents, -1), slots, torch.tensor([0])).flatten()
        torch.testing.assert_close(complete, waveform, atol=2e-5, rtol=2e-4)
    sys.modules.pop(spec.name)
