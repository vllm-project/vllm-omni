# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Official precomputed-speaker prompt preparation."""

import torch
from safetensors import safe_open
from transformers.utils.hub import cached_file


def load_voice(model_path, config, revision=None):
    filename = cached_file(model_path, "default-voice.safetensors", revision=revision)
    with safe_open(filename, framework="pt", device="cpu") as file:
        metadata = file.metadata() or {}
        allowed = config.get("embeddings_ext", [])
        identifier = f"{metadata.get('sig')}@{metadata.get('epoch')}"
        if allowed and identifier not in allowed:
            raise ValueError(f"Voice metadata does not match embeddings_ext: {metadata}")
        keys = list(file.keys())
        name = "speaker_wavs" if "speaker_wavs" in keys else "emb"
        return file.get_tensor(name), name == "speaker_wavs"


def prepare_text(text):
    text = " ".join(text.split())
    if not text:
        raise ValueError("Phonon input text cannot be empty")
    tail = 3 if len(text.split()) <= 4 else 1
    text = text[0].upper() + text[1:]
    if text[-1].isalnum():
        text += "."
    return text, tail


def build_prompt(lm, text_ids, voice, *, project_voice=True):
    parameter = lm.bos_emb
    voice = voice.to(device=parameter.device, dtype=parameter.dtype)
    if voice.ndim == 3:
        voice = voice.squeeze(0)
    if project_voice:
        voice = lm.speaker_projection(voice.transpose(0, 1))
    text = lm.text_embedding(torch.as_tensor(text_ids, device=parameter.device, dtype=torch.long))
    bos = lm.input_linear(lm.bos_emb) + lm.conditioning()
    return torch.cat((voice, text, bos[None]), 0)
