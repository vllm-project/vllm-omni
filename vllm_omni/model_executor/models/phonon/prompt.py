# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Official precomputed-speaker prompt preparation."""

import torch
from safetensors import safe_open
from transformers.utils.hub import cached_file

from vllm_omni.utils.device_copy import to_device_nonblocking


def voice_layout(file):
    # Official formats are emb [T, D] / [1, T, D] and
    # speaker_wavs [C, T] / [1, C, T], which requires projection.
    name = "emb" if "emb" in file.keys() else "speaker_wavs"
    if name not in file.keys():
        raise ValueError("Phonon voice requires emb [T, D] or speaker_wavs [C, T], optionally batched")
    shape = file.get_slice(name).get_shape()
    if len(shape) not in (2, 3) or (len(shape) == 3 and shape[0] != 1):
        raise ValueError(f"Invalid Phonon {name} shape {shape}: expected a 2D tensor or batch size 1")
    project = name == "speaker_wavs"
    return name, shape[-1] if project else shape[-2], project


def load_voice(model_path, config, revision=None):
    filename = cached_file(model_path, "default-voice.safetensors", revision=revision)
    with safe_open(filename, framework="pt", device="cpu") as file:
        metadata = file.metadata() or {}
        allowed = config.get("embeddings_ext", [])
        identifier = f"{metadata.get('sig')}@{metadata.get('epoch')}"
        if allowed and identifier not in allowed:
            raise ValueError(f"Voice metadata does not match embeddings_ext: {metadata}")
        name, _, project = voice_layout(file)
        return file.get_tensor(name), project


@torch.inference_mode()
def load_default_voice(model_config, lm):
    voice, project = load_voice(
        model_config.model, model_config.hf_config.to_dict(), getattr(model_config, "revision", None)
    )
    voice = to_device_nonblocking(voice.to(dtype=lm.bos_emb.dtype), lm.bos_emb.device)
    if voice.ndim == 3:
        voice = voice.squeeze(0)
    return lm.speaker_projection(voice.transpose(0, 1)) if project else voice


def prepare_text(text):
    text = " ".join(text.split())
    if not text:
        raise ValueError("Phonon input text cannot be empty")
    tail = 3 if len(text.split()) <= 4 else 1
    text = text[0].upper() + text[1:]
    if text[-1].isalnum():
        text += "."
    return text, tail


def build_prompt(lm, text_ids, voice, *, project_voice=True, conditioning=None):
    parameter = lm.bos_emb
    voice = to_device_nonblocking(voice.to(dtype=parameter.dtype), parameter.device)
    if voice.ndim == 3:
        voice = voice.squeeze(0)
    if project_voice:
        voice = lm.speaker_projection(voice.transpose(0, 1))
    ids = to_device_nonblocking(torch.as_tensor(text_ids, dtype=torch.long), parameter.device)
    text = lm.text_embedding(ids)
    if conditioning is None:
        conditioning = lm.conditioning()
    bos = lm.input_linear(lm.bos_emb) + conditioning
    return torch.cat((voice, text, bos[None]), 0)
