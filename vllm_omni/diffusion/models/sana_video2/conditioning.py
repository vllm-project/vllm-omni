# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Gemma and image conditioning matching the pinned Sana-Video 2.0 release."""

import numpy as np
import torch
from PIL import Image

DEFAULT_NEGATIVE_PROMPT = (
    "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, "
    "jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, "
    "jitter, and ghosting effects, creating a disorienting visual experience."
)


PROMPT_INSTRUCTION = (
    (
        'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions '
        "suitable for image generation. Evaluate the level of detail in the user prompt:"
    ),
    (
        "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and "
        "spatial relationships to create vivid and concrete scenes."
    ),
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    (
        "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape,"
        " sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers."
    ),
    (
        "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring "
        "glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus "
        "passing by towering glass skyscrapers."
    ),
    (
        "Please generate only the enhanced description for the prompt below and avoid including any "
        "additional commentary or evaluations:"
    ),
    "User Prompt: ",
)


def prepare_image(image: Image.Image, height: int, width: int) -> torch.Tensor:
    """Bicubic resize-to-fill, center crop, then [-1,1], matching upstream."""
    image = image.convert("RGB")
    w, h = image.size
    rh, rw = height / h, width / w
    if rh > rw:
        sh, sw = height, round(w * rh)
        top, left = 0, int(round((sw - width) / 2.0))
    else:
        sh, sw = round(h * rw), width
        top, left = int(round((sh - height) / 2.0)), 0
    resized = image.resize((sw, sh), Image.Resampling.BICUBIC)
    pixels = np.array(resized)[top : top + height, left : left + width].copy()
    return torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255).sub_(0.5).div_(0.5)


@torch.no_grad()
def encode_text(tokenizer, text_encoder, prompts, *, instruction, device, max_length=300):
    """Keep BOS plus the last 299 tokens after instruction-aware tokenization."""
    tokenizer.padding_side = "right"
    prefix = "\n".join(instruction)
    length = len(tokenizer.encode(prefix)) + max_length - 2 if prefix else max_length
    tokens = tokenizer(
        [prefix + prompt for prompt in prompts],
        max_length=length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    indices = [0] + list(range(-max_length + 1, 0))
    embeddings = text_encoder(tokens.input_ids, tokens.attention_mask, use_cache=False)[0]
    return embeddings[:, indices], tokens.attention_mask[:, indices].bool()


def normalize_latents(vae, latents):
    mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(latents)
    std = vae.latents_std.view(1, -1, 1, 1, 1).to(latents)
    return (latents - mean) * vae.config.scaling_factor / std


def denormalize_latents(vae, latents):
    mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(latents)
    std = vae.latents_std.view(1, -1, 1, 1, 1).to(latents)
    return latents * std / vae.config.scaling_factor + mean
