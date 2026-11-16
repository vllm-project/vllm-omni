# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from https://github.com/higurashi-ai/Haru (Apache-2.0), haru/haru/prompt.py @ d3612b0.
"""Conversation prompt: a character card, then a voice reference.

    position:  [0, P)          [P, S)                 [S, ...)
    text:      system card     PAD ... PAD EOP        live text stream
    model:     -               reference latents      model audio
    user:      -               -                      user audio

The card is a ChatML system turn and carries no audio. The reference is audio
only, on the model channel; its text is PAD on every frame and EOP on the last.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm_omni.model_executor.models.haru.configuration_haru import HaruSpec
from vllm_omni.model_executor.models.haru.vibevoice_codec import VibeVoiceCodec

if TYPE_CHECKING:
    from tokenizers import Tokenizer


def system_prompt_ids(tokenizer: Tokenizer, system_prompt: str) -> list[int]:
    if not system_prompt:
        return []
    text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    return tokenizer.encode(text, add_special_tokens=False).ids


def reference_ids(frames: int, pad_id: int, eop_id: int) -> list[int]:
    return [pad_id] * (frames - 1) + [eop_id] if frames > 0 else []


def reference_frames(num_samples: int, frame_samples: int) -> int:
    return math.ceil(num_samples / frame_samples)


@dataclass
class Prompt:
    """Inputs of the prompt positions: text tokens and both audio streams."""

    tokens: torch.Tensor  # [n]
    model_latents: torch.Tensor  # [n, D]
    user_latents: torch.Tensor  # [n, D]
    model_mask: torch.Tensor  # [n] positions that carry model audio
    user_mask: torch.Tensor  # [n] positions that carry user audio

    @property
    def length(self) -> int:
        return self.tokens.shape[0]


def prompt_length(
    tokenizer: Tokenizer, system_prompt: str, reference_samples: int, speak_first: bool, frame_samples: int
) -> int:
    """Number of positions :func:`build_prompt` produces."""
    n = len(system_prompt_ids(tokenizer, system_prompt)) + reference_frames(reference_samples, frame_samples)
    return n + 2 if speak_first else max(n, 1)


@torch.no_grad()
def build_prompt(
    spec: HaruSpec,
    codec: VibeVoiceCodec,
    tokenizer: Tokenizer,
    system_prompt: str = "",
    reference: torch.Tensor | None = None,
    speak_first: bool = False,
    generator: torch.Generator | None = None,
) -> Prompt:
    """System prompt, then the voice reference, then (``speak_first``) an opening turn."""
    device = codec.device
    pad, eop = spec.pad_token_id, spec.eop_token_id
    card = system_prompt_ids(tokenizer, system_prompt)
    ref_latents = None
    if reference is not None and len(reference):
        ref_latents = codec.sample(codec.encode(reference), generator)[0].float()
    n_ref = 0 if ref_latents is None else ref_latents.shape[0]
    # A reference already ends in EOP, so an opening turn starts with PAD, EOP.
    grid = [pad, eop] if speak_first else []
    if not card and not n_ref and not grid:
        grid = [pad]
    tokens = card + reference_ids(n_ref, pad, eop) + grid
    n, n_card, n_grid = len(tokens), len(card), len(grid)

    # Audio streams per position: none on the card, the model's only on the
    # reference. Positions before any generated audio carry the speaker's
    # room tone, and encoded silence on the user side.
    model_latents = torch.zeros(n, spec.latent_dim, device=device)
    user_latents = torch.zeros(n, spec.latent_dim, device=device)
    if n_ref:
        model_latents[n_card : n_card + n_ref] = ref_latents
    if n_grid:
        model_latents[n - n_grid :] = _room_tone(codec, reference, ref_latents)
        user_latents[n - n_grid :] = silence_latent(codec)
    positions = torch.arange(n, device=device)
    return Prompt(
        tokens=torch.tensor(tokens, device=device),
        model_latents=model_latents,
        user_latents=user_latents,
        model_mask=positions >= n_card,
        user_mask=positions >= n - n_grid,
    )


def silence_latent(codec: VibeVoiceCodec) -> torch.Tensor:
    """Latent of digital silence: the input of an audio stream with no sound yet."""
    cached = getattr(codec, "_silence_latent", None)
    if cached is None:
        silence = torch.zeros(16 * codec.config.frame_samples)
        cached = codec.encode(silence)[0, -1].float()
        codec._silence_latent = cached
    return cached


def _room_tone(codec: VibeVoiceCodec, reference: torch.Tensor | None, ref_latents: torch.Tensor | None) -> torch.Tensor:
    """The reference's quietest frame (the speaker's noise floor), else silence."""
    frame = codec.config.frame_samples
    whole = 0 if reference is None else len(reference) // frame
    if ref_latents is None or whole == 0:
        return silence_latent(codec)
    energy = reference[: whole * frame].reshape(whole, -1).square().mean(dim=1)
    return ref_latents[int(energy.argmin())]
