# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request helpers for AuK: prompt rendering, sampling params, duration policy.

AuK's contract is one instruction plus an optional reference clip in, audio
out; the task (zero-shot TTS, instruct TTS, content or acoustic editing,
enhancement, separation) is carried by the instruction text alone, so there is
no per-task request surface.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from vllm import SamplingParams

from vllm_omni.inputs.data import OmniDiffusionSamplingParams

if TYPE_CHECKING:
    import numpy as np
    import torch

# The upstream ChatML render of a single user turn through
# ``Qwen2_5OmniProcessor.apply_chat_template(..., add_generation_prompt=True)``
# with the template's default system turn.
_CHATML = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "{instruction}{audio_part}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
_AUDIO_PART = "<|audio_bos|><|AUDIO|><|audio_eos|>"

# Literal marker the reference appends to the instruction when no reference
# clip is supplied. The model was trained with it, so it is not decorative.
NO_PROMPT_AUDIO = "|<no_prompt_audio>|"

# Latent frame rate: 24 kHz waveform, VAE downsample 480 -> 50 frames/s.
SAMPLE_RATE = 24000
HOP = 480

DEFAULT_NFE = 32
DEFAULT_CFG = 2.0
DEFAULT_SWAY = -1.0


def auk_prompt(
    instruction: str,
    audio: tuple[np.ndarray | torch.Tensor, int] | None = None,
    *,
    gen_seconds: float | None = None,
    sway: float = DEFAULT_SWAY,
    t_grid: list[float] | None = None,
    vae_sample: bool = False,
) -> dict[str, Any]:
    """Build the stage-0 prompt for one AuK request.

    Args:
        instruction: The task instruction; for text-only requests the
            no-reference marker is appended if it is not already there.
        audio: Reference or source clip as ``(waveform, sample_rate)``.
            ``None`` selects the text-only (instruct TTS) path.
        gen_seconds: Target duration. ``None`` reuses the source clip's
            length and is an error for text-only requests.
        sway: Sway coefficient for the Euler time grid.
        t_grid: Explicit time grid, overriding ``sway``.
        vae_sample: Draw the reference latent from the VAE posterior instead
            of taking its mean.

    Returns:
        A prompt dict for ``Omni.generate``.
    """
    has_audio = audio is not None and not (isinstance(audio, (tuple, list)) and (len(audio) == 0 or audio[0] is None))
    if not has_audio and not instruction.endswith(NO_PROMPT_AUDIO):
        instruction = instruction + NO_PROMPT_AUDIO

    prompt: dict[str, Any] = {
        "prompt": _CHATML.format(
            instruction=instruction,
            audio_part=_AUDIO_PART if has_audio else "",
        ),
        "additional_information": {
            "auk": {
                "gen_seconds": gen_seconds,
                "sway": sway,
                "t_grid": list(t_grid) if t_grid is not None else None,
                "vae_sample": vae_sample,
                "has_audio": has_audio,
            }
        },
    }
    if has_audio:
        prompt["multi_modal_data"] = {"audio": audio}
    return prompt


def auk_sampling_params(
    *,
    nfe: int = DEFAULT_NFE,
    cfg: float = DEFAULT_CFG,
    seed: int | None = 0,
) -> list[SamplingParams | OmniDiffusionSamplingParams]:
    """Build the per-stage sampling params for one AuK request.

    Stage 0 is a prefill-only encoder, so its vLLM sampling is a no-op that
    only has to satisfy the scheduler. Stage 1 carries the real knobs: ``nfe``
    Euler steps and CFG strength. AuK-Flash ignores both and pins its
    distilled recipe.
    """
    return [
        SamplingParams(max_tokens=1, temperature=0.0, detokenize=False, seed=seed),
        OmniDiffusionSamplingParams(num_inference_steps=nfe, guidance_scale=cfg, seed=seed),
    ]


def resolve_gen_frames(
    gen_seconds: float | None,
    ref_frames: int,
    *,
    sample_rate: int = SAMPLE_RATE,
    hop: int = HOP,
) -> int:
    """Resolve the target latent length in frames.

    An explicit duration wins; otherwise the target is as long as the source
    clip. A text-only request with no duration has nothing to generate, which
    is an error rather than a silent one-frame output.
    """
    if gen_seconds is not None:
        if gen_seconds <= 0:
            raise ValueError(f"gen_seconds must be positive; got {gen_seconds}")
        return max(1, math.ceil(gen_seconds * sample_rate / hop))
    if ref_frames > 0:
        return ref_frames
    raise ValueError("gen_seconds is required when the request carries no source audio")


__all__ = [
    "NO_PROMPT_AUDIO",
    "auk_prompt",
    "auk_sampling_params",
    "resolve_gen_frames",
]
