# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Prompt construction for MOSS-TTS-Realtime."""

from typing import Any

import numpy as np
import torch

REALTIME_AUDIO_CHANNELS = 16
REALTIME_PREFILL_TEXT_TOKENS = 12


def build_realtime_prompt(
    tokenizer: Any,
    processor: Any,
    text: str,
    reference_codes: torch.Tensor,
) -> dict[str, Any]:
    """Build the mixed text/audio prompt consumed by the Realtime talker."""
    audio_tokens = reference_codes.detach().cpu().numpy()
    system_grid = np.asarray(processor.make_ensemble(prompt_audio_tokens=audio_tokens), dtype=np.int64)
    expected_width = REALTIME_AUDIO_CHANNELS + 1
    if system_grid.ndim != 2 or system_grid.shape[1] != expected_width:
        raise ValueError(
            f"MOSS-TTS-Realtime processor returned shape {system_grid.shape}; "
            f"expected (sequence_length, {expected_width})"
        )

    assistant_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    assistant_grid = np.full(
        (len(assistant_ids), expected_width),
        processor.audio_channel_pad,
        dtype=np.int64,
    )
    assistant_grid[:, 0] = assistant_ids

    text_ids = tokenizer.encode(text, add_special_tokens=False)
    if not text_ids:
        raise ValueError("MOSS-TTS-Realtime input is empty after tokenization")

    prefill_len = min(len(text_ids), REALTIME_PREFILL_TEXT_TOKENS)
    text_grid = np.full(
        (prefill_len, expected_width),
        processor.audio_channel_pad,
        dtype=np.int64,
    )
    text_grid[:, 0] = text_ids[:prefill_len]
    text_grid[-1, 1] = processor.audio_bos_token

    prompt_grid = np.concatenate((system_grid, assistant_grid, text_grid), axis=0)
    params: dict[str, Any] = {
        "prompt_token_ids": prompt_grid[:, 0].tolist(),
        "codes": {"ref": torch.from_numpy(prompt_grid[:, 1:].copy())},
    }
    remaining_text_ids = text_ids[prefill_len:]
    if remaining_text_ids:
        params["ids"] = {"all": remaining_text_ids}
    return params
