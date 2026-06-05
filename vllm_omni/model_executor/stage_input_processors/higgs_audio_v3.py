# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-input processor for higgs-audio v3: Talker -> Code2Wav.

Sync mode only in this phase (no async_chunk streaming).

Key difference from v2: BOC/EOC filtering happens AFTER delay pattern reversal
to avoid corrupting valid tail content.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

__all__ = ["talker2code2wav"]

logger = init_logger(__name__)

_NUM_CODEBOOKS = 8
_NUM_REAL_CODES = 1024  # codes in [0, 1023] are real


def _revert_delay_pattern(audio_codes_qt: torch.Tensor) -> torch.Tensor:
    """Reverse the MusicGen-style delay pattern.

    Input shape: [num_codebooks, seq_len + num_codebooks - 1].
    Output shape: [num_codebooks, seq_len].

    For each codebook i, extract delayed[i, i : i + seq_len] to remove
    the i leading BOC pads and Q-1-i trailing EOC entries.
    """
    if audio_codes_qt.ndim != 2:
        raise ValueError(f"_revert_delay_pattern expects [Q, T] input; got {tuple(audio_codes_qt.shape)}")
    q, t = audio_codes_qt.shape
    if t < q:
        raise ValueError(f"Not enough frames to revert delay pattern: T={t} < Q={q}")
    seq_len = t - q + 1
    out_l = []
    for i in range(q):
        out_l.append(audio_codes_qt[i : i + 1, i : seq_len + i])
    return torch.cat(out_l, dim=0)


def _filter_real_code_frames(audio_codes_qt: torch.Tensor) -> torch.Tensor:
    """Keep only frames where ALL codebook values are in [0, 1023].

    Input shape: [num_codebooks, num_frames].
    Called AFTER delay pattern reversal.
    """
    if audio_codes_qt.numel() == 0:
        return audio_codes_qt
    # Transpose to [num_frames, num_codebooks] for per-frame filtering
    frames = audio_codes_qt.t()
    valid = (frames >= 0).all(dim=1) & (frames < _NUM_REAL_CODES).all(dim=1)
    return frames[valid].t().contiguous()


def talker2code2wav(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[Any]:
    """Sync: collect all talker codes, revert delay pattern, filter, pass to code2wav."""
    from vllm_omni.inputs.data import OmniTokensPrompt

    code2wav_inputs: list[OmniTokensPrompt] = []
    for talker_output in source_outputs:
        if not talker_output.finished:
            continue
        output = talker_output.outputs[0]
        mm = output.multimodal_output
        mm_codes = mm.get("codes", {})

        audio_codes = mm_codes.get("audio")
        if audio_codes is None or not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
            code2wav_inputs.append(
                OmniTokensPrompt(
                    prompt_token_ids=[],
                    multi_modal_data=None,
                    mm_processor_kwargs=None,
                    additional_information=None,
                )
            )
            continue

        audio_codes = audio_codes.to(torch.long)
        if audio_codes.ndim == 1:
            if audio_codes.numel() % _NUM_CODEBOOKS != 0:
                raise ValueError(
                    f"flat audio_codes length {audio_codes.numel()} not divisible by num_codebooks={_NUM_CODEBOOKS}"
                )
            audio_codes = audio_codes.reshape(-1, _NUM_CODEBOOKS)

        if audio_codes.ndim != 2:
            raise ValueError(f"audio_codes must be 1D or 2D; got shape {tuple(audio_codes.shape)}")

        # Transpose to [Q, T] for delay pattern reversal
        codes_qt = audio_codes.transpose(0, 1).contiguous().cpu()

        # Step 1: Revert delay pattern FIRST
        codes_qt = _revert_delay_pattern(codes_qt)

        # Step 2: Filter BOC/EOC AFTER de-delay
        codes_qt = _filter_real_code_frames(codes_qt)

        if codes_qt.numel() == 0:
            code2wav_inputs.append(
                OmniTokensPrompt(
                    prompt_token_ids=[],
                    multi_modal_data=None,
                    mm_processor_kwargs=None,
                    additional_information=None,
                )
            )
            continue

        # Code2Wav expects codebook-major flat: [Q * num_frames]
        codec_codes = codes_qt.reshape(-1).tolist()

        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=None,
            )
        )
    return code2wav_inputs
