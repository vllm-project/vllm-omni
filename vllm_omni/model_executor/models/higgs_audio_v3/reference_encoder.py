# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Higgs-Audio v3 voice-clone reference encoder, as the API server runs it.

A clip becomes the talker's ``audio_input_ids``: the higgs-audio v2 codec's
codes with the delay pattern applied. The encoder runs in encoder processes
shared by the server's API processes (``ReferenceEncoderSpec``), so one copy of
the codec serves all of them; each clip is encoded on its own, exactly as the
in-process path does.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch

from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_tokenizer import (
    encode_ref_audio_codes_with,
    load_audio_tokenizer,
)
from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_tokenizer import apply_delay_pattern
from vllm_omni.utils.shared_reference_encoder import WireFormat


def delayed_reference_codes(model, wav: np.ndarray, sr: int) -> torch.Tensor:
    """``[T + N - 1, N]`` delayed codes of one clip; ``model`` is the loaded codec."""
    codes_tn = encode_ref_audio_codes_with(model, wav, sr).transpose(0, 1).contiguous()
    return apply_delay_pattern(codes_tn).to(torch.long)


class HiggsAudioV3ReferenceEncoder:
    """Loads the codec once (on this process's device) and encodes clips one at a time."""

    def __init__(self) -> None:
        self.model = load_audio_tokenizer()

    def encode_batch(self, clips: list[tuple[np.ndarray, int]]) -> Iterator[torch.Tensor | Exception]:
        for wav, sr in clips:
            try:
                yield delayed_reference_codes(self.model, wav, sr)
            except Exception as exc:  # noqa: BLE001 - reported for this clip only
                yield exc


# Codes are < 1026 (codebook entries plus BOC/EOC): int16 would do, int32 keeps it simple.
HIGGS_AUDIO_V3_REFERENCE_WIRE = WireFormat(
    clip_to_array=lambda clip: (clip[0], int(clip[1])),
    clip_from_array=lambda array, sr: (array.copy(), int(sr)),
    artifact_to_arrays=lambda codes: [codes.to(torch.int32).numpy()],
    artifact_from_arrays=lambda arrays: torch.from_numpy(arrays[0].astype(np.int64)),
)
