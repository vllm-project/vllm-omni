# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Experimental speech-duration estimate; this provides no word-level alignment."""

from __future__ import annotations

import math

from vllm_omni.engine.duplex.session.history_calibration import HeardTextSnapshot
from vllm_omni.model_executor.models.qwen3_omni.duplex.history import text_units


class QwenTokenRateCalibration:
    """Use a reply's complete duration, falling back to configured milliseconds per token."""

    def __init__(self, tokenizer, ms_per_token: float) -> None:
        if not tokenizer.is_fast:
            raise ValueError("Token-rate history calibration requires a fast tokenizer with offsets")
        if not math.isfinite(ms_per_token) or ms_per_token <= 0:
            raise ValueError("ms_per_token must be finite and positive")
        self.tokenizer = tokenizer
        self.ms_per_token = ms_per_token

    async def __call__(self, snapshot: HeardTextSnapshot) -> int | None:
        # Bound CPU work on the session event loop, including tokenization.
        if len(snapshot.text) > 8192:
            return None
        units = text_units(snapshot.text)
        if len(units) > 2048:
            return None
        offsets = self.tokenizer(snapshot.text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        ms_per_token = self.ms_per_token
        duration = snapshot.complete_audio_duration_ms
        if offsets and duration is not None and duration > 0:
            ms_per_token = duration / len(offsets)
        count = min(len(offsets), math.floor(max(0, snapshot.played_ms - 160) / ms_per_token))
        if not count:
            return 0
        end = offsets[count - 1][1]
        if count < len(offsets):
            # Byte-level tokens can overlap one Unicode character. It is heard
            # only after all of that character's tokens have been accounted for.
            end = min(end, offsets[count][0])
        # Retain only whole English words / Chinese characters in original text.
        # This controller runs only for partial playback: withhold the final unit
        # even if the rate predicts that the entire reply has been spoken.
        return max((unit.end for unit in units[:-1] if unit.end <= end), default=0)
