# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""P2-01: NeMo TN -> UTF-8 bytes -> canonical 10-column prompt frames."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm_omni.model_executor.models.zonos2.configuration_zonos2 import Zonos2Config
from vllm_omni.model_executor.models.zonos2.zonos2_keys import FRAMES, SPEAKER_EMBEDDING, SPEAKER_POSITION
from vllm_omni.model_executor.models.zonos2.zonos2_textnorm import Zonos2TextNormalizer

# Frozen official 0.2s silence: 17 real frames, nine DAC codebooks.
# Data from Zyphra/ZONOS2, frozen source 194c0a3a (MIT).
SILENCE_CODES = (
    (568, 778, 338, 524, 967, 360, 728, 550, 90),
    (568, 778, 10, 674, 364, 981, 741, 378, 731),
    *((568, 804, 10, 674, 364, 981, 568, 378, 731),) * 14,
    (568, 778, 721, 842, 264, 974, 989, 507, 308),
)
BYTE_OFFSET = 192
BOS_ID = 2
EOS_ID = 3


@dataclass(frozen=True)
class Zonos2Prompt:
    frames: torch.Tensor
    normalized_text: str
    speaker_embedding: torch.Tensor | None = None

    def to_engine_prompt(self) -> dict[str, Any]:
        info: dict[str, Any] = {FRAMES: self.frames}
        if self.speaker_embedding is not None:
            info[SPEAKER_EMBEDDING] = self.speaker_embedding
            info[SPEAKER_POSITION] = 0
        # One lifecycle id per frame; the full ten columns travel separately.
        return {"prompt_token_ids": self.frames[:, 9].tolist(), "additional_information": info}


class Zonos2Processor:
    def __init__(self, config: Zonos2Config, normalizer: Zonos2TextNormalizer | None = None):
        self.config = config
        self.normalizer = normalizer if normalizer is not None else Zonos2TextNormalizer()
        if (config.n_codebooks, config.codebook_size, config.audio_pad_id, config.text_vocab) != (9, 1024, 1025, 519):
            raise ValueError("ZONOS2 frontend requires the frozen 9-codebook / text_vocab=519 layout")
        self.features = tuple(config.quality_features or tuple(config.quality_buckets or {}))
        self.quality_counts = tuple(len((config.quality_buckets or {}).get(k, ())) for k in self.features)
        background = 2 if config.speaker_background_token_enabled else 0
        accurate = int(config.accurate_mode_token_enabled and background > 0)
        if 448 + config.speaking_rate_num_buckets + config.quality_num_buckets + background + accurate != 519:
            raise ValueError("ZONOS2 conditioning vocabulary does not match text_vocab=519")
        if sum(self.quality_counts) != config.quality_num_buckets:
            raise ValueError("ZONOS2 frontend needs the checkpoint quality feature/bucket schema")
        self.background_count = background
        self.accurate_count = accurate

    def silence_frames(self) -> torch.Tensor:
        audio = torch.full((17, 9), 1025, dtype=torch.int32)
        silence = torch.tensor(SILENCE_CODES, dtype=torch.int32)
        for col in range(9):
            audio[col:, col] = silence[: 17 - col, col]
        return torch.cat((audio, torch.full((17, 1), 519, dtype=torch.int32)), dim=1)

    def build(
        self,
        text: str,
        *,
        language: str = "en_us",
        text_normalization: bool = True,
        speaking_rate_bucket: int | None = None,
        quality_buckets: Mapping[str, int | None] | Sequence[int | None] | None = None,
        speaker_embedding: torch.Tensor | None = None,
        clean_speaker_background: bool = False,
        accurate_mode: bool = True,
    ) -> Zonos2Prompt:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text cannot be empty")
        normalized = self.normalizer.normalize(text, language) if text_normalization else text
        tokens: list[int] = []
        if speaking_rate_bucket is not None:
            if not 0 <= speaking_rate_bucket < self.config.speaking_rate_num_buckets:
                raise ValueError("speaking_rate_bucket is out of range")
            tokens.append(448 + speaking_rate_bucket)
        quality = {"trailing_silence_s": 3} if quality_buckets is None else quality_buckets
        if isinstance(quality, Mapping):
            unknown = set(quality) - set(self.features)
            if unknown:
                raise ValueError(f"Unknown quality features: {sorted(unknown)}")
            values = [quality.get(name) for name in self.features]
        else:
            values = list(quality)
            if len(values) > len(self.features):
                raise ValueError("Too many quality buckets")
            values += [None] * (len(self.features) - len(values))
        base = 448 + self.config.speaking_rate_num_buckets
        for count, bucket in zip(self.quality_counts, values, strict=True):
            if bucket is not None:
                if not 0 <= bucket < count:
                    raise ValueError("quality bucket is out of range")
                tokens.append(base + bucket)
            base += count
        speaker = None
        prefix: list[int] = []
        if speaker_embedding is not None:
            if not torch.is_floating_point(speaker_embedding):
                raise TypeError("speaker_embedding must be floating point")
            if speaker_embedding.shape not in ((2048,), (1, 2048)):
                raise ValueError("speaker_embedding must have shape [2048] or [1,2048]")
            speaker = speaker_embedding.detach().reshape(2048).to(device="cpu", dtype=torch.float32).clone()
            if not torch.isfinite(speaker).all():
                raise ValueError("speaker_embedding must be finite")
            prefix.append(519)
            if self.background_count:
                prefix.append(base + (0 if clean_speaker_background else 1))
                if self.accurate_count and accurate_mode:
                    prefix.append(base + self.background_count)
        tokens = prefix + tokens + [BOS_ID, *(byte + BYTE_OFFSET for byte in normalized.encode("utf-8")), EOS_ID]
        frames = torch.full((len(tokens), 10), 1025, dtype=torch.int32)
        frames[:, 9] = torch.tensor(tokens, dtype=torch.int32)
        frames = torch.cat((frames, self.silence_frames()), dim=0)
        if len(frames) > self.config.max_seqlen:
            raise ValueError("ZONOS2 prompt exceeds max_seqlen")
        return Zonos2Prompt(frames, normalized, speaker)

    def build_prompt(self, text: str, **kwargs: Any) -> dict[str, Any]:
        return self.build(text, **kwargs).to_engine_prompt()
