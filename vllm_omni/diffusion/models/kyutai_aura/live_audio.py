# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from .audio import FEATURE_RATE, video_frames_for
from .kv_cache import RollingChunkKVCache

if TYPE_CHECKING:
    from .pipeline_kyutai_aura import KyutaiAuraPipeline


class AuraLiveAudio:
    """Worker-owned state for one portrait and a sequence of 24 kHz PCM16 packets."""

    def __init__(self, pipeline: KyutaiAuraPipeline, image, prompt: str, seed: int) -> None:
        self.pipeline = pipeline
        self.seed = seed
        self.mimi = pipeline.audio_features.start_stream()
        self.sample_rate = pipeline.audio_features.sample_rate
        self.reference = pipeline.encode_reference(pipeline.load_portrait(image))
        self.context = pipeline.encode_prompt(prompt)
        cfg = pipeline.cfg
        h, w = self.reference.shape[-2:]
        self.shape = (cfg.latent_channels, cfg.frames_per_chunk, h, w)
        tokens = cfg.frames_per_chunk * (h // cfg.patch_size[1]) * (w // cfg.patch_size[2])
        self.history = RollingChunkKVCache.for_config(cfg, tokens, pipeline.device)
        self.vae_cache = [None] * pipeline.vae._cached_conv_counts["decoder"]
        self.features = self.reference.new_empty(1, 0, cfg.audio_dim)
        self.feature_offset = 0
        self.pcm = np.empty(0, dtype=np.float32)
        self.samples_received = 0
        self.frames_sent = 0

    def _positions(self, chunk: int) -> np.ndarray:
        cfg = self.pipeline.cfg
        count = cfg.video_frames_per_chunk
        positions = np.arange(chunk * count, (chunk + 1) * count)
        return np.rint(positions * FEATURE_RATE / cfg.fps).astype(np.int64)

    def push(self, pcm: bytes, *, final: bool) -> list[tuple]:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        self.samples_received += samples.size
        self.pcm = np.concatenate((self.pcm, samples))
        new = self.mimi.push(samples, final=final)
        if new is not None:
            self.features = torch.cat((self.features, new), dim=1)

        pipeline = self.pipeline
        cfg = pipeline.cfg
        target = video_frames_for(self.samples_received, self.sample_rate, cfg.fps) if self.samples_received else 0
        outputs = []
        while self.frames_sent < target:
            chunk = self.history.chunk
            positions = self._positions(chunk)
            # 25 Hz grid points are spaced by half a native Mimi frame.
            left = positions // 2 - self.feature_offset
            right = left + positions % 2
            available = self.features.shape[1]
            if not final and (right[-1] >= available or self.samples_received < (chunk + 1) * self.sample_rate):
                break
            valid = (left >= 0) & (left < available)
            condition = self.features.new_zeros(1, positions.size, cfg.audio_dim)
            if valid.any():
                a = self.features[:, left[valid]]
                b = self.features[:, np.minimum(right[valid], available - 1)]
                weight = condition.new_tensor(positions[valid] % 2).view(1, -1, 1) * 0.5
                condition[:, valid] = torch.lerp(a, b, weight)
            speech_input = condition.transpose(1, 2).unsqueeze(0).to(torch.bfloat16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                speech = pipeline.transformer.encode_audio(speech_input)
                latent = pipeline._sample_chunk(
                    pipeline.chunk_generator(self.seed, chunk),
                    self.shape,
                    self.context,
                    self.reference,
                    speech,
                    self.history,
                )
                self.history.advance()
                video = pipeline.decode_chunk(latent, self.vae_cache, first=chunk == 0)
            video = video[:, : target - self.frames_sent]
            start = round(self.frames_sent * self.sample_rate / cfg.fps)
            self.frames_sent += video.shape[1]
            stop = round(self.frames_sent * self.sample_rate / cfg.fps)
            audio = self.pcm[: stop - start].copy()
            self.pcm = self.pcm[stop - start :].copy()
            outputs.append((video[None].cpu(), audio, self.sample_rate, cfg.fps))
            next_left = self._positions(self.history.chunk)[0] // 2
            discard = min(max(0, next_left - self.feature_offset), available)
            self.features = self.features[:, discard:].clone()
            self.feature_offset += discard
        return outputs
