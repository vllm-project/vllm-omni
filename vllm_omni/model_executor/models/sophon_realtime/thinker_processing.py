# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio + text preprocessing for the DuplexJev thinker.

Same features as Ultravox (Whisper-style 128-bin log-mel, 30 s chunks), with two
differences the DuplexJev connectors were trained with:

* every clip is right-padded with silence to a whole number of audio tokens, so a
  clip yields the same tokens whatever it is batched with;
* the number of audio tokens per chunk is either ``ceil(mel / (8 * stack))``
  (Ultravox's rule) or, for checkpoints with ``"audio_full_length": true``, the
  encoder's true output length divided by the stack factor.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import transformers

HOP = 160  # mel hop at 16 kHz


def qwen3_asr_output_lengths(mel_lengths: torch.Tensor, n_window: int = 50) -> torch.Tensor:
    """Encoder frames produced by the Qwen3-ASR front end for ``mel_lengths`` mel
    frames (chunks of ``2 * n_window`` frames, three stride-2 convolutions each)."""
    chunk = n_window * 2
    rest = mel_lengths % chunk
    rest = (rest - 1) // 2 + 1
    return ((rest - 1) // 2 + 1 - 1) // 2 + 1 + (mel_lengths // chunk) * 13


class DuplexJevProcessor:
    """HF-style processor: ``(text with <|audio|> placeholders, audios)`` to
    ``input_ids`` plus the Ultravox audio fields (``audio_values``,
    ``audio_lens``, ``audio_token_len``, ``audio_token_start_idx``,
    ``audio_num_chunks``)."""

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        *,
        stack_factor: int = 2,
        encoder_ds_factor: int = 8,
        audio_context_size: int = 3000,
        audio_placeholder: str = "<|audio|>",
        full_length: bool = False,
    ):
        self.audio_processor = feature_extractor  # vLLM reads the WhisperFeatureExtractor here
        self.tokenizer = tokenizer
        self.stack_factor = stack_factor
        self.encoder_ds_factor = encoder_ds_factor
        self.full_length = full_length
        self.audio_context_size = audio_context_size
        self.audio_placeholder = audio_placeholder
        self.vocab = tokenizer.get_vocab()
        # vLLM overrides both with a reserved token (see UltravoxProcessingInfo).
        self.audio_token_replacement = tokenizer.eos_token
        self.audio_replacement_token_id = None

    def _align(self, audio: np.ndarray) -> np.ndarray:
        step = HOP * self.encoder_ds_factor * self.stack_factor
        n = max(len(audio), 2 * HOP)
        n = -(-n // step) * step
        return np.pad(audio, (0, n - len(audio))) if n > len(audio) else audio

    def _chunk(self, values: torch.Tensor, lens: torch.Tensor, include_audio_num_chunks: bool) -> dict:
        ctx = self.audio_context_size or values.shape[-1]
        chunks: list[torch.Tensor] = []
        chunk_lens: list[int] = []
        is_continuation: list[bool] = []
        num_chunks: list[int] = []
        for i in range(values.shape[0]):
            length = int(lens[i])
            num_chunks.append(int(np.ceil(length / ctx)))
            for off in range(0, length, ctx):
                chunk = values[i, :, off : off + ctx]
                if off > 0 and chunk.shape[-1] < ctx:
                    chunk = F.pad(chunk, (0, ctx - chunk.shape[-1]))
                chunks.append(chunk)
                chunk_lens.append(min(length - off, ctx))
                is_continuation.append(off > 0)
        out = {
            "audio_values": torch.stack(chunks, 0),
            "audio_lens": torch.tensor(chunk_lens, dtype=torch.int64),
            "audio_is_continuation": torch.tensor(is_continuation, dtype=torch.bool),
            "audio_batch_size": torch.tensor([len(chunks)]),
        }
        if include_audio_num_chunks:
            out["audio_num_chunks"] = torch.tensor(num_chunks, dtype=torch.int64)
        return out

    def __call__(
        self,
        text: str | None = None,
        audio=None,
        audios=None,
        sampling_rate: int | None = None,
        return_tensors="pt",
        include_audio_num_chunks: bool = False,
        **kwargs,
    ) -> transformers.BatchFeature:
        if audio is not None:
            audios = audio if isinstance(audio, list) else [audio]
        audios = [a.numpy() if isinstance(a, torch.Tensor) else np.asarray(a) for a in (audios or [])]
        data: dict = {}
        is_continuation: torch.Tensor | list = []
        if audios:
            audios = [self._align(a.astype(np.float32)) for a in audios]
            feats = self.audio_processor(
                audios,
                sampling_rate=sampling_rate or self.audio_processor.sampling_rate,
                padding="longest",
                pad_to_multiple_of=HOP,
                truncation=False,
                return_attention_mask=True,
                return_tensors="pt",
            )
            data.update(
                self._chunk(
                    torch.as_tensor(feats.input_features),
                    torch.as_tensor(feats.attention_mask).sum(-1),
                    include_audio_num_chunks,
                )
            )
            is_continuation = data.pop("audio_is_continuation")
            if self.full_length and self.encoder_ds_factor == 8:
                frames = qwen3_asr_output_lengths(data["audio_lens"]).float()
                data["audio_token_len"] = torch.ceil(frames / self.stack_factor).to(torch.int)
            else:
                data["audio_token_len"] = torch.ceil(
                    data["audio_lens"] / (self.encoder_ds_factor * self.stack_factor)
                ).to(torch.int)
        if text is not None:
            parts = self.tokenizer(text.split(self.audio_placeholder), add_special_tokens=False)["input_ids"]
            rep = self.vocab[self.audio_token_replacement]
            ids: list[int] = []
            starts: list[int] = []
            p = -1
            for i, n in enumerate(data.get("audio_token_len", [])):
                if not is_continuation[i]:
                    p += 1
                    if p >= len(parts):
                        raise ValueError("Text contains too few audio placeholders.")
                    ids.extend(parts[p])
                starts.append(len(ids))
                ids.extend([rep] * int(n))
            p += 1
            if p != len(parts) - 1:
                raise ValueError("Text contains too many audio placeholders.")
            ids.extend(parts[p])
            if "audio_token_len" in data:
                data["audio_token_start_idx"] = torch.as_tensor(starts)
            data["input_ids"] = [ids]
            data["attention_mask"] = [[1] * len(ids)]
        return transformers.BatchFeature(data=data, tensor_type=return_tensors)
