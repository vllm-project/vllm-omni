# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoModel

if TYPE_CHECKING:
    from .modeling_audex_causal_speech_decoder import AudexCausalSpeechDecoderModel


def freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_audex_causal_speech_decoder(
    model_path: str,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype | str | None = None,
) -> AudexCausalSpeechDecoderModel:
    kwargs = {"trust_remote_code": True}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    model = AutoModel.from_pretrained(model_path, **kwargs)
    return freeze_model(model.to(device))


def _decoder_device(decoder: AudexCausalSpeechDecoderModel) -> torch.device:
    return next(decoder.parameters()).device


def _embed_speech_token_frames(
    decoder: AudexCausalSpeechDecoderModel,
    token_frames: Sequence[Sequence[int]],
    device: torch.device,
) -> torch.Tensor:
    indices = torch.tensor(token_frames, dtype=torch.long, device=device).unsqueeze(0)
    return decoder.audex_speech_token_embedder.get_output_from_indices(indices)


def decode_speech_token_frames(
    token_frames: Iterable[Sequence[int]],
    decoder: AudexCausalSpeechDecoderModel,
    *,
    device: torch.device | str | None = None,
    chunk_frames: int = 1,
) -> Iterator[torch.Tensor]:
    if chunk_frames <= 0:
        raise ValueError(f"chunk_frames must be positive, got {chunk_frames}")

    device = torch.device(device) if device is not None else _decoder_device(decoder)
    cache = decoder.create_cache()
    buffer: list[list[int]] = []
    lookahead_steps = decoder.lookahead_steps

    for token_frame in token_frames:
        buffer.append(list(token_frame))
        while len(buffer) > lookahead_steps:
            emit_frames = min(chunk_frames, len(buffer) - lookahead_steps)
            wav = _decode_buffered_frames(decoder, cache, buffer, emit_frames, device, flush=False)
            del buffer[:emit_frames]
            yield wav

    while buffer:
        emit_frames = min(chunk_frames, len(buffer))
        wav = _decode_buffered_frames(decoder, cache, buffer, emit_frames, device, flush=True)
        del buffer[:emit_frames]
        yield wav


async def decode_speech_token_stream(
    token_stream: AsyncIterable[Sequence[int]],
    decoder: AudexCausalSpeechDecoderModel,
    *,
    device: torch.device | str | None = None,
    chunk_frames: int = 1,
) -> AsyncIterator[torch.Tensor]:
    if chunk_frames <= 0:
        raise ValueError(f"chunk_frames must be positive, got {chunk_frames}")

    device = torch.device(device) if device is not None else _decoder_device(decoder)
    cache = decoder.create_cache()
    buffer: list[list[int]] = []
    lookahead_steps = decoder.lookahead_steps

    async for token_frame in token_stream:
        buffer.append(list(token_frame))
        while len(buffer) > lookahead_steps:
            emit_frames = min(chunk_frames, len(buffer) - lookahead_steps)
            wav = _decode_buffered_frames(decoder, cache, buffer, emit_frames, device, flush=False)
            del buffer[:emit_frames]
            yield wav

    while buffer:
        emit_frames = min(chunk_frames, len(buffer))
        wav = _decode_buffered_frames(decoder, cache, buffer, emit_frames, device, flush=True)
        del buffer[:emit_frames]
        yield wav


def _decode_buffered_frames(
    decoder: AudexCausalSpeechDecoderModel,
    cache: Any,
    buffer: Sequence[Sequence[int]],
    emit_frames: int,
    device: torch.device,
    *,
    flush: bool,
) -> torch.Tensor:
    with torch.inference_mode():
        vq_emb = _embed_speech_token_frames(decoder, buffer[:emit_frames], device)
        lookahead_vq_emb = None
        if decoder.lookahead_steps > 0:
            future_frames = buffer[emit_frames : emit_frames + decoder.lookahead_steps]
            future_parts = []
            if future_frames:
                future_parts.append(_embed_speech_token_frames(decoder, future_frames, device))
            missing_frames = decoder.lookahead_steps - len(future_frames) if flush else 0
            if missing_frames > 0:
                future_parts.append(vq_emb.new_zeros(vq_emb.size(0), missing_frames, vq_emb.size(-1)))
            lookahead_vq_emb = torch.cat(future_parts, dim=1) if future_parts else None
        return decoder.decode_cached(vq_emb, cache, lookahead_vq_emb=lookahead_vq_emb)
