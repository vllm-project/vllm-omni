# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

import numpy as np
from vllm.multimodal.media import MediaConnector


async def resolve_ref_audio(ref_audio: str, *, model_config: Any) -> tuple[np.ndarray, int]:
    connector = MediaConnector(
        allowed_local_media_path=getattr(model_config, "allowed_local_media_path", None),
        allowed_media_domains=getattr(model_config, "allowed_media_domains", None),
    )
    wav_np, sr = await connector.fetch_audio_async(ref_audio)
    return np.asarray(wav_np, dtype=np.float32), int(sr)


def normalize_ref_audio(wav_np: np.ndarray, sample_rate: int, *, target_sr: int) -> np.ndarray:
    wav_np = np.asarray(wav_np, dtype=np.float32)
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=-1)
    wav_np = wav_np.reshape(-1)
    if sample_rate <= 0 or sample_rate == target_sr or wav_np.size == 0:
        return wav_np.astype(np.float32, copy=False)
    import torch
    import torchaudio

    audio = torch.from_numpy(wav_np).to(dtype=torch.float32).unsqueeze(0)
    resampled = torchaudio.functional.resample(audio, int(sample_rate), int(target_sr))
    return resampled.squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def load_native_tokenizer(model_config: Any) -> Any | None:
    model_path = getattr(model_config, "model", None)
    if not isinstance(model_path, str) or not model_path:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception:
        return None


def convert_token_to_id(tokenizer: Any, token: str) -> int | None:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    value = None
    if callable(convert):
        value = convert(token)
        if isinstance(value, list):
            value = value[0] if len(value) == 1 else None
    try:
        token_id = int(value)
    except (TypeError, ValueError):
        token_id = -1
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if token_id >= 0 and token_id != unk_token_id:
        return token_id
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            ids = list(encode(token, add_special_tokens=False))
        except TypeError:
            ids = list(encode(token))
        if len(ids) == 1:
            try:
                token_id = int(ids[0])
            except (TypeError, ValueError):
                token_id = -1
            if token_id >= 0 and token_id != unk_token_id:
                return token_id
    return None
