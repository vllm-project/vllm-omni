"""Shared helpers for Voxtream2 prompt and audio handling."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import soundfile as sf
import torch

VOXTREAM2_DEFAULT_SAMPLE_RATE = 24_000
VOXTREAM2_ALLOWED_LOCAL_MEDIA_PATH_ENV = "VOXTREAM2_ALLOWED_LOCAL_MEDIA_PATH"


def estimate_voxtream2_prompt_len(text: str) -> int:
    return max(64, len(text) // 2 + 64)


def build_voxtream2_prompt(text: str, ref_audio_path: str) -> dict[str, Any]:
    return {
        "prompt_token_ids": [1] * estimate_voxtream2_prompt_len(text),
        "additional_information": {
            "text": [text],
            "ref_audio_path": ref_audio_path,
        },
    }


def flatten_voxtream2_audio_output(
    multimodal_output: Mapping[str, Any],
    *,
    default_sample_rate: int = VOXTREAM2_DEFAULT_SAMPLE_RATE,
    require_audio: bool = True,
) -> tuple[torch.Tensor, int]:
    audio = multimodal_output.get("audio")
    if audio is None:
        audio = multimodal_output.get("model_outputs")
    if audio is None:
        if require_audio:
            raise ValueError(f"No audio found in multimodal_output keys={list(multimodal_output.keys())}")
        return torch.zeros((0,), dtype=torch.float32), default_sample_rate

    if isinstance(audio, list):
        tensors = [torch.as_tensor(item).reshape(-1).float().cpu() for item in audio if item is not None]
        audio_tensor = torch.cat(tensors, dim=-1) if tensors else torch.zeros((0,), dtype=torch.float32)
    else:
        audio_tensor = torch.as_tensor(audio).reshape(-1).float().cpu()

    sr_raw = multimodal_output.get("sr", default_sample_rate)
    if isinstance(sr_raw, list):
        sr_raw = sr_raw[-1] if sr_raw else default_sample_rate
    sample_rate = int(sr_raw.item()) if hasattr(sr_raw, "item") else int(sr_raw)
    return audio_tensor, sample_rate


def split_voxtream2_local_media_roots(value: Any) -> list[str]:
    if value is None or value is False:
        return []
    if isinstance(value, os.PathLike):
        return [os.fspath(value)]
    if isinstance(value, str):
        return [part for part in value.split(os.pathsep) if part]
    if isinstance(value, (list, tuple, set)):
        roots: list[str] = []
        for item in value:
            roots.extend(split_voxtream2_local_media_roots(item))
        return roots
    return []


def iter_voxtream2_local_audio_roots(allowed_local_media_path: Any = None) -> list[Path]:
    candidates = split_voxtream2_local_media_roots(allowed_local_media_path)
    for env_name in (
        VOXTREAM2_ALLOWED_LOCAL_MEDIA_PATH_ENV,
        "VOXTREAM2_ROOT",
        "VOXTREAM_ROOT",
        "VLLM_OMNI_VOXTREAM_CODE_PATH",
    ):
        candidates.extend(split_voxtream2_local_media_roots(os.environ.get(env_name)))

    roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            root = Path(candidate).expanduser().resolve()
        except OSError:
            continue
        if root in seen:
            continue
        seen.add(root)
        roots.append(root)
    return roots


def resolve_voxtream2_local_ref_audio_path(
    ref_audio: str,
    *,
    allowed_local_media_path: Any = None,
) -> str | None:
    parsed = urlparse(ref_audio)
    if parsed.scheme != "file":
        return None
    if parsed.netloc not in ("", "localhost"):
        raise ValueError("Voxtream2 ref_audio file URI must point to a local file")

    ref_path = Path(unquote(parsed.path)).expanduser().resolve()
    if not ref_path.is_file():
        raise ValueError(f"Voxtream2 ref_audio file does not exist: {ref_path}")

    roots = iter_voxtream2_local_audio_roots(allowed_local_media_path)
    if any(ref_path == root or root in ref_path.parents for root in roots):
        return str(ref_path)

    root_hint = ", ".join(str(root) for root in roots) if roots else "none"
    raise ValueError(
        "Voxtream2 local file ref_audio must be under one of the configured "
        f"media roots ({root_hint}). Set {VOXTREAM2_ALLOWED_LOCAL_MEDIA_PATH_ENV} "
        "or VOXTREAM2_ROOT to allow this path."
    )


async def prepare_voxtream2_ref_audio_path(
    ref_audio: str,
    *,
    resolve_ref_audio: Callable[[str], Awaitable[tuple[list[float], int]]],
    uploaded_speakers_dir: str | os.PathLike[str],
    allowed_local_media_path: Any = None,
) -> tuple[str, str | None]:
    ref_audio_path = resolve_voxtream2_local_ref_audio_path(
        ref_audio,
        allowed_local_media_path=allowed_local_media_path,
    )
    if ref_audio_path is not None:
        return ref_audio_path, None

    wav_samples, sr = await resolve_ref_audio(ref_audio)
    ref_audio_wav = np.asarray(wav_samples, dtype=np.float32)

    fd, tmp_path = tempfile.mkstemp(
        prefix="voxtream2_ref_",
        suffix=".wav",
        dir=str(uploaded_speakers_dir),
    )
    os.close(fd)
    sf.write(tmp_path, ref_audio_wav, sr, format="WAV")
    return tmp_path, tmp_path
