# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import io
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from vllm.multimodal.audio import AudioResampler

from vllm_omni.model_executor.models.minicpmo4_5.minicpmo4_5_code2wav import MiniCPMToken2wavCore

MODEL_PATH_ENV = "MINICPMO45_MODEL_PATH"
ASSET_DIR_ENV = "MINICPMO45_TOKEN2WAV_ASSETS"

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _resample_audio(audio: np.ndarray, *, orig_sr: int, target_sr: int) -> np.ndarray:
    if int(orig_sr) == int(target_sr):
        return np.asarray(audio, dtype=np.float32)
    resampler = AudioResampler(target_sr=int(target_sr))
    return np.asarray(resampler.resample(np.asarray(audio, dtype=np.float32), orig_sr=int(orig_sr)), dtype=np.float32)


def _resolve_assets_dir() -> Path:
    asset_dir = os.environ.get(ASSET_DIR_ENV)
    if asset_dir:
        path = Path(asset_dir).expanduser().resolve()
    else:
        model_path = os.environ.get(MODEL_PATH_ENV)
        if not model_path:
            pytest.skip(f"Set {MODEL_PATH_ENV} or {ASSET_DIR_ENV} to run MiniCPM Token2wav parity tests")
        path = Path(model_path).expanduser().resolve() / "assets" / "token2wav"
    if not path.is_dir():
        pytest.skip(f"MiniCPM Token2wav assets not found: {path}")
    return path


def _write_prompt_wav(tmp_path: Path) -> Path:
    sample_rate = 16000
    seconds = 1.2
    t = np.linspace(0.0, seconds, int(sample_rate * seconds), endpoint=False, dtype=np.float32)
    waveform = 0.15 * np.sin(2 * np.pi * 220.0 * t) + 0.05 * np.sin(2 * np.pi * 440.0 * t)
    prompt_path = tmp_path / "prompt.wav"
    sf.write(prompt_path, waveform.astype(np.float32), sample_rate)
    return prompt_path


def _decode_wav_bytes(wav_bytes: bytes) -> np.ndarray:
    audio, _ = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    audio_np = np.asarray(audio, dtype=np.float32)
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=-1)
    return audio_np.reshape(-1)


def _patch_upstream_audio_io(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("s3tokenizer")
    pytest.importorskip("stepaudio2")
    import s3tokenizer
    import s3tokenizer.utils as s3_utils
    import stepaudio2.token2wav as upstream_mod

    def load_audio(file: str | None, sr: int = 16000) -> torch.Tensor:
        if file is None:
            return torch.zeros((0,), dtype=torch.float32)
        audio, sample_rate = sf.read(file, dtype="float32", always_2d=False)
        audio_np = np.asarray(audio, dtype=np.float32)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=-1)
        audio_np = audio_np.reshape(-1)
        if int(sample_rate) != int(sr):
            audio_np = _resample_audio(audio_np, orig_sr=int(sample_rate), target_sr=int(sr))
        return torch.from_numpy(np.asarray(audio_np, dtype=np.float32))

    def torchaudio_load(file: str | None, *args, **kwargs) -> tuple[torch.Tensor, int]:
        if file is None:
            raise ValueError("prompt wav path is required")
        audio, sample_rate = sf.read(file, dtype="float32", always_2d=False)
        audio_np = np.asarray(audio, dtype=np.float32)
        if audio_np.ndim == 1:
            audio_np = audio_np[None, :]
        elif audio_np.ndim > 1:
            audio_np = np.asarray(audio_np, dtype=np.float32).T
        return torch.from_numpy(np.ascontiguousarray(audio_np, dtype=np.float32)), int(sample_rate)

    def torchaudio_save(file, src: torch.Tensor, sample_rate: int, *args, **kwargs) -> None:
        save_format = kwargs.pop("format", None)
        if save_format is not None:
            save_format = str(save_format).upper()
        audio_np = np.asarray(src.detach().cpu().numpy(), dtype=np.float32)
        if audio_np.ndim == 2:
            if audio_np.shape[0] == 1:
                audio_np = audio_np[0]
            else:
                audio_np = audio_np.T
        elif audio_np.ndim != 1:
            raise ValueError(f"Expected 1-D or 2-D audio tensor, got shape {tuple(audio_np.shape)}")
        sf.write(file, audio_np, int(sample_rate), format=save_format)

    monkeypatch.setattr(s3tokenizer, "load_audio", load_audio, raising=False)
    monkeypatch.setattr(s3_utils, "load_audio", load_audio, raising=False)
    monkeypatch.setattr(upstream_mod, "load_audio", load_audio, raising=False)
    monkeypatch.setattr(upstream_mod.s3tokenizer, "load_audio", load_audio, raising=False)
    monkeypatch.setattr(upstream_mod.torchaudio, "load", torchaudio_load, raising=False)
    monkeypatch.setattr(upstream_mod.torchaudio, "save", torchaudio_save, raising=False)


def _instantiate_upstream(asset_dir: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_upstream_audio_io(monkeypatch)
    from stepaudio2 import Token2wav

    return Token2wav(str(asset_dir), float16=False, n_timesteps=10)


def _instantiate_local(asset_dir: Path):
    return MiniCPMToken2wavCore(str(asset_dir), float16=False, n_timesteps=10, device="cuda")


def _cleanup_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _assert_waveforms_close(lhs: np.ndarray, rhs: np.ndarray) -> None:
    assert lhs.shape == rhs.shape
    torch.testing.assert_close(
        torch.from_numpy(lhs),
        torch.from_numpy(rhs),
        rtol=1e-4,
        atol=5e-4,
    )


def _prepare_stream_state(token2wav, prompt_wav: Path) -> None:
    stream_cache, hift_cache_dict = token2wav.set_stream_cache(str(prompt_wav))
    token2wav.stream_cache = stream_cache
    token2wav.hift_cache_dict = hift_cache_dict


def test_token2wav_full_decode_matches_upstream(tmp_path, monkeypatch):
    asset_dir = _resolve_assets_dir()
    prompt_wav = _write_prompt_wav(tmp_path)
    token_ids = [((idx * 137) % 6500) + 1 for idx in range(1, 33)]

    upstream = _instantiate_upstream(asset_dir, monkeypatch)
    upstream_audio = _decode_wav_bytes(upstream(token_ids, str(prompt_wav)))
    _cleanup_model(upstream)

    local = _instantiate_local(asset_dir)
    local_audio = _decode_wav_bytes(local(token_ids, str(prompt_wav)))
    _cleanup_model(local)

    _assert_waveforms_close(upstream_audio, local_audio)


def test_token2wav_streaming_matches_upstream(tmp_path, monkeypatch):
    asset_dir = _resolve_assets_dir()
    prompt_wav = _write_prompt_wav(tmp_path)
    token_ids = [((idx * 173) % 6500) + 1 for idx in range(1, 36)]
    first_chunk = [4218, 4218, 4218] + token_ids[:25]
    final_chunk = token_ids[22:]

    upstream = _instantiate_upstream(asset_dir, monkeypatch)
    _prepare_stream_state(upstream, prompt_wav)
    upstream_audio = np.concatenate(
        [
            np.asarray(
                upstream.stream(first_chunk, None, last_chunk=False, return_waveform=True),
                dtype=np.float32,
            ).reshape(-1),
            np.asarray(
                upstream.stream(final_chunk, None, last_chunk=True, return_waveform=True),
                dtype=np.float32,
            ).reshape(-1),
        ]
    )
    _cleanup_model(upstream)

    local = _instantiate_local(asset_dir)
    _prepare_stream_state(local, prompt_wav)
    local_audio = np.concatenate(
        [
            np.asarray(
                local.stream(first_chunk, None, last_chunk=False, return_waveform=True),
                dtype=np.float32,
            ).reshape(-1),
            np.asarray(
                local.stream(final_chunk, None, last_chunk=True, return_waveform=True),
                dtype=np.float32,
            ).reshape(-1),
        ]
    )
    _cleanup_model(local)

    _assert_waveforms_close(upstream_audio, local_audio)
