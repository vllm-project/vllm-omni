"""Shared helpers for SoulX preprocess (I/O, config, checkpoints, pitch)."""

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from vllm.logger import init_logger
from vllm.multimodal.audio import AudioResampler
from vllm.multimodal.media.audio import load_audio

logger = init_logger(__name__)


# --- audio I/O ---


def resample_mono(
    audio: np.ndarray,
    *,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample mono float waveform to ``target_sr``."""
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    if orig_sr == target_sr:
        return audio
    resampler = AudioResampler(target_sr=float(target_sr))
    return np.asarray(
        resampler.resample(audio, orig_sr=float(orig_sr)),
        dtype=np.float32,
    )


def get_audio_duration(path: str) -> float:
    """Return audio duration in seconds for a file on disk."""
    return float(sf.info(path).duration)


def load_mono_audio(
    source: str | tuple[np.ndarray, int],
    *,
    target_sr: int | None = None,
) -> tuple[np.ndarray, int]:
    if isinstance(source, tuple):
        audio, sr = source
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        if target_sr is not None and sr != target_sr:
            audio = resample_mono(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return audio, sr

    if target_sr is None:
        audio, sr = load_audio(str(source), sr=None, mono=True)
    else:
        audio, sr = load_audio(str(source), sr=target_sr, mono=True)
    return np.asarray(audio, dtype=np.float32).squeeze(), int(sr)


# --- tensor padding ---


def pad_or_cut_xd(values: torch.Tensor, tgt_len: int, dim: int = -1, pad_value=0) -> torch.Tensor:
    ndim = values.ndim
    if ndim == 1:
        if values.shape[0] < tgt_len:
            return F.pad(values, [0, tgt_len - values.shape[0]], value=pad_value)
        return values[:tgt_len]
    if ndim == 2:
        if dim in (0, -2):
            src_len = values.shape[0]
            if src_len < tgt_len:
                return F.pad(values, [0, 0, 0, tgt_len - src_len], value=pad_value)
            return values[:tgt_len]
        src_len = values.shape[1]
        if src_len < tgt_len:
            return F.pad(values, [0, tgt_len - src_len], value=pad_value)
        return values[:, :tgt_len]
    raise NotImplementedError(f"pad_or_cut_xd supports 1D/2D tensors, got ndim={ndim}")


# --- YAML config (OmegaConf) ---


def _apply_hparams_str(config: dict[str, Any], hparams_str: str) -> None:
    if not hparams_str:
        return
    for item in hparams_str.split(","):
        key, raw_value = item.split("=", maxsplit=1)
        raw_value = raw_value.strip().strip("'\"")
        node: dict[str, Any] = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        leaf = parts[-1]
        current = node.get(leaf)
        if raw_value in {"True", "False"}:
            node[leaf] = raw_value == "True"
        elif isinstance(current, bool):
            node[leaf] = raw_value.lower() == "true"
        elif isinstance(current, int):
            node[leaf] = int(raw_value)
        elif isinstance(current, float):
            node[leaf] = float(raw_value)
        else:
            node[leaf] = type(current)(raw_value) if current is not None else raw_value


def load_yaml_config(
    config: str | Path,
    *,
    hparams_str: str = "",
) -> dict[str, Any]:
    """Load YAML with optional ``base_config`` chain; returns a plain dict."""
    config_path = Path(config).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    merged = OmegaConf.create({})
    pending = [config_path.resolve()]
    loaded: set[str] = set()

    while pending:
        path = pending.pop(0)
        path_key = str(path)
        if path_key in loaded:
            continue
        loaded.add(path_key)

        current = OmegaConf.load(path_key)
        base_configs = current.get("base_config")
        if base_configs is not None:
            entries = base_configs if OmegaConf.is_list(base_configs) else [base_configs]
            for entry in entries:
                base_path = Path(entry)
                if not base_path.is_absolute():
                    base_path = (path.parent / entry).resolve()
                if not base_path.is_file():
                    logger.warning("Skipping missing base config: %s", base_path)
                    continue
                pending.insert(0, base_path)
        merged = OmegaConf.merge(current, merged)

    result = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(result, dict):
        raise TypeError(f"Expected dict config, got {type(result)}")
    _apply_hparams_str(result, hparams_str)
    return result


def load_rosvot_config(config: str | Path, *, hparams_str: str = "") -> dict[str, Any]:
    return load_yaml_config(config, hparams_str=hparams_str)


# --- checkpoint loading ---


def load_model_ckpt(
    model: nn.Module,
    checkpoint_path: str,
    *,
    model_name: str = "model",
    verbose: bool = True,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Unexpected ROSVOT checkpoint format: {checkpoint_path}")

    state_dict = checkpoint["state_dict"]
    if state_dict and "." in next(iter(state_dict)):
        prefix = f"{model_name}."
        state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
    elif model_name in state_dict and isinstance(state_dict[model_name], dict):
        state_dict = state_dict[model_name]

    model.load_state_dict(state_dict, strict=True)
    if verbose:
        logger.info("Loaded ROSVOT weights from %s", checkpoint_path)


# --- pitch helpers (ROSVOT / RMVPE) ---


def f0_to_coarse(f0, f0_bin=256, f0_max=900.0, f0_min=50.0):
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    is_torch = isinstance(f0, torch.Tensor)
    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    return (f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(int)


def norm_f0(f0, uv, pitch_norm="log", f0_mean=400, f0_std=100):
    is_torch = isinstance(f0, torch.Tensor)
    if pitch_norm == "standard":
        f0 = (f0 - f0_mean) / f0_std
    if pitch_norm == "log":
        f0 = torch.log2(f0 + 1e-8) if is_torch else np.log2(f0 + 1e-8)
    if uv is not None:
        f0[uv > 0] = 0
    return f0


def norm_interp_f0(f0, pitch_norm="log", f0_mean=None, f0_std=None):
    is_torch = isinstance(f0, torch.Tensor)
    if is_torch:
        device = f0.device
        f0 = f0.detach().cpu().numpy()
    uv = f0 == 0
    f0 = norm_f0(f0, uv, pitch_norm, f0_mean, f0_std)
    if sum(uv) == len(f0):
        f0[uv] = 0
    elif sum(uv) > 0:
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    if is_torch:
        uv_t = torch.FloatTensor(uv)
        f0_t = torch.FloatTensor(f0)
        return f0_t.to(device), uv_t.to(device)
    return f0, uv


def denorm_f0(f0, uv, pitch_norm="log", f0_mean=400, f0_std=100, pitch_padding=None, min=50, max=900):
    is_torch = isinstance(f0, torch.Tensor)
    if pitch_norm == "standard":
        f0 = f0 * f0_std + f0_mean
    if pitch_norm == "log":
        f0 = 2**f0
    f0 = f0.clamp(min=min, max=max) if is_torch else np.clip(f0, a_min=min, a_max=max)
    if uv is not None:
        f0[uv > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0


def boundary2Interval(bd):
    is_torch = isinstance(bd, torch.Tensor)
    if is_torch:
        device = bd.device
        bd = bd.detach().cpu().numpy()
    ret = np.zeros(shape=(bd.sum() + 1, 2), dtype=int)
    ret_idx = 0
    ret[0, 0] = 0
    for i, u in enumerate(bd):
        if i == 0:
            continue
        if u == 1:
            ret[ret_idx, 1] = i
            ret[ret_idx + 1, 0] = i
            ret_idx += 1
    ret[-1, 1] = bd.shape[0] - 1
    if is_torch:
        return torch.LongTensor(ret).to(device)
    return ret
