# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from vllm.logger import init_logger

from vllm_omni.engine.stage_init_utils import _resolve_model_to_local_path
from vllm_omni.model_executor.models.common.ming.audio_vae import AudioVAE

from .audio_prep import (
    _coerce_prompt_latents,
    _normalize_prompt_waveform,
    count_prompt_latent_patches,
    pad_prompt_waveform,
)
from .config_ming_tts import KEY_PROMPT_LATENTS

logger = init_logger(__name__)
_PROMPT_ENCODER_LOAD_LOCK = threading.Lock()


def _resolve_prompt_latents(wrapper: Any, info_dict: dict[str, Any]) -> dict[str, torch.Tensor] | None:
    raw_latents = info_dict.get(KEY_PROMPT_LATENTS, info_dict.get("prompt_latents"))
    raw_waveform = info_dict.get("prompt_waveform", info_dict.get("prompt_waveforms"))
    if raw_latents is not None and raw_waveform is not None:
        raise ValueError(
            "Ming waveform cloning request provided both raw prompt_waveform and explicit prompt_latents. "
            "Choose exactly one source of truth."
        )

    direct_latents = _coerce_prompt_latents(
        raw_latents,
        patch_size=wrapper.ming_config.patch_size,
        latent_dim=wrapper.ming_config.latent_dim,
    )
    if direct_latents is not None:
        return direct_latents
    if raw_waveform is None:
        return None

    encode_fn = getattr(wrapper, "_encode_prompt_waveform_to_latents", None)
    if callable(encode_fn):
        latents = encode_fn(raw_waveform, info_dict.get("prompt_waveform_length"))
    else:
        latents = _encode_prompt_waveform_to_latents(
            wrapper,
            raw_waveform,
            info_dict.get("prompt_waveform_length"),
        )
    return _coerce_prompt_latents(
        latents,
        patch_size=wrapper.ming_config.patch_size,
        latent_dim=wrapper.ming_config.latent_dim,
    )


def _load_prompt_encoder(wrapper: Any) -> AudioVAE:
    if wrapper._prompt_encoder is not None:
        return wrapper._prompt_encoder
    with _PROMPT_ENCODER_LOAD_LOCK:
        if wrapper._prompt_encoder is not None:
            return wrapper._prompt_encoder
        if wrapper.ming_config.audio_tokenizer_config is None:
            raise RuntimeError("Ming Stage-0 requires audio_tokenizer_config to encode prompt audio.")

        load_start = time.perf_counter()
        encoder = AudioVAE(wrapper.ming_config.audio_tokenizer_config).eval()
        state_dict = encoder.state_dict()
        loaded = 0
        loaded_encoder_params = set()
        with torch.no_grad():
            for shard_path in _iter_model_safetensors(
                _resolve_model_to_local_path(str(wrapper.vllm_config.model_config.model))
            ):
                with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        if not key.startswith("audio.encoder."):
                            continue
                        name = key[len("audio.") :]
                        if name not in state_dict:
                            continue
                        target = state_dict[name]
                        target.copy_(handle.get_tensor(key).to(device=target.device, dtype=target.dtype))
                        loaded += 1
                        loaded_encoder_params.add(name)
        if loaded == 0:
            raise RuntimeError("Ming prompt encoder received no audio.encoder.* weights from checkpoint.")

        expected_encoder_params = {f"encoder.{name}" for name, _ in encoder.encoder.named_parameters()}
        missing = expected_encoder_params - loaded_encoder_params
        if missing:
            raise RuntimeError(
                f"Ming prompt encoder: {len(missing)} params not loaded. First few: {sorted(missing)[:5]}"
            )

        dev = next(wrapper.parameters()).device
        try:
            del encoder.decoder
            encoder.decoder = None
            if dev.type != "cpu":
                encoder.encoder.to(dev, dtype=getattr(wrapper.model, "fm_dtype", torch.bfloat16))
            else:
                encoder.encoder.to(dev)
        except Exception as exc:
            raise RuntimeError(f"Failed to move Ming prompt encoder to {dev}: {exc}") from exc
        wrapper._prompt_encoder = encoder
        logger.info("Ming prompt encoder cold-loaded in %.3f ms", (time.perf_counter() - load_start) * 1000.0)
        return encoder


@torch.inference_mode()
def _encode_prompt_waveform_to_latents(wrapper: Any, waveform: Any, waveform_length: Any = None) -> torch.Tensor:
    encoder = _load_prompt_encoder(wrapper)
    waveform = _normalize_prompt_waveform(waveform, target_sr=wrapper.ming_config.sample_rate)
    waveform = pad_prompt_waveform(
        waveform,
        patch_size=wrapper.ming_config.patch_size,
        sample_rate=wrapper.ming_config.sample_rate,
    )
    dev = next(encoder.encoder.parameters()).device
    waveform = waveform.to(device=dev, dtype=next(encoder.encoder.parameters()).dtype)
    if waveform_length is None:
        waveform_length = torch.full((waveform.shape[0],), waveform.shape[-1], dtype=torch.int32, device=dev)
    elif not isinstance(waveform_length, torch.Tensor):
        waveform_length = torch.as_tensor(waveform_length, dtype=torch.int32, device=dev)
    else:
        waveform_length = waveform_length.to(device=dev, dtype=torch.int32)

    latents, _ = encoder.encode_latent(waveform, waveform_length)
    if latents.ndim == 3 and latents.shape[0] == 1:
        latents = latents.squeeze(0)
    count_prompt_latent_patches(
        latents,
        patch_size=wrapper.ming_config.patch_size,
        latent_dim=wrapper.ming_config.latent_dim,
    )
    return latents.detach().to(dtype=torch.float32).contiguous()


def _iter_model_safetensors(local_model_path: str) -> list[Path]:
    model_root = Path(local_model_path)
    index_path = model_root / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as handle:
            index_data = json.load(handle)
        filenames = sorted(set(index_data.get("weight_map", {}).values()))
        if not filenames:
            raise RuntimeError(f"No checkpoint shards listed in {index_path}")
        return [model_root / filename for filename in filenames]

    single_file = model_root / "model.safetensors"
    if single_file.exists():
        return [single_file]

    files = sorted(model_root.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"No .safetensors checkpoint found under {local_model_path}")
    return files


__all__ = [
    "_encode_prompt_waveform_to_latents",
    "_iter_model_safetensors",
    "_load_prompt_encoder",
    "_resolve_prompt_latents",
]
