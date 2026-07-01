# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""External model loading, audio I/O, and emotion conditioning for IndexTTS2."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers.utils.hub import cached_file
from vllm.logger import init_logger

from vllm_omni.diffusion.model_loader.hub_prefetch import _repo_prefetch_lock

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy external model caches — keyed by (model_path, device) for multi-device
# ---------------------------------------------------------------------------

_wav2vec2_cache: dict[tuple[str, str], tuple] = {}
_semantic_codec_cache: dict[tuple[str, str], Any] = {}
_campplus_cache: dict[tuple[str, str], Any] = {}
_qwen_emotion_cache: dict[tuple[str, str], tuple] = {}


def _freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def resolve_model_file(model_path: str, filename: str) -> str | None:
    """Resolve an IndexTTS2 asset from a local model dir or HF repo id."""
    local_path = os.path.join(model_path, filename)
    if os.path.isfile(local_path):
        return local_path
    try:
        return cached_file(model_path, filename)
    except Exception as e:
        logger.warning("Could not resolve IndexTTS2 asset %s from %s: %s", filename, model_path, e)
        return None


def load_wav2vec2(model_path: str, device: torch.device):
    cache_key = (model_path, str(device))
    if cache_key in _wav2vec2_cache:
        return _wav2vec2_cache[cache_key]
    from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

    model, processor = None, None
    # Try local subfolder first (no lock needed), then HF hub with lock.
    try:
        src_args = dict(pretrained_model_name_or_path=model_path, subfolder="wav2vec2bert")
        processor = AutoFeatureExtractor.from_pretrained(**src_args)
        model = Wav2Vec2BertModel.from_pretrained(**src_args)
    except Exception:
        try:
            with _repo_prefetch_lock("facebook/w2v-bert-2.0"):
                src_args = dict(pretrained_model_name_or_path="facebook/w2v-bert-2.0")
                processor = AutoFeatureExtractor.from_pretrained(**src_args)
                model = Wav2Vec2BertModel.from_pretrained(**src_args)
        except Exception:
            pass
    if model is None or processor is None:
        raise RuntimeError(f"Could not load Wav2Vec2-BERT from {model_path}/wav2vec2bert or facebook/w2v-bert-2.0")
    _freeze(model.to(device=device, dtype=torch.float32))
    _wav2vec2_cache[cache_key] = (model, processor)
    return model, processor


def load_semantic_codec(model_path: str, config: dict, device: torch.device):
    cache_key = (model_path, str(device))
    if cache_key in _semantic_codec_cache:
        return _semantic_codec_cache[cache_key]
    from .utils.maskgct.repcodec_model import RepCodec

    codec = RepCodec(
        codebook_size=config.get("codebook_size", 8192),
        hidden_size=config.get("hidden_size", 1024),
        codebook_dim=config.get("codebook_dim", 8),
    )
    ckpt_path = resolve_model_file(model_path, "semantic_codec.pth")
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        codec.load_state_dict(state, strict=False)
    else:
        import safetensors.torch
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
        safetensors.torch.load_model(codec, ckpt_path)
    _freeze(codec.to(device=device, dtype=torch.float32))
    _semantic_codec_cache[cache_key] = codec
    return codec


def load_campplus(model_path: str, device: torch.device):
    cache_key = (model_path, str(device))
    if cache_key in _campplus_cache:
        return _campplus_cache[cache_key]
    from .utils.campplus.dtdnn import CAMPPlus

    campplus = CAMPPlus(feat_dim=80, embedding_size=192)
    ckpt_path = resolve_model_file(model_path, "campplus.pth")
    if ckpt_path is None:
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    campplus.load_state_dict(state, strict=False)
    _freeze(campplus.to(device=device, dtype=torch.float32))
    _campplus_cache[cache_key] = campplus
    return campplus


def load_qwen_emotion(model_path: str, device: torch.device, *, trust_remote_code: bool = True):
    cache_key = (model_path, str(device))
    if cache_key in _qwen_emotion_cache:
        return _qwen_emotion_cache[cache_key]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    subfolder = "qwen0.6bemo4-merge"
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            subfolder=subfolder,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            subfolder=subfolder,
            torch_dtype="float16",
            device_map={"": str(device)},
            trust_remote_code=trust_remote_code,
        )
    except Exception as e:
        logger.warning(
            "Could not load QwenEmotion from %s/%s: %s. use_emo_text will fall back to default emotion.",
            model_path,
            subfolder,
            e,
        )
        return None, None
    _freeze(model)
    _qwen_emotion_cache[cache_key] = (model, tokenizer)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------


def compute_fbank(wav_16k: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Compute 80-dim fbank features for CAMPPlus. Input: [T] at 16kHz."""
    try:
        import torchaudio.compliance.kaldi as kaldi

        # kaldi.fbank is pure torch — keep on the wav's device (GPU when
        # available) instead of forcing a CPU round-trip (~100ms for 15s).
        wav = wav_16k.unsqueeze(0).float()
        fbank = kaldi.fbank(
            wav,
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        fbank = fbank - fbank.mean(dim=0, keepdim=True)
        return fbank.unsqueeze(0).to(device=device)  # [1, T, 80]
    except (ImportError, RuntimeError, OSError):
        logger.warning("torchaudio not available, returning zero fbank")
        return torch.zeros(1, 100, 80, device=device)


def wav2vec_extract(
    wav_16k: torch.Tensor,
    model: Any,
    processor: Any,
    device: torch.device,
    w2v_stat: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract Wav2Vec2-BERT features. Returns [1, T, 1024]."""
    wav_np = wav_16k.cpu().numpy()
    inputs = processor(wav_np, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(device=device, dtype=torch.float32)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)
    with torch.no_grad():
        # bf16 autocast roughly halves the w2v-bert forward (~600M params,
        # fp32 weights). Features are cast back to fp32 before stat
        # normalization; downstream consumers are fp32.
        if input_features.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
        else:
            outputs = model(
                input_features=input_features,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
    feat = outputs.hidden_states[17].float()
    if w2v_stat is not None:
        mean = w2v_stat[0:1, :].to(device=device)
        std = torch.sqrt(w2v_stat[1:2, :].to(device=device))
        feat = (feat - mean) / std
    return feat


def load_reference_audio(
    audio_path: str | tuple | list,
    device: torch.device,
    max_audio_length_seconds: float | None = 15,
    mode: str = "speaker",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load reference audio and resample to 16kHz and 22.05kHz.

    Accepts either a file path (str) or a pre-loaded (wav_list, sr) tuple
    from the serving layer.

    ``mode`` mirrors official IndexTTS2 v2:
    - speaker: librosa default path first normalizes to 22.05kHz, truncates,
      then derives the 16kHz wav2vec/CAMPPlus input from that 22.05kHz signal.
    - emotion: librosa loads directly at 16kHz, then truncates.
    """
    if mode not in {"speaker", "emotion"}:
        raise ValueError("IndexTTS2 audio mode must be 'speaker' or 'emotion'")

    wav, sr = _load_audio_1d(audio_path)
    # Resample on GPU when available: CPU sinc resampling for awkward ratios
    # (e.g. 16000->22050, gcd=50) costs hundreds of ms; on GPU it is <5ms.
    # Downstream consumers move tensors to CPU explicitly where needed.
    if device.type == "cuda":
        wav = wav.to(device)
    if mode == "speaker":
        wav_22k = _resample(wav, sr, 22050)
        wav_22k = _truncate_audio(wav_22k, 22050, max_audio_length_seconds)
        wav_16k = _resample(wav_22k, 22050, 16000)
        return wav_16k, wav_22k

    wav_16k = _resample(wav, sr, 16000)
    wav_16k = _truncate_audio(wav_16k, 16000, max_audio_length_seconds)
    wav_22k = _resample(wav_16k, 16000, 22050)
    return wav_16k, wav_22k


def _load_audio_1d(audio_path: str | tuple | list) -> tuple[torch.Tensor, int]:
    if isinstance(audio_path, (list, tuple)) and len(audio_path) == 2:
        wav_data, sr = audio_path
        if isinstance(wav_data, np.ndarray):
            wav = torch.from_numpy(wav_data).float()
        elif isinstance(wav_data, list):
            # np.asarray + from_numpy is ~10x faster than torch.tensor(list)
            # for the long float lists the serving layer passes through msgspec.
            wav = torch.from_numpy(np.asarray(wav_data, dtype=np.float32))
        elif isinstance(wav_data, torch.Tensor):
            wav = wav_data.float()
        else:
            raise TypeError(f"Unsupported audio data type: {type(wav_data)}")
        return _mono_1d(wav), int(sr)

    try:
        import torchaudio

        wav, sr = torchaudio.load(audio_path)
        return _mono_1d(wav, channels_first=True), int(sr)
    except (ImportError, RuntimeError, OSError):
        pass

    import soundfile as sf

    audio_np, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    return _mono_1d(torch.from_numpy(audio_np).float(), channels_first=False), int(sr)


def _mono_1d(wav: torch.Tensor, channels_first: bool | None = None) -> torch.Tensor:
    if wav.ndim == 1:
        return wav.contiguous()
    if channels_first is True:
        return wav.mean(dim=0).contiguous()
    if channels_first is False:
        return wav.mean(dim=-1).contiguous()
    if wav.shape[0] <= 8 and wav.shape[-1] > wav.shape[0]:
        return wav.mean(dim=0).contiguous()
    return wav.mean(dim=-1).contiguous()


def _truncate_audio(
    wav: torch.Tensor,
    sample_rate: int,
    max_audio_length_seconds: float | None,
) -> torch.Tensor:
    if max_audio_length_seconds is None:
        return wav
    max_audio_samples = int(max_audio_length_seconds * sample_rate)
    if max_audio_samples > 0 and wav.shape[-1] > max_audio_samples:
        return wav[..., :max_audio_samples]
    return wav


def _resample(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return wav
    try:
        import torchaudio

        return torchaudio.functional.resample(wav, orig_sr, target_sr)
    except (ImportError, RuntimeError, OSError):
        return (
            F.interpolate(
                wav.unsqueeze(0).unsqueeze(0), scale_factor=target_sr / orig_sr, mode="linear", align_corners=False
            )
            .squeeze(0)
            .squeeze(0)
        )
