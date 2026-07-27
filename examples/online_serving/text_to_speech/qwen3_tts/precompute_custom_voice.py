"""Pre-compute Qwen3-TTS custom voice profiles.

The generated directory can be passed to the server via
``custom_voice_dir`` in ``vllm_omni/deploy/qwen3_tts.yaml``. Requests can then
use ``/v1/audio/speech`` with ``voice="<name>"`` and no per-request ref_audio.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni.utils.custom_voice_io import safe_voice_stem  # noqa: E402

MANIFEST_NAME = "custom_voice_manifest.json"
_SPEAKER_ENCODER_SAMPLE_RATE = 24000
_SPEAKER_ENCODER_NUM_MELS = 128
_SPEAKER_ENCODER_N_FFT = 1024
_SPEAKER_ENCODER_HOP_SIZE = 256
_SPEAKER_ENCODER_WIN_SIZE = 1024
_SPEAKER_ENCODER_FMIN = 0
_SPEAKER_ENCODER_FMAX = 12000


def _resolve_model_dir(model: str) -> str:
    if os.path.isdir(model):
        return model
    from transformers.utils.hub import cached_file

    config_path = cached_file(model, "config.json")
    if config_path is None:
        raise FileNotFoundError(f"Could not resolve config.json for {model}")
    return os.path.dirname(config_path)


def _read_audio_mono(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    return np.asarray(wav, dtype=np.float32), int(sr)


def _resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return wav.astype(np.float32)
    from vllm.multimodal.audio import AudioResampler

    return AudioResampler(target_sr=target_sr).resample(wav.astype(np.float32), orig_sr=int(sr))


def _load_speaker_encoder(model_dir: str, device: torch.device):
    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import Qwen3TTSSpeakerEncoder

    config = Qwen3TTSConfig.from_pretrained(model_dir)
    encoder = Qwen3TTSSpeakerEncoder(config.speaker_encoder_config)
    index_path = Path(model_dir) / "model.safetensors.index.json"
    state: dict[str, torch.Tensor] = {}
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        shard_names = sorted({v for k, v in weight_map.items() if k.startswith("speaker_encoder.")})
        shard_paths = [Path(model_dir) / name for name in shard_names]
    else:
        shard_paths = sorted(Path(model_dir).glob("model*.safetensors"))
    for shard in shard_paths:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("speaker_encoder."):
                    state[key.removeprefix("speaker_encoder.")] = f.get_tensor(key)
    if not state:
        raise RuntimeError(
            f"No speaker_encoder.* tensors found under {model_dir}. "
            "Use a Qwen3-TTS Base checkpoint for pre-computing custom voice profiles."
        )
    encoder.load_state_dict(state)
    encoder.to(device=device, dtype=torch.bfloat16).eval()
    return config, encoder


def _speaker_embedding(
    encoder: torch.nn.Module,
    speaker_encoder_config: object,
    wav: np.ndarray,
    sr: int,
    device: torch.device,
) -> torch.Tensor:
    from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import mel_spectrogram

    target_sr = int(getattr(speaker_encoder_config, "sample_rate", _SPEAKER_ENCODER_SAMPLE_RATE))
    mel_dim = int(getattr(speaker_encoder_config, "mel_dim", _SPEAKER_ENCODER_NUM_MELS))
    if target_sr != _SPEAKER_ENCODER_SAMPLE_RATE or mel_dim != _SPEAKER_ENCODER_NUM_MELS:
        raise ValueError(
            "Unsupported Qwen3-TTS speaker encoder mel config for precompute: "
            f"sample_rate={target_sr}, mel_dim={mel_dim}. "
            f"Expected sample_rate={_SPEAKER_ENCODER_SAMPLE_RATE}, mel_dim={_SPEAKER_ENCODER_NUM_MELS} "
            "to match the serving-time speaker embedding path."
        )
    wav = _resample(wav, sr, target_sr)
    wav_t = torch.from_numpy(wav).to(device=device, dtype=torch.float32)
    mels = mel_spectrogram(
        wav_t.unsqueeze(0),
        n_fft=_SPEAKER_ENCODER_N_FFT,
        num_mels=_SPEAKER_ENCODER_NUM_MELS,
        sampling_rate=_SPEAKER_ENCODER_SAMPLE_RATE,
        hop_size=_SPEAKER_ENCODER_HOP_SIZE,
        win_size=_SPEAKER_ENCODER_WIN_SIZE,
        fmin=_SPEAKER_ENCODER_FMIN,
        fmax=_SPEAKER_ENCODER_FMAX,
    ).transpose(1, 2)
    with torch.inference_mode():
        return encoder(mels.to(device=device, dtype=torch.bfloat16))[0].float().cpu().contiguous()


def _ref_code(model_dir: str, wav: np.ndarray, sr: int, device: torch.device) -> torch.Tensor:
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_tokenizer import Qwen3TTSTokenizer

    tok = Qwen3TTSTokenizer.from_pretrained(
        str(Path(model_dir) / "speech_tokenizer"),
        torch_dtype=torch.bfloat16,
    )
    try:
        del tok.model.decoder
        tok.model.decoder = None
        tok.model.encoder.to(device)
        tok.device = device
    except Exception:
        tok.device = device
    with torch.inference_mode():
        enc = tok.encode(wav, sr=int(sr), return_dict=True)
    codes = getattr(enc, "audio_codes", None)
    if isinstance(codes, list):
        codes = codes[0] if codes else None
    if not isinstance(codes, torch.Tensor):
        raise RuntimeError("Speech tokenizer did not return audio_codes")
    if codes.ndim == 3:
        codes = codes[0]
    return codes.to(dtype=torch.int32, device="cpu").contiguous()


def _load_manifest(output_dir: Path, model: str, hidden_size: int) -> dict[str, Any]:
    path = output_dir / MANIFEST_NAME
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "model_type": "qwen3_tts",
        "model": model,
        "hidden_size": hidden_size,
        "voices": {},
    }


def _write_voice(
    *,
    model: str,
    output_dir: Path,
    voice_name: str,
    ref_audio: str,
    ref_text: str | None,
    mode: str,
    speaker_description: str | None,
    device: torch.device,
) -> None:
    model_dir = _resolve_model_dir(model)
    config, encoder = _load_speaker_encoder(model_dir, device)
    wav, sr = _read_audio_mono(ref_audio)
    if wav.size < 1024:
        raise ValueError(f"Reference audio too short: {wav.size} samples")

    tensors: dict[str, torch.Tensor] = {
        "speaker_embedding": _speaker_embedding(encoder, config.speaker_encoder_config, wav, sr, device),
    }
    if mode == "icl":
        if not ref_text or not ref_text.strip():
            raise ValueError("--ref-text is required for --mode icl")
        tensors["ref_code"] = _ref_code(model_dir, wav, sr, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{safe_voice_stem(voice_name)}.safetensors"
    save_file(tensors, str(output_dir / filename))

    hidden_size = int(getattr(config.talker_config, "hidden_size", tensors["speaker_embedding"].numel()))
    manifest = _load_manifest(output_dir, model, hidden_size)
    entry: dict[str, Any] = {
        "name": voice_name,
        "file": filename,
        "mode": mode,
        "embedding_dim": int(tensors["speaker_embedding"].numel()),
    }
    if "ref_code" in tensors:
        entry["ref_code_length"] = int(tensors["ref_code"].shape[0])
    if ref_text:
        entry["ref_text"] = ref_text
    if speaker_description:
        entry["speaker_description"] = speaker_description
    manifest.setdefault("voices", {})[voice_name] = entry
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output_dir / filename}")
    print(f"Updated {output_dir / MANIFEST_NAME}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute Qwen3-TTS custom voice profile")
    parser.add_argument("--model", required=True, help="Qwen3-TTS Base model path or Hugging Face ID")
    parser.add_argument("--voice-name", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--ref-text", default=None)
    parser.add_argument("--mode", choices=["xvec", "icl"], default="xvec")
    parser.add_argument("--speaker-description", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    _write_voice(
        model=args.model,
        output_dir=Path(args.output_dir),
        voice_name=args.voice_name,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        mode=args.mode,
        speaker_description=args.speaker_description,
        device=torch.device(args.device),
    )


if __name__ == "__main__":
    main()
