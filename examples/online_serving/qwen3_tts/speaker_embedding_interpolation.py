"""Speaker embedding extraction and interpolation for Qwen3-TTS.

Extracts speaker embeddings from reference audio files using the ECAPA-TDNN
speaker encoder from a Qwen3-TTS checkpoint, then interpolates between them
using SLERP and sends the result to the /v1/audio/speech API.

Requirements:
    pip install torch librosa soundfile numpy httpx

Examples:
    # Extract and save an embedding
    python speaker_embedding_interpolation.py extract \
        --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --audio voice_a.wav \
        --output voice_a_embedding.json

    # Interpolate between two embeddings and generate speech
    python speaker_embedding_interpolation.py interpolate \
        --embedding-a voice_a_embedding.json \
        --embedding-b voice_b_embedding.json \
        --ratio 0.5 \
        --text "Hello, this is a blended voice." \
        --output blended.wav

    # Full pipeline: extract from two audio files + interpolate + generate
    python speaker_embedding_interpolation.py pipeline \
        --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --audio-a voice_a.wav \
        --audio-b voice_b.wav \
        --ratios 0.0 0.25 0.5 0.75 1.0 \
        --text "Hello, this is a blended voice." \
        --output-dir ./interpolated/
"""

import argparse
import json
import os
import sys

import httpx
import numpy as np
import torch

DEFAULT_API_BASE = "http://localhost:8000"
DEFAULT_API_KEY = "EMPTY"

# ──────────────────────────────────────────────
# Speaker embedding extraction (offline)
# ──────────────────────────────────────────────


def load_speaker_encoder(model_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load just the ECAPA-TDNN speaker encoder from a Qwen3-TTS checkpoint.

    This avoids loading the full TTS model by only instantiating the speaker
    encoder sub-module from the checkpoint weights.
    """
    from transformers import AutoConfig

    # Register the config class so AutoConfig can resolve it
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
            Qwen3TTSConfig,
        )
        from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
            Qwen3TTSSpeakerEncoder,
        )

        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    except ImportError:
        # If running outside the vllm-omni tree, try importing from the HF hub
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # Dynamically import from the downloaded model files
        import importlib

        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(model_path)
        spec = importlib.util.spec_from_file_location(
            "modeling_qwen3_tts",
            os.path.join(model_dir, "modeling_qwen3_tts.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        Qwen3TTSSpeakerEncoder = mod.Qwen3TTSSpeakerEncoder

        spec2 = importlib.util.spec_from_file_location(
            "configuration_qwen3_tts",
            os.path.join(model_dir, "configuration_qwen3_tts.py"),
        )
        mod2 = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(mod2)
        Qwen3TTSConfig = mod2.Qwen3TTSConfig
        config_obj = Qwen3TTSConfig.from_pretrained(model_path)
        encoder = Qwen3TTSSpeakerEncoder(config_obj.speaker_encoder_config)
        # Load weights
        _load_speaker_encoder_weights(encoder, model_path)
        encoder = encoder.to(device).eval()
        return encoder

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    encoder = Qwen3TTSSpeakerEncoder(config.speaker_encoder_config)

    _load_speaker_encoder_weights(encoder, model_path)
    encoder = encoder.to(device).eval()
    return encoder


def _load_speaker_encoder_weights(encoder: torch.nn.Module, model_path: str) -> None:
    """Load only the speaker_encoder.* weights from the checkpoint."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    model_dir = snapshot_download(model_path)

    prefix = "speaker_encoder."
    state_dict = {}

    # Try safetensors first, then pytorch bin
    safetensor_files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    if safetensor_files:
        for fname in safetensor_files:
            shard = load_file(os.path.join(model_dir, fname))
            for k, v in shard.items():
                if k.startswith(prefix):
                    state_dict[k[len(prefix) :]] = v
    else:
        bin_files = sorted(f for f in os.listdir(model_dir) if f.endswith(".bin"))
        for fname in bin_files:
            shard = torch.load(os.path.join(model_dir, fname), map_location="cpu", weights_only=True)
            for k, v in shard.items():
                if k.startswith(prefix):
                    state_dict[k[len(prefix) :]] = v

    if not state_dict:
        raise RuntimeError(
            f"No speaker_encoder weights found in {model_path}. Make sure this is a Qwen3-TTS-*-Base checkpoint."
        )

    encoder.load_state_dict(state_dict)


def compute_mel_spectrogram(audio: np.ndarray, sr: int = 24000) -> torch.Tensor:
    """Compute 128-bin mel spectrogram matching Qwen3-TTS's extraction pipeline."""
    import librosa

    # Resample to 24kHz if needed
    if sr != 24000:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=24000)

    y = torch.from_numpy(audio).unsqueeze(0).float()

    from librosa.filters import mel as librosa_mel_fn

    mel_basis = torch.from_numpy(librosa_mel_fn(sr=24000, n_fft=1024, n_mels=128, fmin=0, fmax=12000)).float()

    n_fft = 1024
    hop_size = 256
    win_size = 1024
    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(y.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

    hann_window = torch.hann_window(win_size)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=False,
        return_complex=True,
    )
    spec = torch.abs(spec)
    mel = torch.matmul(mel_basis, spec)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel.transpose(1, 2)  # (1, T, 128)


@torch.inference_mode()
def extract_embedding(encoder: torch.nn.Module, audio_path: str, device: str = "cpu") -> np.ndarray:
    """Extract a 1024-dim speaker embedding from an audio file."""
    import librosa

    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    mel = compute_mel_spectrogram(audio, sr).to(device)
    embedding = encoder(mel.to(next(encoder.parameters()).dtype))[0]
    return embedding.float().cpu().numpy()


# ──────────────────────────────────────────────
# Interpolation
# ──────────────────────────────────────────────


def slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two vectors."""
    v0_norm = v0 / (np.linalg.norm(v0) + 1e-8)
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)

    dot = np.clip(np.dot(v0_norm, v1_norm), -1.0, 1.0)
    omega = np.arccos(dot)

    if np.abs(omega) < 1e-6:
        # Vectors are nearly parallel, fall back to lerp
        return (1.0 - t) * v0 + t * v1

    sin_omega = np.sin(omega)
    return (np.sin((1.0 - t) * omega) / sin_omega) * v0 + (np.sin(t * omega) / sin_omega) * v1


# ──────────────────────────────────────────────
# API client
# ──────────────────────────────────────────────


def generate_speech(
    text: str,
    speaker_embedding: list[float],
    output_path: str,
    api_base: str = DEFAULT_API_BASE,
    api_key: str = DEFAULT_API_KEY,
    model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
) -> None:
    """Send a speaker_embedding to the TTS API and save the output."""
    payload = {
        "model": model,
        "input": text,
        "task_type": "Base",
        "speaker_embedding": speaker_embedding,
        "response_format": "wav",
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    with httpx.Client(timeout=300.0) as client:
        resp = client.post(f"{api_base}/v1/audio/speech", json=payload, headers=headers)

    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        sys.exit(1)

    with open(output_path, "wb") as f:
        f.write(resp.content)
    print(f"Saved: {output_path}")


# ──────────────────────────────────────────────
# CLI commands
# ──────────────────────────────────────────────


def cmd_extract(args):
    """Extract a speaker embedding from audio and save as JSON."""
    print(f"Loading speaker encoder from {args.model}...")
    encoder = load_speaker_encoder(args.model, device=args.device)

    print(f"Extracting embedding from {args.audio}...")
    emb = extract_embedding(encoder, args.audio, device=args.device)
    print(f"Embedding shape: {emb.shape}")

    output_path = args.output or os.path.splitext(args.audio)[0] + "_embedding.json"
    with open(output_path, "w") as f:
        json.dump(emb.tolist(), f)
    print(f"Saved embedding to {output_path}")


def cmd_interpolate(args):
    """Interpolate between two embeddings and generate speech."""
    with open(args.embedding_a) as f:
        emb_a = np.array(json.load(f), dtype=np.float32)
    with open(args.embedding_b) as f:
        emb_b = np.array(json.load(f), dtype=np.float32)

    print(f"Embedding A: {args.embedding_a} (dim={emb_a.shape[0]})")
    print(f"Embedding B: {args.embedding_b} (dim={emb_b.shape[0]})")
    print(f"SLERP ratio: {args.ratio}")

    blended = slerp(emb_a, emb_b, args.ratio)
    output_path = args.output or f"interpolated_t{args.ratio:.2f}.wav"

    print(f"Generating speech: {args.text!r}")
    generate_speech(
        text=args.text,
        speaker_embedding=blended.tolist(),
        output_path=output_path,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
    )


def cmd_pipeline(args):
    """Full pipeline: extract two embeddings, SLERP at multiple ratios, generate."""
    print(f"Loading speaker encoder from {args.model}...")
    encoder = load_speaker_encoder(args.model, device=args.device)

    print(f"Extracting embedding A from {args.audio_a}...")
    emb_a = extract_embedding(encoder, args.audio_a, device=args.device)
    print(f"Extracting embedding B from {args.audio_b}...")
    emb_b = extract_embedding(encoder, args.audio_b, device=args.device)

    # Save extracted embeddings
    os.makedirs(args.output_dir, exist_ok=True)
    for label, emb, audio_path in [("a", emb_a, args.audio_a), ("b", emb_b, args.audio_b)]:
        emb_path = os.path.join(args.output_dir, f"embedding_{label}.json")
        with open(emb_path, "w") as f:
            json.dump(emb.tolist(), f)
        print(f"Saved embedding {label} to {emb_path}")

    # Generate at each ratio
    for t in args.ratios:
        blended = slerp(emb_a, emb_b, t)
        out_path = os.path.join(args.output_dir, f"t{t:.2f}.wav")
        print(f"\n--- SLERP t={t:.2f} ---")
        print(f"Generating: {args.text!r}")
        generate_speech(
            text=args.text,
            speaker_embedding=blended.tolist(),
            output_path=out_path,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
        )

    print(f"\nAll outputs saved to {args.output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Speaker embedding extraction and interpolation for Qwen3-TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="TTS API base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Model name (used for both weight loading and API requests)",
    )
    parser.add_argument("--device", default="cpu", help="Device for embedding extraction (cpu/cuda)")

    sub = parser.add_subparsers(dest="command", required=True)

    # extract
    p_ext = sub.add_parser("extract", help="Extract speaker embedding from audio")
    p_ext.add_argument("--audio", required=True, help="Input audio file")
    p_ext.add_argument("--output", "-o", help="Output JSON path (default: <audio>_embedding.json)")

    # interpolate
    p_interp = sub.add_parser("interpolate", help="Interpolate between two embeddings and generate speech")
    p_interp.add_argument("--embedding-a", required=True, help="JSON file with embedding A")
    p_interp.add_argument("--embedding-b", required=True, help="JSON file with embedding B")
    p_interp.add_argument("--ratio", "-r", type=float, default=0.5, help="SLERP ratio (0=A, 1=B)")
    p_interp.add_argument("--text", required=True, help="Text to synthesize")
    p_interp.add_argument("--output", "-o", help="Output wav path")

    # pipeline
    p_pipe = sub.add_parser("pipeline", help="Full pipeline: extract + interpolate + generate")
    p_pipe.add_argument("--audio-a", required=True, help="Audio file for voice A")
    p_pipe.add_argument("--audio-b", required=True, help="Audio file for voice B")
    p_pipe.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="SLERP ratios to generate (default: 0.0 0.25 0.5 0.75 1.0)",
    )
    p_pipe.add_argument("--text", required=True, help="Text to synthesize")
    p_pipe.add_argument("--output-dir", default="./interpolated", help="Output directory")

    args = parser.parse_args()
    {"extract": cmd_extract, "interpolate": cmd_interpolate, "pipeline": cmd_pipeline}[args.command](args)


if __name__ == "__main__":
    main()
