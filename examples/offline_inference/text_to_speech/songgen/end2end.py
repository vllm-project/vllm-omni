# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline inference example for SongGen via vLLM-Omni.

Single-stage pipeline: the 1.3B AR LM and X-Codec decoder both run inside one
generation stage. Output is 16 kHz mono WAV.

SongGen takes two text inputs:
  --lyrics       : song lyrics (English only)
  --description  : music style / genre description (e.g. "upbeat pop song with
                   electric guitar and female vocals")

Optional voice conditioning:
  --ref-voice    : path to reference voice audio (WAV/MP3) for voice timbre
                   conditioning. Should be pre-separated vocals (no Demucs run).

Usage:
  # Basic text-to-song (no voice conditioning):
  python end2end.py \\
    --lyrics "Under the moonlight, we dance through the night" \\
    --description "dreamy pop ballad with piano and strings"

  # With voice conditioning:
  python end2end.py \\
    --lyrics "Under the moonlight, we dance through the night" \\
    --description "dreamy pop ballad with piano and strings" \\
    --ref-voice /path/to/reference_voice.wav

Prerequisites:
  pip install git+https://github.com/LiuZH-19/SongGen.git
  # Download model:
  python -c "from huggingface_hub import snapshot_download; \\
             snapshot_download('LiuZH-19/SongGen_mixed_pro')"
"""

from __future__ import annotations

import os
from pathlib import Path

import soundfile as sf
import torch
from vllm import SamplingParams

from vllm_omni.utils.tracking_parser import TrackingArgumentParser

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm_omni import Omni  # noqa: E402

MODEL = "LiuZH-19/SongGen_mixed_pro"


def build_request(
    lyrics: str,
    description: str = "a pop song",
    ref_voice_path: str | None = None,
    seed: int | None = None,
) -> dict:
    """Build an Omni request payload for SongGen.

    Args:
        lyrics: Song lyrics (English only).
        description: Music style / genre description text.
        ref_voice_path: Optional path to reference voice WAV for voice
            conditioning. Provide pre-separated vocals (Demucs is skipped).
        seed: Optional random seed for reproducible generation.
    """
    additional: dict = {
        "lyrics": [lyrics],
        "text_description": [description],
    }
    if ref_voice_path is not None:
        # Load the reference voice and pass as an array so the model
        # code can write it to a temp file without re-reading from disk.
        import soundfile as sf

        wav_data, sr_in = sf.read(str(ref_voice_path), dtype="float32", always_2d=False)
        additional["ref_voice_array"] = [[wav_data.tolist(), sr_in]]
    if seed is not None:
        additional["seed"] = [seed]

    return {
        "prompt": "<|im_start|>assistant\n",
        "additional_information": additional,
    }


def save_audio(waveform: torch.Tensor, path: str, sample_rate: int = 16000) -> None:
    audio_np = waveform.float().numpy()
    sf.write(path, audio_np, sample_rate)
    print(f"  Saved {path} ({audio_np.shape}, {sample_rate} Hz)")


def main(args) -> None:
    omni = Omni(
        model=MODEL,
        deploy_config=args.deploy_config,
        stage_init_timeout=args.stage_init_timeout,
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        max_tokens=4096,
        seed=args.seed if args.seed is not None else 42,
        detokenize=False,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lyrics:      {args.lyrics!r}")
    print(f"Description: {args.description!r}")
    if args.ref_voice:
        print(f"Ref voice:   {args.ref_voice}")

    inputs = build_request(
        lyrics=args.lyrics,
        description=args.description,
        ref_voice_path=args.ref_voice,
        seed=args.seed,
    )

    for stage_outputs in omni.generate(inputs, sampling_params):
        for i, req_output in enumerate(stage_outputs.request_output):
            for j, out in enumerate(req_output.outputs):
                mm = out.multimodal_output
                if mm is None:
                    print(f"  [req {i}] No audio output.")
                    continue
                audio = mm.get("audio")
                sr_tensor = mm.get("sr")
                if audio is None:
                    print(f"  [req {i}] No waveform in multimodal_output.")
                    continue
                sr = int(sr_tensor.item()) if sr_tensor is not None else 16000
                out_path = str(output_dir / f"output_{i}_{j}.wav")
                save_audio(audio.cpu(), out_path, sr)

    print("Done.")


def parse_args():
    parser = TrackingArgumentParser(description="SongGen offline inference")
    parser.add_argument(
        "--lyrics",
        default="Under the moonlight, we dance through the night, stars above shining bright.",
        help="Song lyrics to synthesize (English only).",
    )
    parser.add_argument(
        "--description",
        default="dreamy pop ballad with piano and strings, female vocals",
        help="Music style / genre description text.",
    )
    parser.add_argument(
        "--ref-voice",
        default=None,
        help="Path to reference voice audio for voice timbre conditioning (optional).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")),
            "songgen_output",
        ),
        help="Directory for WAV outputs (default: ~/.cache/songgen_output).",
    )
    parser.add_argument(
        "--deploy-config",
        default=None,
        help="Path to a deploy YAML; leave unset to auto-load vllm_omni/deploy/songgen.yaml.",
    )
    parser.add_argument("--stage-init-timeout", type=int, default=180)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
