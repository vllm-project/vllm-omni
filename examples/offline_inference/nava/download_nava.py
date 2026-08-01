# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Download and assemble a local NAVA directory for vLLM-Omni."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

MODEL_INDEX = {
    "_class_name": "NAVAPipeline",
    "nava_ckpt": "NAVA.safetensors",
    "config": "configs/nava.yaml",
    "wan_dir": "Wan2.2-TI2V-5B",
    "audio_vae_dir": "params",
    "speaker_dir": "speaker",
}

SPEAKER_REPO = "https://github.com/IDRnD/ReDimNet.git"
REQUIRED_DIRS = (
    "Wan2.2-TI2V-5B/google/umt5-xxl",
    "params/LTX2",
)
REQUIRED_FILES = (
    "NAVA.safetensors",
    "Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
    "params/LTX2/ltx-2.3-22b-dev_audio_vae.safetensors",
    "speaker/hubconf.py",
)
CONFIG_PATHS = ("configs/nava.yaml", "nava.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download baidu/NAVA and write model_index.json.")
    parser.add_argument("--repo-id", default="baidu/NAVA", help="NAVA Hugging Face repository.")
    parser.add_argument("--local-dir", required=True, help="Target local model directory.")
    parser.add_argument("--verify-only", action="store_true", help="Validate an existing local directory.")
    parser.add_argument(
        "--prepare-speaker-dir",
        action="store_true",
        help="Deprecated; speaker assets are prepared by default.",
    )
    parser.add_argument(
        "--speaker-repo", default=SPEAKER_REPO, help="ReDimNet git repository for local speaker assets."
    )
    return parser.parse_args()


def build_model_index(local_dir: Path) -> dict[str, str]:
    model_index = dict(MODEL_INDEX)
    if not (local_dir / model_index["config"]).exists() and (local_dir / "nava.yaml").exists():
        model_index["config"] = "nava.yaml"
    return model_index


def verify_model_dir(local_dir: Path) -> None:
    missing = [name for name in REQUIRED_DIRS if not (local_dir / name).exists()]
    if not any((local_dir / name).exists() for name in CONFIG_PATHS):
        missing.append("configs/nava.yaml or nava.yaml")
    missing.extend(name for name in REQUIRED_FILES if not (local_dir / name).exists())
    if missing:
        raise SystemExit(f"NAVA model directory is incomplete: missing {', '.join(missing)}.")


def prepare_speaker_dir(local_dir: Path, speaker_repo: str) -> None:
    speaker_dir = local_dir / MODEL_INDEX["speaker_dir"]
    if (speaker_dir / "hubconf.py").exists():
        return
    if speaker_dir.exists() and any(speaker_dir.iterdir()):
        raise SystemExit(f"NAVA speaker directory exists but is missing hubconf.py: {speaker_dir}")
    subprocess.run(["git", "clone", "--depth", "1", speaker_repo, str(speaker_dir)], check=True)


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    if not args.verify_only:
        subprocess.run(["huggingface-cli", "download", args.repo_id, "--local-dir", str(local_dir)], check=True)
        prepare_speaker_dir(local_dir, args.speaker_repo)
        model_index_path = local_dir / "model_index.json"
        model_index_path.write_text(json.dumps(build_model_index(local_dir), indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {model_index_path}")

    if args.prepare_speaker_dir:
        prepare_speaker_dir(local_dir, args.speaker_repo)

    verify_model_dir(local_dir)
    print(f"Verified {local_dir}")


if __name__ == "__main__":
    main()
