"""Pre-compute VoxCPM2 custom voice profiles.

The generated directory can be passed to the server via
``custom_voice_dir`` in ``vllm_omni/deploy/voxcpm2.yaml``. Requests can then
use ``/v1/audio/speech`` with ``voice="<name>"`` and no per-request ref_audio.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni.utils.custom_voice_io import safe_voice_stem  # noqa: E402

MANIFEST_NAME = "custom_voice_manifest.json"


def _load_tts(model: str, device: torch.device):
    from vllm_omni.model_executor.models.voxcpm2.voxcpm2_import_utils import import_voxcpm2_core

    VoxCPM = import_voxcpm2_core()
    native = VoxCPM.from_pretrained(model, load_denoiser=False, optimize=False)
    return native.tts_model.to(device).eval()


def _load_manifest(output_dir: Path, model: str) -> dict[str, Any]:
    path = output_dir / MANIFEST_NAME
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "model_type": "voxcpm2",
        "model": model,
        "voices": {},
    }


def _write_voice(
    *,
    model: str,
    output_dir: Path,
    voice_name: str,
    ref_audio: str,
    prompt_text: str | None,
    mode: str,
    speaker_description: str | None,
    device: torch.device,
) -> None:
    if mode in ("continuation", "ref_continuation") and not prompt_text:
        raise ValueError("--prompt-text is required for continuation/ref_continuation modes")

    tts = _load_tts(model, device)
    tensors: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        if mode in ("reference", "ref_continuation"):
            tensors["ref_audio_feat"] = tts._encode_wav(ref_audio, padding_mode="right").float().cpu().contiguous()
        if mode in ("continuation", "ref_continuation"):
            tensors["audio_feat"] = tts._encode_wav(ref_audio, padding_mode="left").float().cpu().contiguous()

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{safe_voice_stem(voice_name)}.safetensors"
    save_file(tensors, str(output_dir / filename))

    manifest = _load_manifest(output_dir, model)
    entry: dict[str, Any] = {
        "name": voice_name,
        "file": filename,
        "mode": mode,
    }
    if "ref_audio_feat" in tensors:
        entry["ref_audio_feat_len"] = int(tensors["ref_audio_feat"].shape[0])
    if "audio_feat" in tensors:
        entry["audio_feat_len"] = int(tensors["audio_feat"].shape[0])
    if prompt_text:
        entry["prompt_text"] = prompt_text
    if speaker_description:
        entry["speaker_description"] = speaker_description

    manifest.setdefault("voices", {})[voice_name] = entry
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output_dir / filename}")
    print(f"Updated {output_dir / MANIFEST_NAME}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute VoxCPM2 custom voice profile")
    parser.add_argument("--model", default="openbmb/VoxCPM2", help="VoxCPM2 model path or Hugging Face ID")
    parser.add_argument("--voice-name", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument(
        "--prompt-text",
        default=None,
        help="Transcript of ref audio for continuation/ref_continuation modes",
    )
    parser.add_argument(
        "--mode",
        choices=["reference", "continuation", "ref_continuation"],
        default="reference",
    )
    parser.add_argument("--speaker-description", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    _write_voice(
        model=args.model,
        output_dir=Path(args.output_dir),
        voice_name=args.voice_name,
        ref_audio=args.ref_audio,
        prompt_text=args.prompt_text,
        mode=args.mode,
        speaker_description=args.speaker_description,
        device=torch.device(args.device),
    )


if __name__ == "__main__":
    main()
