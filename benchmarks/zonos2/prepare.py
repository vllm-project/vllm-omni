# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Normalize once and freeze identical prompt frames for both backends."""

import argparse
import hashlib
import json
from pathlib import Path

import soundfile as sf
import torch

from benchmarks.zonos2.protocol import CASES, PARAMS, THRESHOLDS
from vllm_omni.model_executor.models.zonos2.configuration_zonos2 import Zonos2Config
from vllm_omni.model_executor.models.zonos2.zonos2_processor import Zonos2Processor
from vllm_omni.model_executor.models.zonos2.zonos2_speaker import Zonos2SpeakerEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    processor = Zonos2Processor(Zonos2Config.from_pretrained(args.model, local_files_only=True))
    samples, rate = sf.read(args.reference, dtype="float32", always_2d=True)
    speaker = Zonos2SpeakerEncoder().encode(samples.T, rate)
    rows = []
    for key, text, language, clone in CASES:
        base = processor.build(text, language=language)
        full = processor.build(text, language=language, speaker_embedding=speaker if clone else None)
        item = {
            "id": key,
            "text": text,
            "language": language,
            "normalized_text": full.normalized_text,
            "base_frames": base.frames,
            "frames": full.frames,
            "speaker_embedding": speaker if clone else None,
        }
        path = args.out / f"{key}.pt"
        torch.save(item, path)
        rows.append(
            {
                "id": key,
                "text": text,
                "language": language,
                "truth": full.normalized_text,
                "clone": clone,
                "bundle_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "cases": rows,
        "params": PARAMS,
        "thresholds": THRESHOLDS,
        "reference_audio": str(Path(args.reference).resolve()),
        "reference_sha256": hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
        "input_contract": "TN once; UTF8 frames identical, official prepends matching speaker markers",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
