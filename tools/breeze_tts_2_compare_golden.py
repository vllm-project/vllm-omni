#!/usr/bin/env python3
"""Compare upstream and vLLM-Omni Breeze-TTS-2 golden artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import soundfile as sf
import torch

TEMPLATES = {
    "plain": "tts_plain",
    "instruction": "tts_instruction",
    "reference_edit": "ref_edit_tata",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audio_metrics(path: Path) -> dict[str, object]:
    data, sample_rate = sf.read(path, dtype="float32")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "samples": int(data.shape[0]),
        "sample_rate": int(sample_rate),
        "channels": int(1 if data.ndim == 1 else data.shape[1]),
        "duration_s": round(float(data.shape[0]) / sample_rate, 6),
        "finite": bool(torch.isfinite(torch.from_numpy(data)).all().item()),
        "peak": round(float(abs(data).max()), 6),
        "rms": round(float((data**2).mean() ** 0.5), 6),
    }


def _vector_metrics(upstream: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    difference = upstream.double() - actual.double()
    return {
        "mae": round(float(difference.abs().mean()), 8),
        "rmse": round(float(difference.pow(2).mean().sqrt()), 8),
        "max_abs_error": round(float(difference.abs().max()), 8),
        "relative_l2": round(float(difference.norm() / upstream.norm()), 8),
        "cosine": round(
            float(torch.dot(upstream.double(), actual.double()) / (upstream.norm() * actual.norm())),
            8,
        ),
    }


def _load_vllm_payload(vllm_dir: Path, template: str) -> dict[str, object]:
    matches = []
    for path in vllm_dir.glob("breeze_*.pt"):
        payload = torch.load(path, weights_only=False)
        if payload.get("template") == template:
            matches.append(payload)
    if len(matches) != 1:
        raise ValueError(f"expected one vLLM dump for {template!r}, found {len(matches)}")
    return matches[0]


def _longest_exact_frame_prefix(upstream: torch.Tensor, actual: torch.Tensor) -> int:
    count = min(int(upstream.shape[0]), int(actual.shape[0]))
    equal = (upstream[:count] == actual[:count]).all(dim=-1)
    prefix = 0
    for is_equal in equal.tolist():
        if not is_equal:
            break
        prefix += 1
    return prefix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--vllm-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plain-audio", type=Path)
    parser.add_argument("--instruction-audio", type=Path)
    parser.add_argument("--reference-audio", type=Path)
    args = parser.parse_args()

    audio_paths = {
        "plain": args.plain_audio,
        "instruction": args.instruction_audio,
        "reference_edit": args.reference_audio,
    }
    report: dict[str, object] = {
        "upstream_dir": str(args.upstream_dir.resolve()),
        "vllm_dir": str(args.vllm_dir.resolve()),
        "cases": [],
    }
    for name, template in TEMPLATES.items():
        upstream_codes = torch.load(args.upstream_dir / name / "codes.pt", weights_only=False)
        upstream_prompt = torch.load(args.upstream_dir / name / "prompt_ids.pt", weights_only=False)
        actual_payload = _load_vllm_payload(args.vllm_dir, template)
        actual_codes = actual_payload["codes"]
        actual_prompt = actual_payload["prompt_ids"]
        common_frames = min(int(upstream_codes.shape[0]), int(actual_codes.shape[0]))
        code_equal = upstream_codes[:common_frames] == actual_codes[:common_frames]
        frame_equal = code_equal.all(dim=-1)
        prefill_metrics = None
        upstream_hidden_path = args.upstream_dir / name / "prefill_hidden.pt"
        if upstream_hidden_path.exists():
            prefill_files = [
                path
                for path in args.vllm_dir.glob("prefill_*.pt")
                if torch.load(path, weights_only=False).get("template") == template
            ]
            if len(prefill_files) == 1:
                actual_hidden = torch.load(prefill_files[0], weights_only=False)["hidden"]
                prefill_metrics = _vector_metrics(
                    torch.load(upstream_hidden_path, weights_only=False),
                    actual_hidden,
                )
        entry: dict[str, object] = {
            "name": name,
            "template": template,
            "prompt": {
                "upstream_tokens": int(upstream_prompt.numel()),
                "vllm_tokens": int(actual_prompt.numel()),
                "exact": bool(torch.equal(upstream_prompt, actual_prompt)),
            },
            "frames": {
                "upstream": int(upstream_codes.shape[0]),
                "vllm": int(actual_codes.shape[0]),
                "codebooks": int(upstream_codes.shape[-1]),
                "exact_sequence": bool(torch.equal(upstream_codes, actual_codes)),
                "common_frames": common_frames,
                "exact_common_frames": int(frame_equal.sum()),
                "longest_exact_prefix": _longest_exact_frame_prefix(upstream_codes, actual_codes),
                "exact_codes": int(code_equal.sum()),
                "total_common_codes": int(code_equal.numel()),
                "code_accuracy": round(float(code_equal.float().mean()), 8),
                "codebook0_accuracy": round(
                    float((upstream_codes[:common_frames, 0] == actual_codes[:common_frames, 0]).float().mean()),
                    8,
                ),
                "range_valid": bool(
                    upstream_codes.min() >= 0
                    and upstream_codes.max() < 2048
                    and actual_codes.min() >= 0
                    and actual_codes.max() < 2048
                ),
            },
            "prefill_hidden": prefill_metrics,
        }
        if audio_paths[name] is not None and Path(audio_paths[name]).exists():  # type: ignore[arg-type]
            entry["audio"] = _audio_metrics(Path(audio_paths[name]))  # type: ignore[arg-type]
        report["cases"].append(entry)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
