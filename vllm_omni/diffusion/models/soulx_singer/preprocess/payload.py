"""SoulX-Singer payload schema (guidance/precomputed tables, dummy builders)."""

from __future__ import annotations

from typing import Any

import torch

SOULX_PRECOMPUTED_KEYS_BY_KIND: dict[str, tuple[str, ...]] = {
    "svs": ("prompt_metadata_path", "target_metadata_path", "audio_path"),
    "svc": ("prompt_wav_path", "target_wav_path", "prompt_f0_path", "target_f0_path"),
}


# Key under which the SoulX preprocess payload is stored in
# prompt["additional_information"] or DiffusionOutput.custom_output.
SOULX_PREPROCESSED_KEY = "soulx_preprocessed"


def get_soulx_preprocessed_payload(prompt: dict[str, Any]) -> dict[str, Any] | None:
    """Extract attached payload from a diffusion request prompt dict."""
    additional = prompt.get("additional_information") or {}
    payload = additional.get(SOULX_PREPROCESSED_KEY)
    return payload if isinstance(payload, dict) else None


def relocate_tensors(payload: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    """Recursively move all tensors inside the payload to the target device.

    Called after decode when the payload is about to be consumed by the
    diffusion model (which may live on a different device / rank).
    """
    if isinstance(payload, torch.Tensor):
        return payload.to(device, non_blocking=True)
    if isinstance(payload, dict):
        return {key: relocate_tensors(value, device) for key, value in payload.items()}
    if isinstance(payload, list):
        return [relocate_tensors(item, device) for item in payload]
    return payload


def consume_payload(
    req: Any,
    expected_kind: str,
    device: torch.device | str,
) -> dict[str, Any]:
    """Read the payload attached by pre_process_func, validate the kind,
    and relocate tensors to the compute device.
    """
    prompts = getattr(req, "prompts", None) or []
    if not prompts:
        raise ValueError(f"SoulX-Singer {expected_kind} request has no prompts.")
    prompt = prompts[0]
    if isinstance(prompt, str):
        raise ValueError(f"SoulX-Singer {expected_kind} forward requires pre_process_func output on the prompt.")
    payload = get_soulx_preprocessed_payload(prompt)
    if payload is None or payload.get("kind") != expected_kind:
        raise ValueError(
            f"SoulX-Singer {expected_kind} forward requires additional_information['soulx_preprocessed'] "
            "produced by pre_process_func."
        )
    return relocate_tensors(payload, device)


def has_precomputed(extra_args: dict[str, Any], kind: str) -> bool:
    keys = SOULX_PRECOMPUTED_KEYS_BY_KIND[kind]
    return all(extra_args.get(key) for key in keys)


def build_dummy_payload(kind: str, device: torch.device) -> dict[str, Any]:
    mel_frames = 4
    hop_size = 480
    n_fft = 1920
    wav_samples = max(mel_frames * hop_size, n_fft)
    if kind == "svc":
        voiced_f0 = torch.full((1, mel_frames), 100.0, device=device, dtype=torch.float64)
        return {
            "kind": "svc",
            "prompt_wav": torch.zeros(1, wav_samples, device=device, dtype=torch.float32),
            "target_wav": torch.zeros(1, wav_samples, device=device, dtype=torch.float32),
            "prompt_f0": voiced_f0.clone(),
            "target_f0": voiced_f0.clone(),
        }
    return {
        "kind": "svs",
        "prompt_meta": {
            "phoneme": torch.zeros(1, 2, device=device, dtype=torch.long),
            "note_pitch": torch.zeros(1, mel_frames, device=device, dtype=torch.long),
            "note_type": torch.zeros(1, mel_frames, device=device, dtype=torch.long),
            "mel2note": torch.zeros(1, mel_frames, device=device, dtype=torch.long),
            "f0": torch.zeros(1, mel_frames, device=device, dtype=torch.float32),
            "wav": torch.zeros(1, wav_samples, device=device, dtype=torch.float32),
        },
        "target_meta_list": [
            {
                "text": "dummy",
                "phoneme": " ".join(["<SP>"] * mel_frames),
                "f0": " ".join(["100"] * mel_frames),
                "language": "Mandarin",
                "note_pitch": " ".join(["0"] * mel_frames),
                "note_type": " ".join(["1"] * mel_frames),
                "duration": " ".join(["1"] * mel_frames),
                "time": [0, max(int(wav_samples / 24000 * 1000), 100)],
            }
        ],
    }
