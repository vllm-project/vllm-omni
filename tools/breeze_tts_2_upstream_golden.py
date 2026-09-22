#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Generate deterministic Breeze-TTS-2 golden frames with the upstream runtime.

This tool is intentionally separate from vLLM-Omni: it imports the unmodified
``breezeblue-ai/breeze-tts`` reference implementation, fixes greedy decoding,
and writes both generated codec frames and decoded audio for offline parity
checks. It is a diagnostic tool, not a serving path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import soundfile as sf
import torch


def _plain_template(upstream_templates):
    """The upstream CLI does not register tts_plain; reuse its segment builder."""
    return SimpleNamespace(
        name="tts_plain",
        required_fields=("text",),
        build_segments=upstream_templates._tts_plain_segments,
        build_negative_segments=None,
        build_dual_branches=None,
    )


def _greedy(model) -> None:
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    depth = model.depth_decoder.generation_config
    depth.depth_decoder_do_sample = False
    depth.do_sample = False
    depth.temperature = None
    depth.top_p = None
    depth.top_k = None


def _valid_frames(sequences: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    frames = sequences[0].detach().cpu().to(torch.long)
    if frames.ndim == 1:
        frames = frames.unsqueeze(0)
    is_pad = (frames == pad_token_id).all(dim=-1)
    if bool(is_pad.any()):
        frames = frames[: int(torch.argmax(is_pad.int()).item())]
    return frames.contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    upstream_posix = str(args.upstream.resolve())
    if upstream_posix not in sys.path:
        sys.path.insert(0, upstream_posix)

    from breeze_infer.runtime import load_runtime, resolve_device
    from breeze_infer.templates import get_template, prepare_inputs

    device = resolve_device()
    # ``torch.manual_seed`` seeds CUDA RNGs as well, so no device-specific
    # seeding call is needed here.
    torch.manual_seed(0)
    tokenizer, model, audio_tokenizer = load_runtime(
        args.model,
        device=device,
        attn_implementation="eager",
    )
    # Do not enable Transformers 4.x's ``fix_mistral_regex`` compatibility
    # path: vLLM-Omni runs with Transformers 5.x, and direct tokenization checks
    # showed both runtimes produce the same token IDs without that patch.
    _greedy(model)
    model.eval()
    first_backbone_hidden: list[torch.Tensor] = []

    def _capture_first_backbone_hidden(_module, _inputs, output):
        if first_backbone_hidden:
            return
        hidden = getattr(output, "last_hidden_state", None)
        if isinstance(hidden, torch.Tensor):
            first_backbone_hidden.append(hidden[0, -1].detach().cpu().to(torch.float32).contiguous())

    model.backbone_model.register_forward_hook(_capture_first_backbone_hidden)

    cases = [
        {
            "name": "plain",
            "template": _plain_template(sys.modules["breeze_infer.templates"]),
            "request": {
                "id": "golden-plain",
                "text": "Golden plain alignment check.",
                "speaker": "S0",
            },
        },
        {
            "name": "instruction",
            "template": get_template("tts_instruction"),
            "request": {
                "id": "golden-instruction",
                "text": "Golden instruction alignment check.",
                "instruction": "Speak calmly, warmly, and clearly.",
                "speaker": "S0",
            },
        },
    ]
    if args.reference_audio is not None:
        cases.append(
            {
                "name": "reference_edit",
                "template": get_template("ref_edit_tata"),
                "request": {
                    "id": "golden-reference-edit",
                    "text": "Golden reference alignment check.",
                    "instruction": "Speak calmly and clearly.",
                    "ref_audio_path": str(args.reference_audio),
                    "ref_text": "Hello from Breeze.",
                    "speaker": "S0",
                },
            }
        )

    manifest = {
        "runtime": "upstream breezeblue-ai/breeze-tts",
        "model": str(args.model.resolve()),
        "upstream": upstream_posix,
        "device": device,
        "seed": 0,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "cfg_scale": 1.0,
        "cases": [],
    }
    for case in cases:
        first_backbone_hidden.clear()
        inputs = prepare_inputs(
            tokenizer,
            audio_tokenizer,
            model,
            [case["request"]],
            case["template"],
            guidance_scale=1.0,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_audio=True,
            audio_tokenizer=audio_tokenizer,
        )
        frames = _valid_frames(output.sequences, model.config.codebook_pad_token_id)
        audio = output.audio[0].detach().cpu().to(torch.float32).reshape(-1)
        case_dir = args.output_dir / case["name"]
        case_dir.mkdir(parents=True, exist_ok=True)
        torch.save(frames, case_dir / "codes.pt")
        prompt_ids = inputs["input_ids"].detach().cpu().to(torch.long).reshape(-1)
        torch.save(prompt_ids, case_dir / "prompt_ids.pt")
        if first_backbone_hidden:
            torch.save(first_backbone_hidden[0], case_dir / "prefill_hidden.pt")
        sf.write(case_dir / "upstream.wav", audio.numpy(), 24_000, subtype="PCM_16")
        entry = {
            "name": case["name"],
            "request": case["request"],
            "prompt_tokens": int(prompt_ids.numel()),
            "frames": int(frames.shape[0]),
            "codebooks": int(frames.shape[-1]) if frames.ndim == 2 else 0,
            "min_code": int(frames.min().item()) if frames.numel() else None,
            "max_code": int(frames.max().item()) if frames.numel() else None,
            "audio_samples": int(audio.numel()),
            "audio_sha256": __import__("hashlib").sha256((case_dir / "upstream.wav").read_bytes()).hexdigest(),
        }
        manifest["cases"].append(entry)
        print(json.dumps(entry), flush=True)

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
