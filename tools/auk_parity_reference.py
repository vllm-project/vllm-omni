#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Generate the upstream AuK reference artifacts consumed by the parity tests.

This runs the *upstream* implementation, so it must be executed in an
environment where the upstream package is installed
(``pip install -e /path/to/AuK``; it pins its own torch/transformers line and
does not need vLLM-Omni). It writes:

    <out>/parity_ref/<variant>/<case>.pt, <case>.wav, meta.json
        Upstream generations for four cookbook cases (zero-shot TTS, instruct
        TTS, content edit, denoise) with the reference latents, the generated
        latents and the waveform. Point ``AUK_PARITY_REF`` at
        ``<out>/parity_ref/<variant>`` for tests/diffusion/models/auk/test_pipeline_auk.py.
    <out>/fusion_ref/zs_tts_audio.pt, instruct_text_only.pt
        The fp32 layer-fused text condition from the HF thinker for the
        zero-shot and text-only prompts (``AUK_FUSION_REF``).

By default the VAE encode uses the posterior mean, matching the port's default
and making the artifacts deterministic; ``--vae-sample`` reproduces the
upstream stochastic draw (seeded through ``--global-seed``). The ODE noise is
drawn from ``--seed`` on both sides.

Usage:
    python tools/auk_parity_reference.py \\
        --auk-repo /path/to/AuK --ckpt-dir /path/to/AuK \\
        --qwen-dir /path/to/Qwen2.5-Omni-3B --out /path/to/auk-parity \\
        [--variant base|flash] [--vae-sample] [--fusion-device cpu|cuda] [--skip-fusion]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

WEIGHTS = {"base": "auk_base.safetensors", "flash": "auk_flash.safetensors"}

# Cookbook cases; audio paths are relative to <auk-repo>/assets/demo-input-audio.
CASES: dict[str, dict[str, Any]] = {
    "zs_tts_en": {
        "text": (
            "Say the following with the same voice: 'Ladies and gentlemen, it's an honor to have the "
            "opportunity to address such a distinguished audience'"
        ),
        "audio": "zero-shot-tts/ref.wav",
        "gen_seconds": 6.0,
    },
    "instruct_en": {
        "text": (
            'Generate speech based on the following description: "A calm young woman speaking warmly and slowly.". '
            'The content to speak is: "Welcome back, how was your day at work?".'
        ),
        "audio": None,
        "gen_seconds": 3.5,
    },
    "content_edit": {
        "text": "Replace 'but accepting what we cannot have' with 'and living well with dreams unmet'.",
        "audio": "content-edit/content.wav",
        "gen_seconds": 7.0,
    },
    "denoise": {
        "text": "Keep pure speech voice, remove noise and reverberation.",
        "audio": "vocal-extraction/vocal-1-input.wav",
        "gen_seconds": None,
    },
}
# The two prompts the fusion reference covers (they must match the cases above).
FUSION_CASES = {"zs_tts_audio": "zs_tts_en", "instruct_text_only": "instruct_en"}
NO_PROMPT_AUDIO = "|<no_prompt_audio>|"


def _messages(case: dict[str, Any], assets: Path) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": case["text"]}]
    if case["audio"]:
        content.append({"type": "audio", "audio": str(assets / case["audio"])})
    return [{"role": "user", "content": content}]


def _mean_encode(vae):
    """Posterior-mean version of the upstream ``encoding_and_normalization``."""

    def encode(sample, sample_lengths=None):
        latent_stats = vae.audio_encoder(sample)
        if sample_lengths is None:
            sample_lengths = torch.LongTensor([sample.size(-1)] * sample.size(0)).to(sample.device)
        latent_lens = sample_lengths // vae.hop_size
        mean, _log_std = latent_stats.chunk(2, 1)
        latents = mean.transpose(1, 2).float()
        latents = (latents - vae.global_mean.float()) / torch.sqrt(vae.global_log_std.float())
        return latents, torch.clamp(latent_lens, max=latents.size(1))

    return encode


def generate_references(args: argparse.Namespace) -> None:
    from auk.infer.infer_auk import AukInfer, save_audio

    ckpt = Path(args.ckpt_dir)
    out = Path(args.out) / "parity_ref" / args.variant
    out.mkdir(parents=True, exist_ok=True)
    assets = Path(args.auk_repo) / "assets" / "demo-input-audio"

    engine = AukInfer(str(ckpt / "config.yaml"), str(ckpt / WEIGHTS[args.variant]), qwen_path=args.qwen_dir)
    encode = engine.vae_model.encoding_and_normalization if args.vae_sample else _mean_encode(engine.vae_model)
    recorded: dict[str, torch.Tensor] = {}

    def encode_hook(sample, sample_lengths=None):
        latents, lens = encode(sample, sample_lengths)
        recorded["ref_latents"] = latents.detach().float().cpu()
        recorded["ref_latent_lens"] = lens.detach().cpu()
        return latents, lens

    engine.vae_model.encoding_and_normalization = encode_hook
    upstream_sample = engine.model.sample

    def sample_hook(*a, **k):
        latents, trajectory = upstream_sample(*a, **k)
        recorded["generated"] = latents.detach().float().cpu()
        return latents, trajectory

    engine.model.sample = sample_hook

    meta: dict[str, Any] = {}
    for name, case in CASES.items():
        messages = _messages(case, assets)
        recorded.clear()
        torch.manual_seed(args.global_seed)
        started = time.perf_counter()
        audio, sr = engine.generate(messages, gen_seconds=case["gen_seconds"], seed=args.seed)
        wall = time.perf_counter() - started
        save_audio(audio, sr, str(out / f"{name}.wav"))
        torch.save(
            {
                "messages": messages,
                "gen_seconds": case["gen_seconds"],
                "seed": args.seed,
                "global_seed": args.global_seed,
                "vae_sample": bool(args.vae_sample),
                "audio": audio.float().cpu(),
                "sr": sr,
                **recorded,
            },
            out / f"{name}.pt",
        )
        ref_frames = int(recorded["ref_latent_lens"][0]) if "ref_latent_lens" in recorded else 0
        meta[name] = {
            "wall_s": wall,
            "out_s": audio.shape[-1] / sr,
            "sr": sr,
            "ref_frames": ref_frames,
            "gen_frames": int(recorded["generated"].shape[1]) - ref_frames,
        }
        print(name, meta[name], flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")


def generate_fusion(args: argparse.Namespace) -> None:
    from auk.model.cfm_edit import CFMEdit
    from safetensors import safe_open
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

    out = Path(args.out) / "fusion_ref"
    out.mkdir(parents=True, exist_ok=True)
    assets = Path(args.auk_repo) / "assets" / "demo-input-audio"
    with safe_open(str(Path(args.ckpt_dir) / WEIGHTS[args.variant]), "pt") as f:
        layer_weights = f.get_tensor("layer_weights").float()
        layer_scale = f.get_tensor("layer_scale").float()

    processor = Qwen2_5OmniProcessor.from_pretrained(args.qwen_dir)
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(args.qwen_dir, torch_dtype=torch.float32)
    thinker.visual = None
    thinker = thinker.to(args.fusion_device).eval()

    for out_name, case_name in FUSION_CASES.items():
        case = dict(CASES[case_name])
        if not case["audio"] and not case["text"].endswith(NO_PROMPT_AUDIO):
            case["text"] = case["text"] + NO_PROMPT_AUDIO
        cond = CFMEdit.build_cond_inputs([_messages(case, assets)], processor).to(args.fusion_device)
        with torch.no_grad():
            hidden_states = thinker(**cond, output_hidden_states=True).hidden_states
        width = hidden_states[0].shape[-1]
        stacked = torch.stack([F.layer_norm(h.float(), [width]) for h in hidden_states[1:]], 0)
        weights = F.softmax(layer_weights.to(stacked.device), 0)
        fused = (stacked * weights[:, None, None, None]).sum(0) * layer_scale.to(stacked.device)
        torch.save(
            {
                "input_ids": cond["input_ids"].cpu(),
                "attention_mask": cond["attention_mask"].cpu(),
                "fused": fused[0].float().cpu(),
                "final_normed": hidden_states[-1][0].float().cpu(),
                "text": case["text"],
            },
            out / f"{out_name}.pt",
        )
        print(out_name, {"tokens": int(cond["input_ids"].shape[1]), "fused_norm": float(fused.norm())}, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--auk-repo", required=True, help="checkout of the upstream AuK repository (for demo clips)")
    parser.add_argument("--ckpt-dir", required=True, help="tencent/AuK or tencent/AuK-Flash snapshot")
    parser.add_argument("--qwen-dir", required=True, help="Qwen/Qwen2.5-Omni-3B snapshot")
    parser.add_argument("--out", required=True, help="output root; writes parity_ref/<variant> and fusion_ref")
    parser.add_argument("--variant", choices=sorted(WEIGHTS), default="base")
    parser.add_argument("--seed", type=int, default=0, help="ODE noise seed passed to the upstream sampler")
    parser.add_argument("--global-seed", type=int, default=1234, help="global RNG seed set before each generate()")
    parser.add_argument("--vae-sample", action="store_true", help="draw the VAE posterior sample as upstream does")
    parser.add_argument("--fusion-device", default="cpu", help="device for the fp32 HF thinker forward")
    parser.add_argument("--skip-fusion", action="store_true", help="only write the generation references")
    args = parser.parse_args()

    generate_references(args)
    if not args.skip_fusion:
        generate_fusion(args)
    print(f"reference artifacts written under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
