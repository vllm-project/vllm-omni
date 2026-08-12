# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline inference example for Gepard-1.0 TTS via vLLM-Omni.

Single-stage native-AR pipeline: a Qwen3.5 backbone samples one 32-code FSQ
frame per step; the NeMo NanoCodec decodes the frames to a 22.05 kHz mono
waveform. PR1 is zero-shot (default learned voice) — no reference audio.

Generation length and seed are stage settings, not flags here: one output
token is one audio frame, so ``max_tokens`` in the deploy YAML is the frame
budget, and ``seed`` there makes the in-model 32-head sampling reproducible.
Pass a copy of the YAML via ``--deploy-config`` to change either — a
SamplingParams object handed to ``generate()`` would replace the stage
defaults wholesale rather than merge, dropping the pipeline's stop token.

Usage:
  python end2end.py --text "Hello, this is Gepard speaking."
"""

from __future__ import annotations

import os
from pathlib import Path

import soundfile as sf
import torch

import vllm_omni
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm_omni import Omni  # noqa: E402

MODEL = "nineninesix/gepard-1.0"
SAMPLE_RATE = 22050
# Absolute: the loader hands the path straight to OmegaConf, which resolves a
# bare name against the current working directory, not the package.
DEFAULT_DEPLOY_CONFIG = str(Path(vllm_omni.__file__).resolve().parent / "deploy" / "gepard.yaml")


def build_request(text: str) -> dict:
    """Build an Omni request payload for Gepard (zero-shot, no ref audio).

    ``preprocess`` only consumes the [speaker slots, SOT, text, EOT, SOS]
    layout; nothing on the offline path builds it, so a bare text prompt would
    have no speaker slots and no SOS.
    """
    from transformers import AutoTokenizer

    from vllm_omni.model_executor.models.gepard.configuration_gepard import GepardConfig
    from vllm_omni.model_executor.models.gepard.prompt import build_gepard_prompt_ids

    # From the checkpoint's own gepard_config.json, which is what the worker
    # builds its config from too. The class defaults match today's checkpoint,
    # but reading them here would make this the second source of truth for the
    # layout — and a layout that disagrees with the one the model was trained
    # on does not fail, it just degrades the speech.
    cfg = GepardConfig.from_checkpoint(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt_token_ids = build_gepard_prompt_ids(
        tokenizer(text, add_special_tokens=False)["input_ids"],
        config=cfg,
    )
    return {
        "prompt_token_ids": prompt_token_ids,
        "additional_information": {"text": [text]},
    }


def save_audio(waveform: torch.Tensor, path: str, sample_rate: int = SAMPLE_RATE) -> None:
    sf.write(path, waveform.float().numpy(), sample_rate)
    seconds = waveform.numel() / sample_rate
    print(
        f"  Saved {path} ({tuple(waveform.shape)}, {sample_rate} Hz, "
        f"{seconds:.2f}s, peak {float(waveform.abs().max()):.3f})"
    )


def main(args) -> None:
    omni = Omni(
        model=MODEL,
        deploy_config=args.deploy_config,
        stage_init_timeout=args.stage_init_timeout,
        init_timeout=args.init_timeout,
    )

    # No explicit SamplingParams: a caller-supplied object replaces the stage
    # defaults rather than merging over them, which would drop the pipeline's
    # stop_token_ids and leave the request running to max_tokens.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Synthesizing: {args.text!r}")
    inputs = build_request(text=args.text)

    for i, stage_outputs in enumerate(omni.generate(inputs)):
        # OmniRequestOutput.request_output is a single RequestOutput, not a list.
        req_output = stage_outputs.request_output
        if req_output is not None:
            for j, out in enumerate(req_output.outputs):
                mm = out.multimodal_output
                if mm is None:
                    print(f"  [req {i}] No audio output.")
                    continue
                # The consolidation path renames "model_outputs" to "audio";
                # accept either. Explicit None checks — a multi-element tensor
                # has no truth value.
                audio = mm.get("audio")
                if audio is None:
                    audio = mm.get("model_outputs")
                if isinstance(audio, list):
                    audio = audio[0] if len(audio) else None
                if audio is None:
                    print(f"  [req {i}] No waveform in multimodal_output (keys: {list(mm)}).")
                    continue
                sr_tensor = mm.get("sr")
                sr = int(sr_tensor.item()) if hasattr(sr_tensor, "item") else SAMPLE_RATE
                save_audio(audio.cpu(), str(output_dir / f"output_{i}_{j}.wav"), sr)

    print("Done.")


def parse_args():
    parser = TrackingArgumentParser(description="Gepard-1.0 offline TTS inference (zero-shot)")
    parser.add_argument("--text", default="Hello, this is Gepard speaking.", help="Text to synthesize.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")),
            "gepard_output",
        ),
        help="Directory for WAV outputs (default: ~/.cache/gepard_output).",
    )
    parser.add_argument(
        "--deploy-config",
        default=DEFAULT_DEPLOY_CONFIG,
        help=(
            "Path to the deploy YAML (default: the packaged vllm_omni/deploy/gepard.yaml). This must "
            "stay set — the checkpoint self-identifies as qwen3_5_text, so without the YAML's "
            "`pipeline: gepard` pin the architectures fallback routes Qwen3_5ForCausalLM to the "
            "diffusion registry. Copy it to change max_tokens (the frame budget) or seed."
        ),
    )
    # The upstream defaults are too tight for a cold start (backbone + codec
    # load, then profile/KV-cache/warmup).
    parser.add_argument("--stage-init-timeout", type=int, default=900)
    parser.add_argument("--init-timeout", type=int, default=1800)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
