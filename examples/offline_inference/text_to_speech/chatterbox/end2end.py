# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Generate a complete 24 kHz WAV with the serving model's prompt builder."""

import argparse
import logging
from pathlib import Path

import soundfile as sf
import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from vllm_omni import Omni
from vllm_omni.model_executor.models.chatterbox.conditioning import (
    VoiceConditioner,
    VoiceConditioning,
    build_prompt,
    punc_norm,
)
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig
from vllm_omni.transformers_utils.repo_utils import hf_api


def main():
    """Build one conditioned prompt and save its consolidated audio."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("text")
    parser.add_argument("output", type=Path)
    parser.add_argument("--variant", choices=("turbo", "original"), default="turbo")
    parser.add_argument("--ref-audio", type=Path)
    args = parser.parse_args()
    config = ChatterboxConfig(args.variant)
    text = punc_norm(args.text, args.variant)
    if args.variant == "turbo":
        tokenizer = AutoTokenizer.from_pretrained(args.model, tokenizer_type="gpt2")
        ids = tokenizer.encode(text, add_special_tokens=False)
    else:
        tokenizer = Tokenizer.from_file(hf_api().hf_hub_download(args.model, "tokenizer.json"))
        ids = tokenizer.encode(text.replace(" ", "[SPACE]")).ids
    if args.ref_audio is None:
        voice = VoiceConditioning.from_builtin(args.model)
    else:
        wav, rate = sf.read(args.ref_audio, dtype="float32", always_2d=True)
        voice = VoiceConditioner(args.model, config, torch.device("cpu")).prepare(wav.mean(axis=1), rate)
    prompt = build_prompt(ids, voice, config)
    engine = Omni(model=args.model)
    try:
        output = list(engine.generate([prompt]))[0].outputs[0].multimodal_output
        audio = output["audio"]
        if isinstance(audio, list):
            audio = torch.cat([chunk.flatten() for chunk in audio])
        if audio.numel() == 0 or not torch.isfinite(audio).all():
            raise ValueError("Model returned empty or nonfinite audio")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.output, audio.flatten().numpy(force=True), config.sample_rate)
        logging.info("Saved %s at %d Hz", args.output, config.sample_rate)
    finally:
        engine.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
