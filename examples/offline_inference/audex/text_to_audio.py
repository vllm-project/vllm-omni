# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline Audex (Nemotron-Labs-Audex-2B) text-to-audio inference example.

Caption → general audio through the vLLM-Omni TTA pipeline: the audiogen
thinker generates interleaved 4-codebook <audiocodec_N> RVQ tokens under an
RVQ phase mask, and the external XCodec1 checkpoint decodes them to a 16 kHz
mono WAV.

(For DIFFUSION text-to-audio models — Stable Audio Open — use
``examples/offline_inference/text_to_audio/text_to_audio.py`` instead; Audex
TTA is a two-stage autoregressive pipeline with a different interface.)

Classifier-free guidance is effectively mandatory for TTA quality (official
default scale 3.0), so every request submits a cond/uncond pair; requests
run one at a time.

Pass the HF repo ROOT as --model. XCodec1 resolves from --xcodec1-path /
XCODEC1_PATH / the default hf-audio repo.

Example:

    python examples/offline_inference/audex/text_to_audio.py \\
        --captions "Heavy rain falling on a tin roof." \\
        --output-dir results/audex_tta_wavs
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from common import (
    SAMPLE_RATE,
    default_deploy_config,
    extract_pcm,
    guided_stage_params,
    load_audex_tokenizer,
    load_corpus,
    write_wav,
)

from vllm_omni import Omni
from vllm_omni.model_executor.models.audex.prompt import build_tta_cond_prompt, build_tta_null_prompt
from vllm_omni.model_executor.models.audex.tta import build_tta_phase_token_ids

DEFAULT_CAPTIONS = (
    "Heavy rain falling on a tin roof.",
    "A dog barking in the distance while birds chirp.",
    "Ocean waves crashing on a rocky shore.",
)
DEFAULT_CODEC_CAP = 4000


def parse_args():
    parser = argparse.ArgumentParser(description="Offline Audex TTA inference")
    parser.add_argument("--model", type=str, default="nvidia/Nemotron-Labs-Audex-2B")
    parser.add_argument("--captions", type=str, nargs="+", default=None, help="Audio captions.")
    parser.add_argument(
        "--captions-file",
        type=str,
        default=None,
        help="TSV corpus: one 'utt_id<TAB>caption' per line (overrides --captions).",
    )
    parser.add_argument("--output-dir", type=str, default="results/audex_tta_wavs")
    parser.add_argument(
        "--deploy-config",
        type=str,
        # The model root's default deploy yaml is the TTS pipeline; TTA
        # prompts and RVQ sampling params require the dedicated TTA pipeline.
        default=default_deploy_config("audex_tta.yaml"),
        help="Deploy yaml (defaults to the audex_tta pipeline).",
    )
    parser.add_argument(
        "--xcodec1-path",
        type=str,
        default=None,
        help="Local XCodec1 checkpoint (defaults to $XCODEC1_PATH or the hf-audio repo).",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance strength (official TTA default 3.0; 1.0 disables).",
    )
    parser.add_argument(
        "--codec-cap",
        type=int,
        default=DEFAULT_CODEC_CAP,
        help="Max generated codec tokens before the phase mask forces <audiogen_end>.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.xcodec1_path:
        os.environ["XCODEC1_PATH"] = args.xcodec1_path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(args.captions_file, args.captions, DEFAULT_CAPTIONS)

    tokenizer = load_audex_tokenizer(args.model, profile="tta")
    phase_token_ids, start_tid, end_tid = build_tta_phase_token_ids(tokenizer)
    tta_rvq = {
        "phase_token_ids": phase_token_ids,
        "start_tid": start_tid,
        "end_tid": end_tid,
        "codec_cap": args.codec_cap,
        # The prompt ends with <audiogen_start>, so generation begins at
        # RVQ phase 0 immediately.
        "start_in_prompt": True,
    }

    engine = Omni(model=args.model, deploy_config=args.deploy_config, trust_remote_code=True)

    print(f"Model       : {args.model}")
    print(f"Captions    : {len(corpus)}")
    print(f"CFG scale   : {args.cfg_scale}")
    print(f"Codec cap   : {args.codec_cap}")
    print(f"Output dir  : {output_dir}")

    total_elapsed = 0.0
    total_dur = 0.0
    for utt, caption in corpus:
        cond_prompt = build_tta_cond_prompt(caption)
        sampling = guided_stage_params(
            engine,
            cond_prompt,
            tokenizer,
            args.cfg_scale,
            f"tta-{utt}",
            extra_args={"tta_rvq": tta_rvq},
            null_prompt_builder=build_tta_null_prompt,
        )
        t_start = time.perf_counter()
        outputs = engine.generate([cond_prompt], sampling)
        total_elapsed += time.perf_counter() - t_start

        (req_output,) = outputs
        pcm = extract_pcm(req_output.outputs[0].multimodal_output)
        dur = pcm.numel() / SAMPLE_RATE
        if dur <= 0:
            raise RuntimeError(f"empty audio for {utt}")
        out_path = output_dir / f"{utt}.wav"
        write_wav(out_path, pcm)
        total_dur += dur
        print(f"  {utt:<48} dur={dur:6.2f}s  -> {out_path}")

    rtf = total_elapsed / total_dur if total_dur > 0 else float("inf")
    print(f"Total infer : {total_elapsed:.2f}s  total audio: {total_dur:.2f}s  RTF: {rtf:.3f}")


if __name__ == "__main__":
    main()
