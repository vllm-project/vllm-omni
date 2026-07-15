# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline Audex (Nemotron-Labs-Audex-2B) text-to-audio inference example.

Caption → general audio through the vLLM-Omni TTA pipeline: the audiogen
thinker generates interleaved 4-codebook <audiocodec_N> RVQ tokens under an
RVQ phase mask, and the external XCodec1 checkpoint decodes them to a 16 kHz
mono WAV.

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
import copy
import os
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from vllm_omni import Omni
from vllm_omni.model_executor.models.audex.prompt import build_tta_cond_prompt, build_tta_null_prompt
from vllm_omni.model_executor.models.audex.tta import build_tta_phase_token_ids

SAMPLE_RATE = 16_000
# The model root's default deploy yaml is the TTS pipeline; TTA prompts and
# RVQ sampling params require the dedicated TTA pipeline, so default to it.
_DEFAULT_DEPLOY_CONFIG = str(Path(__file__).resolve().parents[3] / "vllm_omni" / "deploy" / "audex_tta.yaml")
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
        default=_DEFAULT_DEPLOY_CONFIG,
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


def _slugify(text: str) -> str:
    slug = re.sub(r"\s+", "_", text.strip().lower())
    slug = re.sub(r"[^a-z0-9_]+", "", slug)
    return slug[:48] or "caption"


def _load_corpus(args) -> list[tuple[str, str]]:
    if args.captions_file:
        corpus = []
        for line in Path(args.captions_file).read_text().splitlines():
            if not line.strip():
                continue
            utt, caption = line.split("\t", 1)
            corpus.append((utt, caption.strip()))
        return corpus
    captions = args.captions if args.captions else list(DEFAULT_CAPTIONS)
    return [(_slugify(c), c) for c in captions]


def _load_audex_tokenizer(model: str):
    from transformers import AutoTokenizer

    from vllm_omni.model_executor.models.audex.checkpoint import ensure_audex_snapshot

    root = ensure_audex_snapshot(model, profile="tta")
    return AutoTokenizer.from_pretrained(str(Path(root) / "checkpoint_folder_audiogen"))


def _extract_pcm(multimodal_output: dict) -> torch.Tensor:
    audio = multimodal_output.get("model_outputs")
    if audio is None:
        audio = multimodal_output.get("audio")
    if audio is None:
        raise ValueError(f"no audio key in multimodal_output: {list(multimodal_output.keys())}")
    if isinstance(audio, list):
        valid = [torch.as_tensor(a).float().cpu().reshape(-1) for a in audio if a is not None]
        if not valid:
            raise ValueError("audio list is empty")
        return torch.cat(valid, dim=0) if len(valid) > 1 else valid[0]
    return torch.as_tensor(audio).float().cpu().reshape(-1)


def _tta_sampling_params(engine: Omni, args, pair_id: str, cond_prompt: str, tokenizer, tta_rvq: dict):
    params = copy.deepcopy(engine.resolve_sampling_params_list(None))
    stage0 = params[0]
    if stage0.extra_args is None:
        stage0.extra_args = {}
    stage0.extra_args["tta_rvq"] = tta_rvq
    if args.cfg_scale > 1.0:
        stage0.extra_args.update(
            {
                "cfg_scale": float(args.cfg_scale),
                "cfg_role": "cond",
                "cfg_pair_id": pair_id,
                "cfg_null_prompt": build_tta_null_prompt(cond_prompt, tokenizer),
            }
        )
    return params


def main():
    args = parse_args()
    if args.xcodec1_path:
        os.environ["XCODEC1_PATH"] = args.xcodec1_path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = _load_corpus(args)

    tokenizer = _load_audex_tokenizer(args.model)
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
        sampling = _tta_sampling_params(engine, args, f"tta-{utt}", cond_prompt, tokenizer, tta_rvq)
        t_start = time.perf_counter()
        outputs = engine.generate([cond_prompt], sampling)
        total_elapsed += time.perf_counter() - t_start

        (req_output,) = outputs
        pcm = _extract_pcm(req_output.outputs[0].multimodal_output)
        dur = pcm.numel() / SAMPLE_RATE
        if dur <= 0:
            raise RuntimeError(f"empty audio for {utt}")
        arr = (np.clip(pcm.numpy(), -1.0, 1.0) * 32767.0).astype(np.int16)
        out_path = output_dir / f"{utt}.wav"
        sf.write(str(out_path), arr, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        total_dur += dur
        print(f"  {utt:<48} dur={dur:6.2f}s  -> {out_path}")

    rtf = total_elapsed / total_dur if total_dur > 0 else float("inf")
    print(f"Total infer : {total_elapsed:.2f}s  total audio: {total_dur:.2f}s  RTF: {rtf:.3f}")


if __name__ == "__main__":
    main()
