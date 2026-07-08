# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline Audex (Nemotron-Labs-Audex-2B) TTS inference example.

Runs Stage 0 (thinker) + Stage 1 (streaming causal speech decoder) end-to-end
through the vLLM-Omni engine and saves a 16 kHz mono WAV per prompt.

Pass the HF repo ROOT as --model; per-stage subfolders resolve automatically.

Examples:

    python examples/offline_inference/text_to_speech/audex/end2end.py \\
        --texts "The weather is so good today." \\
        --output-dir results/audex_wavs

    # TSV corpus (utt_id<TAB>text per line), all requests in flight at once:
    python examples/offline_inference/text_to_speech/audex/end2end.py \\
        --texts-file corpus.tsv --output-dir results/audex_wavs

    # One request at a time:
    python examples/offline_inference/text_to_speech/audex/end2end.py \\
        --texts-file corpus.tsv --sequential --output-dir results/audex_wavs
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from vllm_omni import Omni
from vllm_omni.model_executor.models.audex.prompt import build_cond_prompt

SAMPLE_RATE = 16_000
DEFAULT_TEXTS = (
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "Today is a beautiful day for a walk in the park.",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Offline Audex TTS inference")
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/Nemotron-Labs-Audex-2B",
        help="HF repo root (or local snapshot root).",
    )
    parser.add_argument("--texts", type=str, nargs="+", default=None, help="Plain-text prompts.")
    parser.add_argument(
        "--texts-file",
        type=str,
        default=None,
        help="TSV corpus: one 'utt_id<TAB>text' per line (overrides --texts).",
    )
    parser.add_argument("--output-dir", type=str, default="results/audex_wavs")
    parser.add_argument("--deploy-config", type=str, default=None, help="Override the deploy config path.")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Submit one request at a time instead of all at once.",
    )
    return parser.parse_args()


def _slugify(text: str) -> str:
    slug = re.sub(r"\s+", "_", text.strip().lower())
    slug = re.sub(r"[^a-z0-9_]+", "", slug)
    return slug[:48] or "prompt"


def _load_corpus(args) -> list[tuple[str, str]]:
    if args.texts_file:
        corpus = []
        for line in Path(args.texts_file).read_text().splitlines():
            utt, text = line.split("\t", 1)
            corpus.append((utt, text.strip()))
        return corpus
    texts = args.texts if args.texts else list(DEFAULT_TEXTS)
    return [(_slugify(t), t) for t in texts]


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


def _pcm_to_int16(pcm: torch.Tensor) -> np.ndarray:
    arr = np.clip(pcm.numpy(), -1.0, 1.0)
    return (arr * 32767.0).astype(np.int16)


def _write_outputs(outputs, batch: list[tuple[str, str]], output_dir: Path) -> float:
    # Omni.generate may return outputs out of submission order; align by the
    # numeric request-id suffix, which follows submission order.
    def _req_index(req_output) -> int:
        match = re.search(r"(\d+)", str(req_output.request_id))
        return int(match.group(1)) if match else 0

    ordered = sorted(outputs, key=_req_index)
    if len(ordered) != len(batch):
        raise RuntimeError(f"expected {len(batch)} outputs, got {len(ordered)}")

    total_dur = 0.0
    for (utt, _text), req_output in zip(batch, ordered):
        mm = req_output.outputs[0].multimodal_output
        pcm = _extract_pcm(mm)
        dur = pcm.numel() / SAMPLE_RATE
        if dur <= 0:
            raise RuntimeError(f"empty audio for {utt}")
        out_path = output_dir / f"{utt}.wav"
        sf.write(str(out_path), _pcm_to_int16(pcm), SAMPLE_RATE, format="WAV", subtype="PCM_16")
        total_dur += dur
        print(f"  {utt:<40} dur={dur:6.2f}s  -> {out_path}")
    return total_dur


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = _load_corpus(args)

    engine = Omni(model=args.model, deploy_config=args.deploy_config, trust_remote_code=True)

    print(f"Model       : {args.model}")
    print(f"Prompts     : {len(corpus)}")
    print(f"Mode        : {'sequential' if args.sequential else 'batched'}")
    print(f"Output dir  : {output_dir}")

    total_elapsed = 0.0
    total_dur = 0.0
    batches = [[item] for item in corpus] if args.sequential else [corpus]
    for batch in batches:
        prompts = [build_cond_prompt(text) for _utt, text in batch]
        t_start = time.perf_counter()
        outputs = engine.generate(prompts)
        total_elapsed += time.perf_counter() - t_start
        total_dur += _write_outputs(outputs, batch, output_dir)

    rtf = total_elapsed / total_dur if total_dur > 0 else float("inf")
    print(f"Total infer : {total_elapsed:.2f}s  total audio: {total_dur:.2f}s  RTF: {rtf:.3f}")


if __name__ == "__main__":
    main()
