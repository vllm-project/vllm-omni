# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline Audex TTS benchmark: untimed warmup -> timed pass -> WAVs + refs.

Audex is English-only plain TTS (no voice cloning), so unlike the other
benchmark scripts this one draws target texts from the seed-tts English test
set by default (``--split en``) instead of the Chinese wenetspeech4tts
voice-clone subset. Writes ``refs.tsv`` next to the WAVs for
transcribe-tts-output.

Example:

    python examples/offline_inference/text_to_speech/audex/offline_benchmark_audex.py \
        --num-samples 16 --batch-size 8 --output-dir results/audex_bench
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
_SEEDTTS_EN_META = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/tts/seedtts_testset/en/meta.lst"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Offline Audex TTS benchmark")
    parser.add_argument("--model", type=str, default="nvidia/Nemotron-Labs-Audex-2B")
    parser.add_argument(
        "--split",
        type=str,
        default="en",
        help="Text source. Only 'en' (seed-tts English test set) is supported: Audex TTS is English-only.",
    )
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-warmups", type=int, default=2)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/audex_bench")
    parser.add_argument("--deploy-config", type=str, default=None)
    return parser.parse_args()


def _load_texts(split: str, limit: int) -> list[tuple[str, str]]:
    if split != "en":
        print(f"WARNING: Audex TTS is English-only; ignoring split={split!r} and using the seed-tts en set.")
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in _SEEDTTS_EN_META.read_text().splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        text = parts[3].strip()
        if text and text not in seen:
            seen.add(text)
            rows.append((f"en_{len(rows):02d}", text))
        if len(rows) >= limit:
            break
    return rows


def _extract_pcm(multimodal_output: dict) -> torch.Tensor:
    audio = multimodal_output.get("model_outputs")
    if audio is None:
        audio = multimodal_output.get("audio")
    if isinstance(audio, list):
        valid = [torch.as_tensor(a).float().cpu().reshape(-1) for a in audio if a is not None]
        return torch.cat(valid, dim=0) if len(valid) > 1 else valid[0]
    return torch.as_tensor(audio).float().cpu().reshape(-1)


def _req_index(req_output) -> int:
    match = re.search(r"(\d+)", str(req_output.request_id))
    return int(match.group(1)) if match else 0


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus = _load_texts(args.split, args.num_samples)
    if not corpus:
        raise SystemExit("no benchmark texts loaded")

    engine = Omni(model=args.model, deploy_config=args.deploy_config, trust_remote_code=True)

    if not args.no_warmup and args.num_warmups > 0:
        warmup = [build_cond_prompt(text) for _u, text in corpus[: args.num_warmups]]
        engine.generate(warmup)
        print(f"warmup done ({len(warmup)} prompts, untimed)")

    total_elapsed = 0.0
    total_audio = 0.0
    refs_lines: list[str] = []
    for start in range(0, len(corpus), args.batch_size):
        batch = corpus[start : start + args.batch_size]
        prompts = [build_cond_prompt(text) for _u, text in batch]
        t0 = time.perf_counter()
        outputs = engine.generate(prompts)
        total_elapsed += time.perf_counter() - t0

        ordered = sorted(outputs, key=_req_index)
        for (utt, text), req_output in zip(batch, ordered):
            pcm = _extract_pcm(req_output.outputs[0].multimodal_output)
            dur = pcm.numel() / SAMPLE_RATE
            assert dur > 0, f"empty audio for {utt}"
            total_audio += dur
            arr = (np.clip(pcm.numpy(), -1.0, 1.0) * 32767.0).astype(np.int16)
            sf.write(str(output_dir / f"{utt}.wav"), arr, SAMPLE_RATE, format="WAV", subtype="PCM_16")
            refs_lines.append(f"{utt}\t{text}\n")
            print(f"  {utt:<10} dur={dur:6.2f}s")

    (output_dir / "refs.tsv").write_text("".join(refs_lines))
    rtf = total_elapsed / total_audio if total_audio > 0 else float("inf")
    thr = len(corpus) / total_elapsed if total_elapsed > 0 else 0.0
    print(
        f"samples={len(corpus)} batch={args.batch_size} wall={total_elapsed:.2f}s "
        f"audio={total_audio:.2f}s RTF={rtf:.3f} throughput={thr:.2f} samples/s"
    )


if __name__ == "__main__":
    main()
