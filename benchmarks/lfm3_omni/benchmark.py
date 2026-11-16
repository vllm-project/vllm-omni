# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure warmed LFM3 speech requests and retain audio for paired comparisons."""

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

from vllm_omni import Omni


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="LiquidAI/LFM3-Omni-2B")
    parser.add_argument("--revision", default="21d75123f86c1f7ffb4bef1ea202710c4e7ebe67")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", choices=["eager", "depth", "optimized"], default="eager")
    parser.add_argument("--audio-temperature", type=float, nargs="+", default=[0.0, 0.4])
    parser.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument(
        "--deploy-config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "vllm_omni/deploy/lfm3_omni.yaml",
        help="vllm_omni/deploy/lfm3_audio.yaml for LiquidAI/LFM3-Audio-2B",
    )
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 1 or n > 4 for n in args.concurrency):
        parser.error("repeats must be positive; the bundled deployment supports concurrency 1 through 4")
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Perform TTS in American English using a female voice."},
            {"role": "user", "content": args.text},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt + "<|audio_start|>", add_special_tokens=False)
    deploy = args.deploy_config
    manifest = {
        "model": args.model,
        "revision": args.revision,
        "mode": args.mode,
        "concurrency": args.concurrency,
        "repeats": args.repeats,
        "warmups_per_concurrency": 1,
        "text": args.text,
        "seed": 42,
        "audio_temperature": args.audio_temperature,
        "max_new_tokens": 96,
        "versions": {name: version(name) for name in ("torch", "transformers", "vllm")},
        "timing": "Omni API streaming output, including orchestration and codec; excludes startup and warmups",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    engine = Omni(
        model=args.model,
        revision=args.revision,
        trust_remote_code=True,
        deploy_config=str(deploy),
        async_chunk=True,
        limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
        hf_overrides={
            "lfm3_depth_graph": args.mode != "eager",
            "lfm3_vectorized_greedy": args.mode == "optimized",
        },
    )
    measurements = []
    try:
        for audio_temperature in args.audio_temperature:
            for concurrency in args.concurrency:
                for repeat in range(-1, args.repeats):
                    params = [
                        SamplingParams(
                            temperature=0,
                            max_tokens=96,
                            seed=42,
                            extra_args={"audio_temperature": audio_temperature},
                        ),
                        SamplingParams(temperature=0, max_tokens=1, output_kind=RequestOutputKind.DELTA),
                    ]
                    first_audio: dict[str, float] = {}
                    finished: dict[str, float] = {}
                    parts: dict[str, list[torch.Tensor]] = {}
                    start = time.perf_counter()
                    outputs = engine.generate(
                        [{"prompt_token_ids": prompt_ids} for _ in range(concurrency)],
                        sampling_params_list=params,
                        py_generator=True,
                        use_tqdm=False,
                    )
                    for output in outputs:
                        if output.stage_id != 1:
                            continue
                        elapsed = time.perf_counter() - start
                        request_id = output.request_id
                        payload = output.outputs[0].multimodal_output
                        # DELTA streams can also emit metadata-only updates.
                        audio = payload.get("audio") if payload else None
                        if audio is not None:
                            audio = torch.cat(audio) if isinstance(audio, list) else audio
                            if audio.numel():
                                first_audio.setdefault(request_id, elapsed)
                                parts.setdefault(request_id, []).append(audio.reshape(-1).cpu())
                        if output.finished:
                            finished[request_id] = elapsed
                    wall = time.perf_counter() - start
                    if len(finished) != concurrency or len(parts) != concurrency:
                        raise RuntimeError("Not every request completed with audio")
                    requests = []
                    for request_id, chunks in parts.items():
                        audio = torch.cat(chunks)
                        if not torch.isfinite(audio).all():
                            raise RuntimeError("Non-finite audio")
                        duration = audio.numel() / 24000
                        index = int(request_id.split("_", 1)[0])
                        if repeat >= 0:
                            torch.save(
                                audio, args.output / f"temp{audio_temperature}-c{concurrency}-r{repeat}-i{index}.pt"
                            )
                        requests.append(
                            {
                                "index": index,
                                "first_audio_s": first_audio[request_id],
                                "latency_s": finished[request_id],
                                "samples": audio.numel(),
                                "duration_s": duration,
                                "rtf": finished[request_id] / duration,
                            }
                        )
                    row = {
                        "audio_temperature": audio_temperature,
                        "concurrency": concurrency,
                        "repeat": repeat,
                        "warmup": repeat < 0,
                        "wall_s": wall,
                        "completed": len(finished),
                        "failed": 0,
                        "requests_per_second": len(finished) / wall,
                        "requests": requests,
                    }
                    measurements.append(row)
                    (args.output / "measurements.json").write_text(json.dumps(measurements, indent=2) + "\n")
                    print(json.dumps(row), flush=True)
    finally:
        engine.close()
    (args.output / "complete.json").write_text('{"completed": true}\n')


if __name__ == "__main__":
    main()
