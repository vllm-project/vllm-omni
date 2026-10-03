# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Run identical streaming workloads against baseline or patched PYTHONPATH."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# CPU scopes are enabled only in the separate profiler process.
os.environ["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "1" if "--trace-only" in sys.argv else "0"

import numpy as np
import soundfile as sf
import torch
import yaml
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

import vllm_omni
from vllm_omni.entrypoints.omni import Omni


def audio_array(value):
    if isinstance(value, (list, tuple)):
        arrays = [audio_array(item) for item in value if item is not None]
        return np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float32)
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy().reshape(-1)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--fixed-tokens", type=int, default=None)
    parser.add_argument("--warmup-batch-size", type=int, default=1)
    parser.add_argument("--no-async-chunk", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument("--validation-dir", type=Path, default=Path("/validation"))
    args = parser.parse_args()
    if args.requests < 1 or args.warmups < 0 or args.concurrency < 1 or args.warmup_batch_size < 1:
        parser.error("requests/concurrency/warmup-batch-size must be positive and warmups nonnegative")
    if args.fixed_tokens is not None and args.fixed_tokens < 1:
        parser.error("fixed-tokens must be positive")
    root = args.validation_dir.resolve()
    out_dir = root / "results" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(vllm_omni.__file__).parent.parent
    model = root / "models" / "Fun-CosyVoice3-0.5B-2512"
    config = yaml.safe_load((source / "vllm_omni/deploy/cosyvoice3.yaml").read_text())
    config["async_chunk"] = not args.no_async_chunk
    config["connectors"]["connector_of_shared_memory"]["extra"]["codec_chunk_frames"] = 25
    for stage in config["stages"]:
        stage["seed"] = 0
    if args.trace_only:
        config["stages"][0]["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(out_dir / "traces"),
            "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": False,
            "torch_profiler_with_memory": False,
        }
    config_path = out_dir / ("deploy-profile.yaml" if args.trace_only else "deploy.yaml")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    reference, sr = sf.read(source / "tests/assets/cosyvoice3/zero_shot_prompt.wav", dtype="float32")
    if reference.ndim > 1:
        reference = reference.mean(axis=1)
    texts = [
        "Hello, this is a voice cloning test with English text.",
        "收到好友从远方寄来的生日礼物，那份意外的惊喜让我十分感动。",
    ]
    prompts = [
        {
            "prompt": text,
            "multi_modal_data": {"audio": (reference, sr)},
            "modalities": ["audio"],
            "mm_processor_kwargs": {
                "prompt_text": "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
            },
        }
        for text in texts
    ]
    params = [
        SamplingParams(
            seed=0,
            temperature=1.0,
            top_p=0.8,
            top_k=25,
            repetition_penalty=1.0001,
            min_tokens=args.fixed_tokens or 0,
            max_tokens=args.fixed_tokens or 512,
            stop_token_ids=[6562],
            detokenize=False,
            output_kind=RequestOutputKind.DELTA,
        ),
        SamplingParams(max_tokens=2048, output_kind=RequestOutputKind.DELTA, detokenize=False),
    ]
    print("BENCHMARK", args.label, "source", source, "TRT", os.environ.get("COSYVOICE3_TRT"), flush=True)
    omni = Omni(
        model=str(model),
        tokenizer=str(model / "CosyVoice-BlankEN"),
        deploy_config=str(config_path),
        trust_remote_code=True,
        log_stats=True,
        async_chunk=not args.no_async_chunk,
    )

    def run_batch(index, count, save):
        submitted = time.perf_counter()
        requests = {}
        # The public py_generator wrapper closes the engine when exhausted.
        # Use its underlying iterator to preserve warmup across batches.
        for output in omni._run_generation(
            [prompts[(index + i) % len(prompts)] for i in range(count)],
            params,
            use_tqdm=False,
        ):
            mm = getattr(output, "multimodal_output", None) or {}
            if "audio" not in mm:
                continue
            audio = audio_array(mm["audio"])
            if not audio.size:
                continue
            now = time.perf_counter()
            rid = output.request_id
            state = requests.setdefault(rid, {"ttfa_ms": (now - submitted) * 1000, "chunks": []})
            state["chunks"].append(audio.copy())
            state["last_audio_ms"] = (now - submitted) * 1000
            output_metrics = getattr(output, "metrics", None)
            if isinstance(output_metrics, dict):
                stage_metrics = output_metrics.get("stage_metrics", {}).get("0", {})
                intervals = stage_metrics.get("vllm_itls_ms")
                if intervals:
                    state["ar_itls_ms"] = list(intervals)
        wall_ms = (time.perf_counter() - submitted) * 1000
        assert len(requests) == count, (count, list(requests))
        rows = []
        for rid, state in sorted(requests.items()):
            state["chunk_samples"] = [int(chunk.size) for chunk in state["chunks"]]
            if not args.no_async_chunk:
                assert len(state["chunk_samples"]) > 1, (
                    "Expected streaming audio chunks; full-output delivery is not TTFA"
                )
            audio = np.concatenate(state.pop("chunks"))
            assert np.isfinite(audio).all() and audio.size > 0
            request_index = index + int(rid.split("_", 1)[0])
            state.update(
                index=request_index,
                samples=int(audio.size),
                duration_s=audio.size / 24000,
                sha256=hashlib.sha256(audio.tobytes()).hexdigest(),
                batch_e2e_ms=wall_ms,
            )
            if save:
                np.save(out_dir / f"audio_{request_index:03d}.npy", audio)
                sf.write(out_dir / f"audio_{request_index:03d}.wav", audio, 24000, subtype="FLOAT")
            rows.append(state)
        return rows

    try:
        for i in range(args.warmups):
            run_batch(i * args.warmup_batch_size, args.warmup_batch_size, False)
        if args.trace_only:
            omni.start_profile(stages=[0])
            run_batch(0, 1, False)
            omni.stop_profile(stages=[0])
        else:
            rows = []
            for i in range(0, args.requests, args.concurrency):
                rows.extend(run_batch(i, min(args.concurrency, args.requests - i), True))
            result = {
                "label": args.label,
                "source": str(source),
                "requests": rows,
                "concurrency": args.concurrency,
                "fixed_tokens": args.fixed_tokens,
                "warmup_requests": args.warmups * args.warmup_batch_size,
                "async_chunk": not args.no_async_chunk,
            }
            (out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2), flush=True)
    finally:
        omni.close()
    if args.profile and not args.trace_only:
        # Start fresh so profiler scopes cannot affect the latency measurements.
        os.execv(sys.executable, [sys.executable, __file__, *sys.argv[1:], "--trace-only"])


if __name__ == "__main__":
    main()
