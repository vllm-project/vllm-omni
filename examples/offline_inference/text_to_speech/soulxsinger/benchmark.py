"""Repeatable SoulX-Singer offline benchmark (preprocess + DiT, RTF + stage timings).

Usage:
    python benchmark.py --model /path/to/SoulX-Singer --svs \\
        --prompt-audio ../SoulX-Singer/example/audio/zh_prompt.mp3 \\
        --target-audio ../SoulX-Singer/example/audio/music.mp3 \\
        --preprocess-weights-dir /path/to/SoulX-Singer-Preprocess \\
        -o benchmark.wav

    python benchmark.py ... --enable-diffusion-pipeline-profiler
"""

import argparse
import statistics
import time
from pathlib import Path

import soundfile as sf
from end2end import (
    SVC_DEPLOY_CONFIG,
    SVS_DEPLOY_CONFIG,
    add_inference_args,
    build_sampling,
    extract_audio,
    resolve_preprocess_weights_dir,
)

from vllm_omni.entrypoints.omni import Omni


def _audio_duration_sec(outputs) -> float:
    audio_np, sr = extract_audio(outputs)
    return audio_np.size / sr


def _stage_durations_from_output(outputs) -> dict[str, float]:
    omni_out = outputs[0]
    durations = getattr(omni_out, "stage_durations", None) or {}
    if durations:
        return dict(durations)
    ro = getattr(omni_out, "request_output", None)
    if ro is not None:
        return dict(getattr(ro, "stage_durations", None) or {})
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SoulX-Singer offline benchmark")
    add_inference_args(parser)
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs (excluded from stats)")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs")
    parser.add_argument(
        "--enable-diffusion-pipeline-profiler",
        action="store_true",
        help="Enable low-overhead stage timing in DiffusionOutput.stage_durations",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable torch.compile on diff_estimator (eager baseline for A/B)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["bfloat16", "bf16", "float16", "fp16", "half"],
        help="DiT trunk dtype (default from deploy YAML). Use float16 to match upstream --fp16.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Save WAV from the last measured run (after warmup)",
    )
    parser.add_argument(
        "--save-each-run",
        action="store_true",
        help="With --output, also write measured runs as <stem>_run1.wav, _run2.wav, ...",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_dir = resolve_preprocess_weights_dir(args.preprocess_weights_dir)
    deploy_config = args.deploy_config or str(SVS_DEPLOY_CONFIG if args.svs else SVC_DEPLOY_CONFIG)
    mode = "SVS" if args.svs else "SVC"

    omni_kwargs: dict = {
        "model": args.model,
        "deploy_config": deploy_config,
        "async_chunk": False,  # SoulX-Singer currently supports only batch mode (pseudo-streaming was stashed)
    }
    if args.enable_diffusion_pipeline_profiler:
        omni_kwargs["enable_diffusion_pipeline_profiler"] = True
    if args.enforce_eager:
        omni_kwargs["enforce_eager"] = True
    if args.dtype is not None:
        omni_kwargs["dtype"] = args.dtype

    compile_mode = "eager" if args.enforce_eager else "torch.compile"
    dtype_label = args.dtype or "from deploy YAML"
    print(f"Loading SoulX-Singer {mode} from {args.model} [{compile_mode}, dtype={dtype_label}]")

    omni = Omni(**omni_kwargs)
    kind = "svs" if args.svs else "svc"
    sampling = build_sampling(args, preprocess_weights_dir=preprocess_dir, kind=kind)
    prompt = {"prompt_token_ids": [0]}

    latencies_ms: list[float] = []
    rtfs: list[float] = []
    last_stages: dict[str, float] = {}
    last_measured_outputs = None

    total_runs = args.warmup + args.runs
    for run_idx in range(total_runs):
        is_warmup = run_idx < args.warmup
        label = "warmup" if is_warmup else f"run {run_idx - args.warmup + 1}/{args.runs}"
        t0 = time.perf_counter()
        outputs = list(omni.generate([prompt], sampling))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        audio_sec = _audio_duration_sec(outputs)
        rtf = (elapsed_ms / 1000.0) / audio_sec if audio_sec > 0 else float("inf")
        print(f"[{label}] client={elapsed_ms:.1f} ms, audio={audio_sec:.2f}s, RTF={rtf:.3f}")
        if not is_warmup:
            latencies_ms.append(elapsed_ms)
            rtfs.append(rtf)
            last_stages = _stage_durations_from_output(outputs)
            last_measured_outputs = outputs
            if args.output and args.save_each_run:
                measured_idx = run_idx - args.warmup + 1
                out_path = Path(args.output)
                run_path = out_path.with_name(f"{out_path.stem}_run{measured_idx}{out_path.suffix}")
                audio_np, sr = extract_audio(outputs)
                sf.write(str(run_path), audio_np, sr)
                print(f"  saved {run_path} ({sr} Hz, {audio_np.size / sr:.2f}s)")

    omni.close()

    if args.output and last_measured_outputs is not None:
        audio_np, sr = extract_audio(last_measured_outputs)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio_np, sr)
        print(f"\nSaved last measured run → {out_path} ({sr} Hz, {audio_np.size / sr:.2f}s)")

    if latencies_ms:
        print("\n=== Summary (measured runs) ===")
        print(f"client_ms: mean={statistics.mean(latencies_ms):.1f}, stdev={statistics.pstdev(latencies_ms):.1f}")
        print(f"RTF:       mean={statistics.mean(rtfs):.3f}, stdev={statistics.pstdev(rtfs):.3f}")
        if last_stages:
            print("\nStage durations (last measured run):")
            for name, value in sorted(last_stages.items()):
                if name.endswith("_ms"):
                    print(f"  {name}: {value:.1f} ms")
                else:
                    print(f"  {name}: {value * 1000.0:.1f} ms")


if __name__ == "__main__":
    main()
