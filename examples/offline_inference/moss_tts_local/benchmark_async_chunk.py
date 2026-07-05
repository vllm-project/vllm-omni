# Measure latency of MOSS-TTS-Local in async_chunk vs sync mode.
"""
Usage (run each mode separately; each needs its own Omni process):

# async_chunk mode:
MOSS_AUDIO_TOKENIZER_PATH=.../moss-audio-tokenizer \\
python benchmark_async_chunk.py \\
    --model .../moss-tts-local \\
    --mode async \\
    --stage-configs-path ../../../vllm_omni/model_executor/stage_configs/moss_tts_local_async.yaml \\
    --text "The weather is so nice today." \\
    --output-dir ./output_async

# sync (batch) mode:
MOSS_AUDIO_TOKENIZER_PATH=.../moss-audio-tokenizer \\
python benchmark_async_chunk.py \\
    --model .../moss-tts-local \\
    --mode sync \\
    --stage-configs-path ../../../vllm_omni/model_executor/stage_configs/moss_tts_local.yaml \\
    --text "The weather is so nice today." \\
    --output-dir ./output_sync

"""

from __future__ import annotations

import copy
import json
import os
import statistics
import time

import torch
import soundfile as sf
from vllm import SamplingParams

from vllm_omni.entrypoints.omni import Omni

SEED = 42

_USER_INST_TEMPLATE = """\
<user_inst>
- Reference(s):
{reference}
- Instruction:
{instruction}
- Tokens:
{tokens}
- Quality:
{quality}
- Sound Event:
{sound_event}
- Ambient Sound:
{ambient_sound}
- Language:
{language}
- Text:
{text}
</user_inst>"""


def build_tts_prompt(text: str, model_path: str) -> str:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.abspath(model_path), trust_remote_code=True
    )
    content = (
        _USER_INST_TEMPLATE
        .replace("{reference}", "None")
        .replace("{instruction}", "None")
        .replace("{tokens}", "None")
        .replace("{quality}", "None")
        .replace("{sound_event}", "None")
        .replace("{ambient_sound}", "None")
        .replace("{language}", "None")
        .replace("{text}", str(text))
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt + "<|audio_start|>"


def _resolve_stop_ids(model_path: str) -> list[int]:
    """Resolve <|audio_end|> + EOS token ids from the model config."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(
        os.path.abspath(model_path), trust_remote_code=True
    )
    audio_end_id = getattr(cfg, "audio_end_token_id", None)
    eos_id = getattr(cfg, "eos_token_id", None)
    if isinstance(eos_id, list):
        eos_ids = eos_id
    elif eos_id is not None:
        eos_ids = [eos_id]
    else:
        eos_ids = []
    return list(dict.fromkeys(
        ([audio_end_id] if audio_end_id is not None else []) + eos_ids
    ))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Latency benchmark for MOSS-TTS-Local: async_chunk vs sync."
    )
    parser.add_argument("--model", default="/workspace/vllm-omni/weights/moss-tts-local")
    parser.add_argument(
        "--stage-configs-path",
        default="../../../vllm_omni/model_executor/stage_configs/moss_tts_local_async.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=["async", "sync"],
        default="async",
        help=(
            "async use moss_tts_local_async.yaml (async_chunk=true); "
            "sync use moss_tts_local.yaml (async_chunk=false). "
            "Pass the matching --stage-configs-path for each mode."
        ),
    )
    parser.add_argument("--text", default="The weather is so nice today.")
    parser.add_argument("--output-dir", default="./bench_output")
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Prompt batch sizes to profile. Defaults to --num-prompts. "
            "This keeps linyueqian's single-prompt path as the default while "
            "supporting the HPML throughput sweep shape."
        ),
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--init-sleep-seconds", type=int, default=20)
    parser.add_argument("--batch-timeout", type=int, default=5)
    parser.add_argument("--init-timeout", type=int, default=5000)
    parser.add_argument("--shm-threshold-bytes", type=int, default=65536)
    args = parser.parse_args()

    if not os.environ.get("MOSS_AUDIO_TOKENIZER_PATH"):
        sibling = os.path.join(
            os.path.dirname(os.path.abspath(args.model)),
            "moss-audio-tokenizer",
        )
        if os.path.exists(sibling):
            os.environ["MOSS_AUDIO_TOKENIZER_PATH"] = sibling
            print(f"[Info] MOSS_AUDIO_TOKENIZER_PATH auto-set -> {sibling}")
        else:
            raise SystemExit(
                "[Error] MOSS_AUDIO_TOKENIZER_PATH must point to a local "
                "MOSS-Audio-Tokenizer snapshot. Download "
                "OpenMOSS-Team/MOSS-Audio-Tokenizer first, or place it beside "
                "the MOSS-TTS-Local model as 'moss-audio-tokenizer'."
            )

    os.makedirs(args.output_dir, exist_ok=True)

    first_chunk_dir = os.path.abspath(os.path.join(args.output_dir, "_first_chunk"))
    os.makedirs(first_chunk_dir, exist_ok=True)
    for _stale in os.listdir(first_chunk_dir):
        if _stale.endswith(".first_chunk.ts"):
            try:
                os.remove(os.path.join(first_chunk_dir, _stale))
            except OSError:
                pass
    os.environ["MOSS_FIRST_CHUNK_DIR"] = first_chunk_dir

    prompt = build_tts_prompt(args.text, args.model)
    print(f"[Info] Mode: {args.mode}")
    print(f"[Info] Config: {args.stage_configs_path}")
    print(f"[Info] Prompt ({len(prompt)} chars):\n{prompt}\n")

    ar_stop_ids = _resolve_stop_ids(args.model)
    print(f"[Info] AR stop token IDs: {ar_stop_ids}")

    ar_params = SamplingParams(
        temperature=0.6, 
        top_p=0.95, 
        top_k=50,
        max_tokens=500, 
        seed=SEED,
        stop_token_ids=ar_stop_ids if ar_stop_ids else None,
    )
    decoder_params = SamplingParams(
        temperature=0.0, 
        top_p=1.0, 
        top_k=-1,
        max_tokens=18192, 
        seed=SEED, 
        detokenize=False,
    )

    omni = Omni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
        init_sleep_seconds=args.init_sleep_seconds,
        batch_timeout=args.batch_timeout,
        init_timeout=args.init_timeout,
        shm_threshold_bytes=args.shm_threshold_bytes,
    )

    batch_sizes = args.batch_sizes or [args.num_prompts]
    if any(bs <= 0 for bs in batch_sizes):
        raise ValueError("--batch-sizes values must be positive.")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")

    results = []
    aggregate_rows = []
    for batch_size in batch_sizes:
        repeat_rows = []
        for repeat_idx in range(args.repeats):
            for _stale in os.listdir(first_chunk_dir):
                if _stale.endswith(".first_chunk.ts"):
                    try:
                        os.remove(os.path.join(first_chunk_dir, _stale))
                    except OSError:
                        pass

            prompts = [copy.deepcopy({"prompt": prompt}) for _ in range(batch_size)]

            print(
                f"[Info] Running batch_size={batch_size}, "
                f"repeat={repeat_idx + 1}/{args.repeats} in {args.mode} mode ..."
            )
            t_start_wall = time.time()
            t_start = time.perf_counter()
            omni_outputs = omni.generate(prompts, [ar_params, decoder_params])
            t_end = time.perf_counter()
            total_time = t_end - t_start

            run_audio_total = 0.0
            run_chunks_total = 0
            run_audio_count = 0
            run_first_chunk_latency = total_time

            for stage_outputs in omni_outputs:
                output = stage_outputs.request_output
                request_id = output.request_id

                if stage_outputs.final_output_type == "text":
                    text_out = output.outputs[0].text
                    print(f"[{request_id}] AR text: {text_out[:200]!r}")
                    continue

                if stage_outputs.final_output_type != "audio":
                    continue

                audio_tensor = output.outputs[0].multimodal_output.get("audio")
                if audio_tensor is None:
                    print(f"[{request_id}] No audio in output.")
                    continue

                num_chunks = 1
                if isinstance(audio_tensor, list):
                    chunks = [
                        t for t in audio_tensor
                        if isinstance(t, torch.Tensor) and t.numel() > 0
                    ]
                    num_chunks = len(chunks)
                    if not chunks:
                        print(f"[{request_id}] No audio chunks in output.")
                        continue
                    audio_tensor = torch.cat(chunks, dim=0)

                audio_np = audio_tensor.float().detach().cpu().numpy()
                if audio_np.ndim > 1:
                    audio_np = audio_np.flatten()

                audio_dur = len(audio_np) / 24000
                run_audio_total += audio_dur
                run_chunks_total += num_chunks
                run_audio_count += 1

                first_chunk_latency: float = total_time
                for key in (request_id, "first"):
                    first_chunk_path = os.path.join(first_chunk_dir, f"{key}.first_chunk.ts")
                    if os.path.exists(first_chunk_path):
                        try:
                            with open(first_chunk_path) as f:
                                ts = float(f.read().strip())
                            first_chunk_latency = max(0.0, ts - t_start_wall)
                            break
                        except (OSError, ValueError):
                            continue
                run_first_chunk_latency = min(run_first_chunk_latency, first_chunk_latency)

                wav_path = os.path.join(
                    args.output_dir,
                    f"bs{batch_size}_rep{repeat_idx}_{request_id}.wav",
                )
                sf.write(wav_path, audio_np, samplerate=24000, format="WAV")

                per_request_rtf = total_time / audio_dur if audio_dur > 0 else float("inf")
                results.append({
                    "request_id": request_id,
                    "mode": args.mode,
                    "batch_size": batch_size,
                    "repeat": repeat_idx,
                    "total_time_s": round(total_time, 3),
                    "audio_dur_s": round(audio_dur, 3),
                    "rtf": round(per_request_rtf, 4),
                    "num_chunks": num_chunks,
                    "first_chunk_latency_s": round(first_chunk_latency, 3),
                    "wav": wav_path,
                })

            throughput = run_audio_total / total_time if total_time > 0 else 0.0
            batch_rtf = total_time / run_audio_total if run_audio_total > 0 else float("inf")
            repeat_row = {
                "mode": args.mode,
                "batch_size": batch_size,
                "repeat": repeat_idx,
                "total_time_s": total_time,
                "audio_total_s": run_audio_total,
                "audio_count": run_audio_count,
                "throughput_audio_s_per_s": throughput,
                "batch_rtf": batch_rtf,
                "first_chunk_latency_s": run_first_chunk_latency,
                "num_chunks_total": run_chunks_total,
            }
            repeat_rows.append(repeat_row)
            aggregate_rows.append(repeat_row)

            print(
                f"\n{'='*60}\n"
                f"  mode               : {args.mode}\n"
                f"  batch_size         : {batch_size}\n"
                f"  repeat             : {repeat_idx + 1}/{args.repeats}\n"
                f"  total_time         : {total_time:.3f}s\n"
                f"  total_audio_dur    : {run_audio_total:.3f}s\n"
                f"  throughput         : {throughput:.4f} audio_s/s\n"
                f"  batch_RTF          : {batch_rtf:.4f}\n"
                f"  first_chunk_latency: {run_first_chunk_latency:.3f}s\n"
                f"  num_chunks_total   : {run_chunks_total}\n"
                f"{'='*60}\n"
            )

        mean_wall = statistics.mean(row["total_time_s"] for row in repeat_rows)
        mean_audio = statistics.mean(row["audio_total_s"] for row in repeat_rows)
        mean_tput = statistics.mean(row["throughput_audio_s_per_s"] for row in repeat_rows)
        mean_rtf = statistics.mean(row["batch_rtf"] for row in repeat_rows)
        mean_fcl = statistics.mean(row["first_chunk_latency_s"] for row in repeat_rows)
        print(
            f"[Summary] mode={args.mode} bs={batch_size} "
            f"wall={mean_wall:.3f}s audio={mean_audio:.3f}s "
            f"throughput={mean_tput:.4f} audio_s/s "
            f"batch_RTF={mean_rtf:.4f} first_chunk={mean_fcl:.3f}s"
        )

    summary_path = os.path.join(args.output_dir, f"bench_{args.mode}.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "mode": args.mode,
                "text": args.text,
                "batch_sizes": batch_sizes,
                "repeats": args.repeats,
                "requests": results,
                "aggregate": [
                    {
                        **row,
                        "total_time_s": round(row["total_time_s"], 6),
                        "audio_total_s": round(row["audio_total_s"], 6),
                        "throughput_audio_s_per_s": round(
                            row["throughput_audio_s_per_s"], 6
                        ),
                        "batch_rtf": round(row["batch_rtf"], 6),
                        "first_chunk_latency_s": round(
                            row["first_chunk_latency_s"], 6
                        ),
                    }
                    for row in aggregate_rows
                ],
            },
            f,
            indent=2,
        )
    print(f"[Info] Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
