# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare uncached MOSS Local NPU eager and ACLGraph reference encoding.

Run with an installed Ascend environment and locally available model/tokenizer.
No HTTP server is needed. Each backend runs in a fresh child process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def compare_codes(left, right) -> dict:
    import numpy as np

    same_shape = left.shape == right.shape
    return {
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "same_shape": same_shape,
        "equal": bool(np.array_equal(left, right)),
        "mismatch_fraction": float(np.mean(left != right)) if same_shape and left.size else None,
    }


def worker(args) -> None:
    os.environ["VLLM_OMNI_MOSS_REF_GRAPHS"] = "1" if args.worker == "graph" else "0"
    os.environ["VLLM_OMNI_MOSS_REF_SHARED_ENCODER"] = "0"
    os.environ["VLLM_OMNI_MOSS_REF_GPU_PREP"] = "0"
    os.environ.pop("VLLM_OMNI_MOSS_REF_CODES_SHARED_DIR", None)

    import numpy as np
    import soundfile as sf
    import torch
    import torch_npu
    from transformers import AutoProcessor

    from vllm_omni.model_executor.models.moss_tts.reference_encoder import _prep_wav_sync, build_reference_encoder
    from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

    device = torch.device(args.device)
    npu = torch.get_device_module("npu")
    load_started = time.perf_counter()
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    proc.audio_tokenizer = proc.audio_tokenizer.to(device).eval()
    npu.synchronize(device)
    model_load_seconds = time.perf_counter() - load_started
    if next(proc.audio_tokenizer.parameters()).device != device:
        raise RuntimeError("Tokenizer did not move to the requested NPU")
    # Use the same graph-compatible kernel settings in both measurement workers.
    # This is a benchmark control; production eager defaults are not changed.
    with npu.device(device):
        npu.config.allow_internal_format = False
        npu.set_compile_mode(jit_compile=False)
    enc = build_reference_encoder(proc, variant="local", speaker_cache=SpeakerEmbeddingCache(max_bytes=8 * 1024**2))
    references = []
    for path in args.references:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        if rate <= 0 or not len(audio) or not np.isfinite(audio).all():
            raise ValueError(f"Invalid or empty reference audio: {path}")
        references.append((audio.T.copy(), rate))

    def sync():
        npu.synchronize(device)

    def measure():
        with npu.device(device):
            sync()
            memory_before = npu.memory_allocated(device)
            npu.reset_peak_memory_stats(device)
            begin = time.perf_counter()
            enc.prepare()
            sync()
            preparation = time.perf_counter() - begin
            graphs = enc._graphs
            if args.worker == "graph" and (graphs is None or graphs.stats["captures"] == 0):
                raise RuntimeError(
                    "No actual reference ACLGraph captured; refusing to report an eager fallback as graphs"
                )
            memory_after = npu.memory_allocated(device)
            rows = []
            for index, item in enumerate(references):
                for iteration in range(args.repeats):
                    before = dict(graphs.stats) if graphs else None
                    sync()
                    begin = time.perf_counter()
                    # Serving worker body before content-cache lookup: no speech generation.
                    codes = enc._encode_batch_sync([item])[0]
                    if isinstance(codes, BaseException):
                        raise codes
                    sync()
                    elapsed = time.perf_counter() - begin
                    array = codes.detach().cpu().numpy()
                    if array.ndim != 2 or not array.size or array.dtype != np.int64 or array.min() < 0:
                        raise ValueError(f"Invalid reference codes: {array.shape}, {array.dtype}")
                    np.save(args.output / f"{args.worker}-ref{index + 1}-{iteration}.npy", array)
                    row = {
                        "reference": index + 1,
                        "iteration": iteration,
                        "seconds": elapsed,
                        "shape": list(array.shape),
                        "graph_stats_before": before,
                        "graph_stats_after": dict(graphs.stats) if graphs else None,
                    }
                    rows.append(row)
                    print(f"参考编码 / Reference encoding: {args.worker} {row}", flush=True)
            # Exercise batching and a shape outside all buckets without an HTTP server.
            mixed = enc._encode_batch_sync(references)
            for index, codes in enumerate(mixed):
                if isinstance(codes, BaseException):
                    raise codes
                np.save(args.output / f"{args.worker}-mixed-ref{index + 1}.npy", codes.cpu().numpy())
            outside = (
                np.tile(references[0][0], (1, 1 + int(9 * references[0][1] / references[0][0].shape[-1]))),
                references[0][1],
            )
            # Ensure the fallback clip is strictly longer than eight seconds.
            outside = (outside[0][:, : int(9 * outside[1])], outside[1])
            before = dict(graphs.stats) if graphs else None
            codes = enc._encode_batch_sync([outside])[0]
            if isinstance(codes, BaseException):
                raise codes
            np.save(args.output / f"{args.worker}-outside.npy", codes.cpu().numpy())
            if graphs and graphs.stats != before:
                raise RuntimeError("A >8-second reference unexpectedly used or expanded the graph cache")

            # Separate padding effects from capture/replay effects: use identical
            # padded audio/lengths for the eager tensor core and captured bucket.
            same_bucket = []
            if graphs:
                for index, (waveform, rate) in enumerate(references):
                    wav = enc._processor_prepare(_prep_wav_sync(waveform, rate, enc._sr_target))
                    bucket = next((s for b, s in graphs.captured if b == 1 and s >= wav.shape[-1]), None)
                    if bucket is None:
                        continue
                    audio = torch.zeros(1, wav.shape[0], bucket, device=device)
                    audio[0, :, : wav.shape[-1]].copy_(wav)
                    lengths = torch.tensor([wav.shape[-1]], dtype=torch.long, device=device)
                    with graphs._context():
                        core_codes, core_lengths = graphs._core(audio, lengths)
                        core_codes = core_codes[:, 0, : int(core_lengths[0].item())].T.contiguous().long().cpu()
                    replay_codes = graphs.encode([wav])[0]
                    np.save(args.output / f"graph-ref{index + 1}-padded-eager.npy", core_codes.numpy())
                    same_bucket.append(
                        {"reference": index + 1, **compare_codes(core_codes.numpy(), replay_codes.numpy())}
                    )
                if graphs.stats["hits"] == 0:
                    raise RuntimeError("No real graph replay observed")
            if args.profile:
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    record_shapes=True,
                    profile_memory=True,
                ) as profile:
                    with torch.profiler.record_function("moss_uncached_reference_encoding"):
                        result = enc._encode_batch_sync([references[0]])[0]
                        if isinstance(result, BaseException):
                            raise result
                        sync()
                profile.export_chrome_trace(str(args.output / f"{args.worker}-trace.json"))
            return {
                "backend": args.worker,
                "device": str(device),
                "device_name": npu.get_device_name(device),
                "torch_version": torch.__version__,
                "torch_npu_version": torch_npu.__version__,
                "compute_dtype": str(getattr(proc.audio_tokenizer, "compute_dtype", None)),
                "model_load_seconds": model_load_seconds,
                "prepare_seconds": preparation,
                "allocated_before_prepare": memory_before,
                "allocated_after_prepare": memory_after,
                "peak_allocated_bytes": npu.max_memory_allocated(device),
                "graph_shapes": graphs.captured if graphs else [],
                "graph_stats": dict(graphs.stats) if graphs else None,
                "same_bucket_eager_vs_graph": same_bucket,
                "rows": rows,
                "cache_bypassed": True,
                "kernel_settings": {"allow_internal_format": False, "jit_compile": False},
            }

    # Capture and replay both run on the same serving-style background worker.
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(measure).result()
    write_json(args.output / f"{args.worker}.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--references", nargs="+", required=True, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action="store_true", help="Export one untimed CPU/NPU trace per backend")
    parser.add_argument("--worker", choices=("eager", "graph"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3 (first use plus warm measurements)")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    for mode in ("eager", "graph"):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker", mode], check=True)
    import numpy as np

    eager = json.loads((args.output / "eager.json").read_text(encoding="utf-8"))
    graph = json.loads((args.output / "graph.json").read_text(encoding="utf-8"))
    comparisons = []
    for index, path in enumerate(args.references, 1):
        left = np.load(args.output / f"eager-ref{index}-0.npy", allow_pickle=False)
        right = np.load(args.output / f"graph-ref{index}-0.npy", allow_pickle=False)
        row = {
            "reference": index,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            **compare_codes(left, right),
        }
        for mode, record in (("eager", eager), ("graph", graph)):
            warm = [r["seconds"] for r in record["rows"] if r["reference"] == index and r["iteration"] > 0]
            row[f"{mode}_warm_seconds"] = {"median": statistics.median(warm), "min": min(warm), "max": max(warm)}
            row[f"{mode}_repeat_equal"] = all(
                compare_codes(
                    np.load(args.output / f"{mode}-ref{index}-0.npy", allow_pickle=False),
                    np.load(args.output / f"{mode}-ref{index}-{repeat}.npy", allow_pickle=False),
                )["equal"]
                for repeat in range(1, args.repeats)
            )
        comparisons.append(row)
    summary = {
        "model": args.model,
        "references": [str(path.resolve()) for path in args.references],
        "repeats": args.repeats,
        "uncached_encoding_only": True,
        "load_and_capture_excluded_from_warm_timing": True,
        "comparisons": comparisons,
        "same_bucket_eager_vs_graph": graph["same_bucket_eager_vs_graph"],
        "outside_bucket": compare_codes(
            np.load(args.output / "eager-outside.npy", allow_pickle=False),
            np.load(args.output / "graph-outside.npy", allow_pickle=False),
        ),
        "mixed_batch": [
            compare_codes(
                np.load(args.output / f"eager-mixed-ref{i}.npy", allow_pickle=False),
                np.load(args.output / f"graph-mixed-ref{i}.npy", allow_pickle=False),
            )
            for i in range(1, len(args.references) + 1)
        ],
    }
    write_json(args.output / "comparison.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
