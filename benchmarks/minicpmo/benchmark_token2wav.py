# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM-o 4.5 Token2Wav isolated microbenchmark.

The benchmark intentionally measures only the Stage-2 Token2Wav boundary used by
``MiniCPMO45OmniTTSForConditionalGeneration`` after audio tokens already exist.
It uses the MiniCPM-o-flavored ``stepaudio2.Token2wav`` package and the official
``assets/token2wav`` files from ``openbmb/MiniCPM-o-4_5``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

SAMPLE_RATE = 24_000
DEFAULT_LENGTHS = (256, 512, 1024, 2048, 4096)
MIN_TAIL = 6


def _load_token2wav_cls():
    try:
        from stepaudio2 import Token2wav
    except ImportError as exc:
        raise RuntimeError(
            "benchmark_token2wav.py requires the MiniCPM-o-flavored "
            "`stepaudio2.Token2wav` entry point. Install `stepaudio2-minicpmo` "
            "or `vllm-omni[minicpmo]`."
        ) from exc
    return Token2wav


def _install_torchaudio_soundfile_shim() -> None:
    """Make Token2wav prompt I/O work when torchcodec/ffmpeg is unavailable."""
    try:
        import torchaudio
    except Exception:
        return

    if not getattr(torchaudio, "_minicpmo_soundfile_shim_installed", False):
        orig_load = torchaudio.load

        def patched_load(uri, *args: Any, **kwargs: Any):
            try:
                return orig_load(uri, *args, **kwargs)
            except Exception:
                data, sr = sf.read(uri, dtype="float32", always_2d=True)
                wav = torch.from_numpy(np.ascontiguousarray(data.T))
                return wav, sr

        torchaudio.load = patched_load

    if not getattr(torchaudio, "_minicpmo_soundfile_save_shim_installed", False):
        orig_save = torchaudio.save

        def patched_save(uri, src, sample_rate, **kwargs: Any):
            kwargs.pop("backend", None)
            if hasattr(uri, "write"):
                sf.write(uri, src.detach().cpu().numpy().T, sample_rate, format="WAV")
                return
            return orig_save(uri, src, sample_rate, backend="soundfile", **kwargs)

        torchaudio.save = patched_save

    torchaudio._minicpmo_soundfile_shim_installed = True
    torchaudio._minicpmo_soundfile_save_shim_installed = True


@dataclass
class IterationResult:
    latency_s: float
    setup_s: float
    decode_or_concat_s: float
    waveform_samples: int
    waveform_duration_s: float
    rtf: float
    peak_allocated_mib: float
    peak_reserved_mib: float
    finite: bool
    nan_count: int
    inf_count: int
    clipping_ratio: float
    rms: float
    peak_abs: float
    mean: float


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        _call_accelerator_or_cuda("synchronize", device)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        _call_accelerator_or_cuda("reset_peak_memory_stats", device)


def _peak_memory_mib(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    return (
        _call_accelerator_or_cuda("max_memory_allocated", device) / (1024**2),
        _call_accelerator_or_cuda("max_memory_reserved", device) / (1024**2),
    )


def _call_accelerator_or_cuda(name: str, *args: Any) -> Any:
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None:
        accelerator_fn = getattr(accelerator, name, None)
        if accelerator_fn is not None:
            return accelerator_fn(*args)
    cuda_fn = getattr(torch.cuda, name)
    return cuda_fn(*args)


def _git_commit() -> str | None:
    if commit := os.getenv("VLLM_OMNI_COMMIT"):
        return commit
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return proc.stdout.strip() or None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _waveform_stats(
    waveform: np.ndarray,
    elapsed_s: float,
    peak_allocated: float,
    peak_reserved: float,
) -> IterationResult:
    wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
    finite_mask = np.isfinite(wav)
    samples = int(wav.shape[0])
    duration_s = samples / SAMPLE_RATE if samples else 0.0
    abs_wav = np.abs(wav[finite_mask]) if finite_mask.any() else np.asarray([], dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(wav[finite_mask])))) if finite_mask.any() else math.nan
    peak_abs = float(abs_wav.max()) if abs_wav.size else math.nan
    return IterationResult(
        latency_s=elapsed_s,
        setup_s=0.0,
        decode_or_concat_s=0.0,
        waveform_samples=samples,
        waveform_duration_s=duration_s,
        rtf=elapsed_s / duration_s if duration_s > 0 else math.inf,
        peak_allocated_mib=peak_allocated,
        peak_reserved_mib=peak_reserved,
        finite=bool(finite_mask.all()),
        nan_count=int(np.isnan(wav).sum()),
        inf_count=int(np.isinf(wav).sum()),
        clipping_ratio=float((abs_wav >= 0.999).mean()) if abs_wav.size else math.nan,
        rms=rms,
        peak_abs=peak_abs,
        mean=float(np.mean(wav[finite_mask])) if finite_mask.any() else math.nan,
    )


def _decode_wav_bytes(wav_bytes: bytes) -> np.ndarray:
    waveform, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if sr != SAMPLE_RATE:
        raise RuntimeError(f"Expected {SAMPLE_RATE} Hz output, got {sr} Hz")
    return np.asarray(waveform, dtype=np.float32).reshape(-1)


def _chunk_bounds(num_tokens: int, chunk_size: int) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < num_tokens:
        end = min(start + chunk_size, num_tokens)
        if 0 < num_tokens - end < MIN_TAIL:
            end = num_tokens
        bounds.append((start, end))
        start = end
    return bounds


def _prepare_seed_tokens(tokenizer: Any, prompt_wav: str, max_len: int) -> list[int]:
    if tokenizer.cache is None:
        tokenizer.cache = tokenizer._prepare_prompt(prompt_wav)
    prompt_tokens = tokenizer.cache[0].detach().reshape(-1).to("cpu").tolist()
    if not prompt_tokens:
        raise RuntimeError("Prompt audio produced no speech tokens")
    repeats = (max_len + len(prompt_tokens) - 1) // len(prompt_tokens)
    return (prompt_tokens * repeats)[:max_len]


def _measure_one_shot(
    tokenizer: Any,
    tokens: list[int],
    prompt_wav: str,
    device: torch.device,
) -> IterationResult:
    _reset_peak_memory(device)
    _sync(device)
    start = time.perf_counter()
    wav_bytes = tokenizer(tokens, prompt_wav)
    after_call = time.perf_counter()
    waveform = _decode_wav_bytes(wav_bytes)
    end = time.perf_counter()
    _sync(device)
    peak_allocated, peak_reserved = _peak_memory_mib(device)
    result = _waveform_stats(waveform, end - start, peak_allocated, peak_reserved)
    result.decode_or_concat_s = end - after_call
    return result


def _measure_streaming(
    tokenizer: Any,
    tokens: list[int],
    prompt_wav: str,
    device: torch.device,
    chunk_size: int,
) -> IterationResult:
    _reset_peak_memory(device)
    _sync(device)
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        stream_cache, hift_cache_dict = tokenizer.set_stream_cache(prompt_wav)
    tokenizer.stream_cache = stream_cache
    tokenizer.hift_cache_dict = hift_cache_dict
    after_setup = time.perf_counter()
    pieces: list[np.ndarray] = []
    try:
        bounds = _chunk_bounds(len(tokens), chunk_size)
        for idx, (chunk_start, chunk_end) in enumerate(bounds):
            wav_np = tokenizer.stream(
                tokens[chunk_start:chunk_end],
                prompt_wav,
                last_chunk=idx == len(bounds) - 1,
                return_waveform=True,
            )
            pieces.append(np.asarray(wav_np, dtype=np.float32).reshape(-1))
        waveform = np.concatenate(pieces, axis=0) if pieces else np.asarray([], dtype=np.float32)
    finally:
        tokenizer.stream_cache = None
        tokenizer.hift_cache_dict = {}
    end = time.perf_counter()
    _sync(device)
    peak_allocated, peak_reserved = _peak_memory_mib(device)
    result = _waveform_stats(waveform, end - start, peak_allocated, peak_reserved)
    result.setup_s = after_setup - start
    result.decode_or_concat_s = end - after_setup
    return result


def _summarize(results: list[IterationResult]) -> dict[str, Any]:
    latencies = [r.latency_s for r in results]
    rtfs = [r.rtf for r in results]
    peak_allocated = [r.peak_allocated_mib for r in results]
    peak_reserved = [r.peak_reserved_mib for r in results]
    return {
        "iterations": len(results),
        "latency_p50_s": statistics.median(latencies),
        "latency_p95_s": _percentile(latencies, 0.95),
        "latency_mean_s": statistics.mean(latencies),
        "rtf_p50": statistics.median(rtfs),
        "rtf_p95": _percentile(rtfs, 0.95),
        "peak_allocated_mib_max": max(peak_allocated),
        "peak_reserved_mib_max": max(peak_reserved),
        "waveform_duration_s_median": statistics.median(r.waveform_duration_s for r in results),
        "finite_all": all(r.finite for r in results),
        "nan_count_total": sum(r.nan_count for r in results),
        "inf_count_total": sum(r.inf_count for r in results),
        "clipping_ratio_max": max(r.clipping_ratio for r in results),
        "rms_median": statistics.median(r.rms for r in results),
        "peak_abs_max": max(r.peak_abs for r in results),
    }


def _environment(model_revision: str | None) -> dict[str, Any]:
    return {
        "model": "openbmb/MiniCPM-o-4_5",
        "model_revision": model_revision,
        "vllm_omni_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_device_count": _call_accelerator_or_cuda("device_count") if torch.cuda.is_available() else 0,
        "pid": os.getpid(),
    }


def _parse_device(value: str | int) -> torch.device:
    text = str(value)
    if text.isdigit():
        return torch.device("cuda", int(text))
    return torch.device(text)


def _result_row(
    *,
    precision: str,
    mode: str,
    length: int,
    phase: str,
    iteration: int,
    result: IterationResult | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "precision": precision,
        "mode": mode,
        "length": length,
        "phase": phase,
        "iteration": iteration,
        "ok": result is not None and error is None,
        "error": error,
    }
    if result is not None:
        row.update(asdict(result))
        return row
    row.update(
        {
            "latency_s": math.nan,
            "setup_s": math.nan,
            "decode_or_concat_s": math.nan,
            "waveform_samples": 0,
            "waveform_duration_s": 0.0,
            "rtf": math.inf,
            "peak_allocated_mib": math.nan,
            "peak_reserved_mib": math.nan,
            "finite": False,
            "nan_count": 0,
            "inf_count": 0,
            "clipping_ratio": math.nan,
            "rms": math.nan,
            "peak_abs": math.nan,
            "mean": math.nan,
        }
    )
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = _parse_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    token2wav_dir = str(args.token2wav_dir)
    prompt_wav = str(args.prompt_wav)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _install_torchaudio_soundfile_shim()
    token2wav_cls = _load_token2wav_cls()
    tokenizer = token2wav_cls(token2wav_dir, float16=args.float16, n_timesteps=args.n_timesteps)
    max_len = max(args.lengths)
    seed_tokens = _prepare_seed_tokens(tokenizer, prompt_wav, max_len)

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for length in args.lengths:
        tokens = seed_tokens[:length]
        precision = "float16" if args.float16 else "float32"
        for mode in args.modes:
            measure = _measure_one_shot if mode == "one_shot" else _measure_streaming
            warmup_failures: list[str] = []
            for warmup_iteration in range(args.warmup):
                try:
                    if mode == "one_shot":
                        measure(tokenizer, tokens, prompt_wav, device)  # type: ignore[misc]
                    else:
                        measure(tokenizer, tokens, prompt_wav, device, args.chunk_size)  # type: ignore[misc]
                except Exception as exc:  # noqa: BLE001 - benchmark should record failures
                    error = f"CUDA OOM: {exc}" if isinstance(exc, torch.cuda.OutOfMemoryError) else repr(exc)
                    warmup_failures.append(error)
                    rows.append(
                        _result_row(
                            precision=precision,
                            mode=mode,
                            length=length,
                            phase="warmup",
                            iteration=warmup_iteration,
                            error=error,
                        )
                    )
                    if isinstance(exc, torch.cuda.OutOfMemoryError):
                        _call_accelerator_or_cuda("empty_cache")
                    break

            measured: list[IterationResult] = []
            failures: list[str] = []
            if not warmup_failures:
                for iteration in range(args.iters):
                    try:
                        if mode == "one_shot":
                            result = measure(tokenizer, tokens, prompt_wav, device)  # type: ignore[misc]
                        else:
                            result = measure(tokenizer, tokens, prompt_wav, device, args.chunk_size)  # type: ignore[misc]
                        measured.append(result)
                        rows.append(
                            _result_row(
                                precision=precision,
                                mode=mode,
                                length=length,
                                phase="measure",
                                iteration=iteration,
                                result=result,
                            )
                        )
                    except torch.cuda.OutOfMemoryError as exc:
                        error = f"CUDA OOM: {exc}"
                        failures.append(error)
                        rows.append(
                            _result_row(
                                precision=precision,
                                mode=mode,
                                length=length,
                                phase="measure",
                                iteration=iteration,
                                error=error,
                            )
                        )
                        _call_accelerator_or_cuda("empty_cache")
                    except Exception as exc:  # noqa: BLE001 - benchmark should record failures
                        error = repr(exc)
                        failures.append(error)
                        rows.append(
                            _result_row(
                                precision=precision,
                                mode=mode,
                                length=length,
                                phase="measure",
                                iteration=iteration,
                                error=error,
                            )
                        )

            summary = {
                "precision": precision,
                "mode": mode,
                "length": length,
                "warmup_failures": warmup_failures,
                "failures": failures,
                "successes": len(measured),
                "requested_iterations": args.iters,
                "chunk_size": args.chunk_size if mode == "streaming" else None,
                "summary": _summarize(measured) if measured else None,
            }
            summaries.append(summary)
            print(json.dumps(summary, sort_keys=True), flush=True)

    result_payload = {
        "environment": _environment(args.model_revision),
        "args": {
            "lengths": args.lengths,
            "warmup": args.warmup,
            "iters": args.iters,
            "chunk_size": args.chunk_size,
            "n_timesteps": args.n_timesteps,
            "float16": args.float16,
            "modes": args.modes,
            "token2wav_dir": token2wav_dir,
            "prompt_wav": prompt_wav,
            "device": str(device),
        },
        "summaries": summaries,
    }
    json_path = output_dir / f"token2wav_{'fp16' if args.float16 else 'fp32'}.json"
    csv_path = output_dir / f"token2wav_{'fp16' if args.float16 else 'fp32'}.csv"
    json_path.write_text(json.dumps(result_payload, indent=2, sort_keys=True), encoding="utf-8")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return result_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token2wav-dir", type=Path, required=True)
    parser.add_argument("--prompt-wav", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/minicpmo/results"))
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-timesteps", type=int, default=10)
    parser.add_argument("--device", default="0", help="CUDA index (e.g. 0) or torch device string (e.g. cuda:0).")
    parser.add_argument("--float16", action="store_true")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--modes", choices=("one_shot", "streaming"), nargs="+", default=["one_shot", "streaming"])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
