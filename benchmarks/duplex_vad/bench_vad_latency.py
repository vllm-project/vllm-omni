# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Isolated latency comparison: ``server_vad`` (Silero) vs ``semantic_vad`` (Silero + Smart Turn).

No model serving and no GPU: the detectors run exactly as the duplex engine runs
them (one shared ONNX backend, one per-session detector, 16 kHz float32 chunks).

Sections:
  1. Per-call model latency: one Silero frame, and Smart Turn split into
     Whisper feature extraction and the ONNX run.
  2. Smart Turn under concurrency (sessions hitting the shared backend at once).
  3. Per-chunk detector cost, ``ServerTurnDetector.process_segments`` vs
     ``SemanticTurnDetector.process_segments``, for ordinary chunks and for the chunk
     that triggers a Smart Turn evaluation.
  4. Smart Turn classification of labeled clips (complete vs incomplete).
  5. Streaming endpoint decisions in audio time, per language:
     - complete: a complete utterance, then silence. Endpoint delay is end of
       speech -> commit.
     - incomplete -> continue: an incomplete utterance, a pause, then a
       complete one, as if the speaker stopped to think. Any commit before the
       end is a false cutoff.
  6. Resident memory added by loading each model.

Labeled clips come from a manifest (a JSON list of ``{"path", "complete",
"language"}``), e.g. a sample of ``pipecat-ai/smart-turn-data-v3.2-test``;
plain ``--wav`` clips count as complete.

Usage:
  python benchmarks/duplex_vad/bench_vad_latency.py --manifest clips.json --markdown
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import resource
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from vllm_omni.engine.duplex.smart_turn import SmartTurnBackendProvider
from vllm_omni.engine.duplex.turn_detection import TurnDetectionConfig
from vllm_omni.engine.duplex.vad import SileroVADBackendProvider

RATE = 16_000


# --------------------------------------------------------------------------- #
# Audio helpers                                                               #
# --------------------------------------------------------------------------- #


def load_16k(path: Path) -> np.ndarray:
    import soundfile

    audio, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != RATE:
        size = int(round(audio.size * RATE / rate))
        audio = np.interp(np.linspace(0, audio.size - 1, size), np.arange(audio.size), audio).astype(np.float32)
    return audio


def _frame_rms(audio: np.ndarray, frame: int = 160) -> np.ndarray:
    count = audio.size // frame
    return np.sqrt((audio[: count * frame].reshape(count, frame) ** 2).mean(axis=1))


def _speech_frames(audio: np.ndarray) -> np.ndarray:
    rms = _frame_rms(audio)
    return np.nonzero(rms > max(0.01, float(rms.max()) * 0.05))[0]


def trim(audio: np.ndarray) -> np.ndarray:
    frames = _speech_frames(audio)
    return audio[frames[0] * 160 : (frames[-1] + 1) * 160]


def speech_end_ms(audio: np.ndarray, silero_provider: SileroVADBackendProvider) -> int:
    """End of the last 32 ms frame Silero scores as speech (P >= 0.5), from a fresh state.

    The same reference for both detectors, so trailing breaths or noise that an
    energy threshold would count as speech do not skew either one.
    """
    backend = silero_provider.get()
    state = backend.new_state()
    last = 0
    for index in range(audio.size // 512):
        probability, state = backend.infer(audio[index * 512 : (index + 1) * 512], state)
        if probability >= 0.5:
            last = index + 1
    return last * 32


def b64(samples: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(samples, dtype="<f4").tobytes()).decode()


# --------------------------------------------------------------------------- #
# Measurements                                                                #
# --------------------------------------------------------------------------- #


def stats_ms(samples_s: list[float]) -> dict[str, float]:
    values = sorted(v * 1000 for v in samples_s)

    def q(p: float) -> float:
        k = (len(values) - 1) * p
        lo, hi = int(np.floor(k)), int(np.ceil(k))
        return values[lo] + (values[hi] - values[lo]) * (k - lo)

    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 3),
        "p50": round(q(0.5), 3),
        "p95": round(q(0.95), 3),
        "p99": round(q(0.99), 3),
    }


def timed(fn, iterations: int, warmup: int = 5) -> list[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        out.append(time.perf_counter() - start)
    return out


def rss_mb() -> float:
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def bench_models(silero, smart_turn, speech: np.ndarray, iterations: int) -> dict[str, dict]:
    frame = speech[:512].copy()
    state = silero.new_state()
    results = {"silero_frame_32ms": stats_ms(timed(lambda: silero.infer(frame, state), iterations * 10))}
    for seconds in (1, 4, 8):
        clip = np.resize(speech, seconds * RATE).astype(np.float32)
        features = smart_turn.features(clip)
        results[f"smart_turn_features_{seconds}s"] = stats_ms(timed(lambda: smart_turn.features(clip), iterations))
        results[f"smart_turn_onnx_{seconds}s"] = stats_ms(timed(lambda: smart_turn.score(features), iterations))
        results[f"smart_turn_total_{seconds}s"] = stats_ms(timed(lambda: smart_turn.predict(clip), iterations))
    return results


def bench_concurrency(smart_turn, speech: np.ndarray, iterations: int) -> dict[str, dict]:
    clip = np.resize(speech, 4 * RATE).astype(np.float32)
    results = {}
    for workers in (1, 4, 8):
        latencies: list[float] = []

        def call() -> None:
            start = time.perf_counter()
            smart_turn.predict(clip)
            latencies.append(time.perf_counter() - start)

        with ThreadPoolExecutor(workers) as pool:
            start = time.perf_counter()
            list(pool.map(lambda _: call(), range(iterations)))
            elapsed = time.perf_counter() - start
        results[f"{workers}_concurrent"] = {**stats_ms(latencies), "throughput_per_s": round(iterations / elapsed, 1)}
    return results


def _detector(kind: str, silero_provider, smart_turn_provider):
    if kind == "server_vad":
        config = TurnDetectionConfig.from_realtime({"type": "server_vad"})
    else:
        config = TurnDetectionConfig.from_realtime({"type": "semantic_vad", "eagerness": "auto"})
    return config.build_detector(silero_provider, smart_turn_provider)


def stream(detector, audio: np.ndarray, chunk_ms: int) -> list[tuple[int, list, float]]:
    """Feed audio then 9 s of silence; return (chunk end ms, decisions, wall seconds)."""
    audio = np.concatenate([np.zeros(RATE // 2, dtype=np.float32), audio, np.zeros(9 * RATE, dtype=np.float32)])
    step = chunk_ms * RATE // 1000
    out = []
    for offset in range(0, audio.size, step):
        payload = b64(audio[offset : offset + step])
        start = time.perf_counter()
        segments = detector.process_segments(payload, fmt="pcm_f32le", sample_rate_hz=RATE)
        out.append(
            ((offset + step) * 1000 // RATE, [segment.result for segment in segments], time.perf_counter() - start)
        )
    return out


def bench_chunk_cost(silero_provider, smart_turn_provider, clips: dict[str, np.ndarray], chunk_ms: int) -> dict:
    results = {}
    for kind in ("server_vad", "semantic_vad"):
        ordinary: list[float] = []
        evaluating: list[float] = []
        for audio in clips.values():
            for _ in range(3):
                detector = _detector(kind, silero_provider, smart_turn_provider)
                for _, decisions, seconds in stream(detector, audio, chunk_ms):
                    target = evaluating if any(r.turn_complete_probability is not None for r in decisions) else ordinary
                    target.append(seconds)
        results[kind] = {"ordinary_chunk": stats_ms(ordinary)}
        if evaluating:
            results[kind]["smart_turn_chunk"] = stats_ms(evaluating)
    return results


def bench_classification(smart_turn, clips: list[dict]) -> list[dict]:
    return [
        {"language": clip["language"], "complete": clip["complete"], "probability": smart_turn.predict(clip["audio"])}
        for clip in clips
    ]


def _endpoint_row(detector, audio: np.ndarray, chunk_ms: int, silero_provider: SileroVADBackendProvider) -> dict:
    end_ms = 500 + speech_end_ms(audio, silero_provider)
    chunks = stream(detector, audio, chunk_ms)
    commits = [t for t, decisions, _ in chunks for result in decisions if result.should_commit]
    final = [t for t in commits if t >= end_ms]
    return {
        "false_cutoffs": sum(1 for t in commits if t < end_ms),
        "endpoint_delay_ms": (final[0] - end_ms) if final else None,
        "smart_turn_calls": sum(
            1 for _, decisions, _ in chunks for result in decisions if result.turn_complete_probability is not None
        ),
    }


def bench_endpoints(
    silero_provider, smart_turn_provider, clips: list[dict], chunk_ms: int, pause_ms: int
) -> list[dict]:
    complete = [clip for clip in clips if clip["complete"]]
    scenarios = [("complete", clip["language"], clip["audio"]) for clip in complete]
    pause = np.zeros(pause_ms * RATE // 1000, dtype=np.float32)
    for index, clip in enumerate(clip for clip in clips if not clip["complete"]):
        same_language = [c for c in complete if c["language"] == clip["language"]] or complete
        continuation = same_language[index % len(same_language)]["audio"]
        scenarios.append(
            ("incomplete -> continue", clip["language"], np.concatenate([clip["audio"], pause, continuation]))
        )
    rows = []
    for scenario, language, audio in scenarios:
        for kind in ("server_vad", "semantic_vad"):
            detector = _detector(kind, silero_provider, smart_turn_provider)
            rows.append(
                {
                    "scenario": scenario,
                    "language": language,
                    "mode": kind,
                    **_endpoint_row(detector, audio, chunk_ms, silero_provider),
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# Report                                                                      #
# --------------------------------------------------------------------------- #


def _q(values: list[int], p: float) -> str:
    if not values:
        return "-"
    values = sorted(values)
    return str(values[min(len(values) - 1, int(round((len(values) - 1) * p)))])


def markdown(report: dict) -> str:
    lines = [f"Host: {report['host']}", ""]
    lines += ["**Per-call model latency (single thread, ms)**", "", "| call | P50 | P95 | P99 |", "|---|---|---|---|"]
    for name, s in report["models"].items():
        lines.append(f"| {name} | {s['p50']:.2f} | {s['p95']:.2f} | {s['p99']:.2f} |")
    lines += [
        "",
        "**Smart Turn under concurrency (4 s turn, ms)**",
        "",
        "| sessions | P50 | P95 | calls/s |",
        "|---|---|---|---|",
    ]
    for name, s in report["concurrency"].items():
        lines.append(f"| {name.split('_')[0]} | {s['p50']:.2f} | {s['p95']:.2f} | {s['throughput_per_s']} |")
    lines += [
        "",
        f"**Detector cost per {report['chunk_ms']} ms chunk (ms)**",
        "",
        "| detector | chunk | P50 | P95 | P99 |",
        "|---|---|---|---|---|",
    ]
    for kind, entry in report["chunk_cost"].items():
        for chunk, s in entry.items():
            lines.append(f"| {kind} | {chunk} | {s['p50']:.2f} | {s['p95']:.2f} | {s['p99']:.2f} |")
    lines += [
        "",
        "**Smart Turn classification of labeled clips (threshold 0.5)**",
        "",
        "| language | label | clips | accuracy | median P(complete) |",
        "|---|---|---|---|---|",
    ]
    for language in sorted({r["language"] for r in report["classification"]}):
        for complete in (True, False):
            rows = [r for r in report["classification"] if r["language"] == language and r["complete"] == complete]
            if not rows:
                continue
            correct = sum((r["probability"] >= 0.5) == complete for r in rows)
            lines.append(
                f"| {language} | {'complete' if complete else 'incomplete'} | {len(rows)} | "
                f"{100 * correct / len(rows):.0f}% | {statistics.median(r['probability'] for r in rows):.3f} |"
            )
    lines += [
        "",
        f"**Streaming endpoint decisions (audio time, {report['chunk_ms']} ms chunks, "
        f"{report['pause_ms']} ms thinking pause, semantic_vad eagerness=auto)**",
        "",
        "| scenario | language | mode | runs | runs with a false cutoff "
        "| endpoint delay P50 / P90 (ms) | silence-limit commits |",
        "|---|---|---|---|---|---|---|",
    ]
    for scenario in ("complete", "incomplete -> continue"):
        for language in sorted({r["language"] for r in report["endpoints"]}):
            for kind in ("server_vad", "semantic_vad"):
                rows = [
                    r
                    for r in report["endpoints"]
                    if r["scenario"] == scenario and r["language"] == language and r["mode"] == kind
                ]
                if not rows:
                    continue
                delays = [r["endpoint_delay_ms"] for r in rows if r["endpoint_delay_ms"] is not None]
                cut = sum(1 for r in rows if r["false_cutoffs"])
                fallback = sum(1 for d in delays if d >= 3900) if kind == "semantic_vad" else "-"
                lines.append(
                    f"| {scenario} | {language} | {kind} | {len(rows)} | {cut} ({100 * cut / len(rows):.0f}%) | "
                    f"{_q(delays, 0.5)} / {_q(delays, 0.9)} | {fallback} |"
                )
    lines += ["", "**Resident memory added by loading (MiB)**", "", "| model | RSS delta |", "|---|---|"]
    for name, value in report["memory_mb"].items():
        lines.append(f"| {name} | {value:.1f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=None, help='JSON list of {"path", "complete", "language"}')
    parser.add_argument("--wav", nargs="*", default=[], help="extra clips, counted as complete (read with soundfile)")
    parser.add_argument("--silero-model", default=None, help="Silero ONNX path (default: pinned HF cache file)")
    parser.add_argument("--smart-turn-model", default=None, help="Smart Turn ONNX path (default: pinned HF cache file)")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--pause-ms", type=int, default=1000, help="thinking pause after an incomplete utterance")
    parser.add_argument("--json", default=None, help="write the raw report here")
    parser.add_argument("--markdown", action="store_true", help="print markdown tables")
    args = parser.parse_args()

    base_rss = rss_mb()
    silero_provider = SileroVADBackendProvider(model_path=args.silero_model)
    silero = silero_provider.get()
    silero_rss = rss_mb()
    smart_turn_provider = SmartTurnBackendProvider(model_path=args.smart_turn_model)
    smart_turn = smart_turn_provider.get()
    smart_turn_rss = rss_mb()

    entries = json.loads(Path(args.manifest).read_text()) if args.manifest else []
    entries += [{"path": w, "complete": True, "language": "unlabeled"} for w in args.wav]
    if not entries:
        parser.error("pass --manifest and/or --wav")
    clips = [{**entry, "audio": trim(load_16k(Path(entry["path"])))} for entry in entries]
    reference = next(clip["audio"] for clip in clips if clip["complete"])
    cost_clips = {str(i): clip["audio"] for i, clip in enumerate(clips[:: max(1, len(clips) // 8)])}

    report = {
        "host": f"{os.cpu_count()} CPUs, onnxruntime intra_op=1",
        "chunk_ms": args.chunk_ms,
        "pause_ms": args.pause_ms,
        "models": bench_models(silero, smart_turn, reference, args.iterations),
        "concurrency": bench_concurrency(smart_turn, reference, args.iterations),
        "chunk_cost": bench_chunk_cost(silero_provider, smart_turn_provider, cost_clips, args.chunk_ms),
        "classification": bench_classification(smart_turn, clips),
        "endpoints": bench_endpoints(silero_provider, smart_turn_provider, clips, args.chunk_ms, args.pause_ms),
        "memory_mb": {"silero": silero_rss - base_rss, "smart_turn": smart_turn_rss - silero_rss},
    }
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
    print(markdown(report) if args.markdown else json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
