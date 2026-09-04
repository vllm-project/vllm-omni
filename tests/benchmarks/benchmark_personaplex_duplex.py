# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A/B benchmark driver for the PersonaPlex duplex server.

Streams a 24 kHz mono WAV to ``/api/chat`` at realtime pace (official Moshi
protocol, the same wire path as
``examples/online_serving/personaplex/duplex_client.py``) and measures how much
reply audio has arrived by the time the input finishes. A server that generates
slower than realtime delivers less audio than the input's duration and the
deficit accumulates for the whole call, so delivered seconds (and ``rate``,
delivered divided by the input duration) are the headline numbers.

Start the server once per side, then point the driver at it:

    python -m vllm_omni.experimental.fullduplex.personaplex.serving.server   # eager
    python -m vllm_omni.experimental.fullduplex.personaplex.serving.server --cuda-graphs

    python tests/benchmarks/benchmark_personaplex_duplex.py \
        --url ws://localhost:8124/api/chat --input user_40s.wav --runs 3

Per run: ready latency, reply seconds received inside the input window, the
realtime rate, the largest receive gap, and (with ``--gpu-index``) peak GPU
memory polled from ``nvidia-smi``. The summary prints mean and sample stddev
across runs. Each run's reply audio is saved for offline quality checks such
as whisper transcription.

Deps: ``websockets``, ``sphn``, ``soundfile``, ``numpy``.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import sphn

SR = 24000  # PersonaPlex / Mimi codec rate
FRAME = 1920  # 80 ms at 24 kHz


@dataclass
class RunMetrics:
    ready_s: float
    in_window_out_s: float
    total_out_s: float
    rate: float
    max_gap_s: float
    peak_gpu_mib: int | None


class _GpuPoller:
    """Polls ``nvidia-smi`` for used memory so the bench needs no server hooks."""

    def __init__(self, gpu_index: int, interval_s: float = 0.5):
        self._index = gpu_index
        self._interval = interval_s
        self._stop = threading.Event()
        self.peak_mib = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--id={self._index}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.peak_mib = max(self.peak_mib, int(out.stdout.strip().splitlines()[0]))
            except (subprocess.SubprocessError, ValueError, IndexError):
                pass
            self._stop.wait(self._interval)

    def __enter__(self) -> _GpuPoller:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


async def _run_once(url: str, pcm: np.ndarray, tail_s: float) -> tuple[RunMetrics, np.ndarray, str]:
    import websockets

    reader = sphn.OpusStreamReader(SR)
    writer = sphn.OpusStreamWriter(SR)
    ready = asyncio.Event()
    chunks: list[np.ndarray] = []
    text: list[str] = []
    recv_times: list[float] = []
    last = {"t": 0.0}
    loop = asyncio.get_event_loop()

    async with websockets.connect(url, max_size=None) as ws:

        async def receive() -> None:
            while True:
                try:
                    msg = await ws.recv()
                except Exception:
                    return
                if not isinstance(msg, (bytes, bytearray)) or not msg:
                    continue
                tag, payload = msg[0], msg[1:]
                if tag == 0:
                    ready.set()
                elif tag == 1:
                    # sphn 0.2.x returns PCM straight from append_bytes; 0.1.x
                    # buffers and needs a read_pcm drain (same split server.py handles).
                    out = reader.append_bytes(bytes(payload))
                    if out is None and hasattr(reader, "read_pcm"):
                        out = reader.read_pcm()
                    if out is not None and np.asarray(out).shape[-1] > 0:
                        chunks.append(np.asarray(out, dtype=np.float32))
                        recv_times.append(loop.time())
                    last["t"] = loop.time()
                elif tag == 2:
                    text.append(payload.decode("utf8", errors="replace"))

        recv_task = asyncio.create_task(receive())
        t_connect = loop.time()
        await asyncio.wait_for(ready.wait(), timeout=300)  # system-prompt prefill
        t_ready = loop.time()

        for i in range(0, len(pcm), FRAME):
            chunk = pcm[i : i + FRAME]
            if len(chunk) < FRAME:
                chunk = np.concatenate([chunk, np.zeros(FRAME - len(chunk), dtype=np.float32)])
            data = writer.append_pcm(chunk)
            if data is None and hasattr(writer, "read_bytes"):
                data = writer.read_bytes()
            if data:
                await ws.send(b"\x01" + bytes(data))
            await asyncio.sleep(FRAME / SR)  # realtime pacing
        in_window_samples = int(sum(c.shape[-1] for c in chunks))

        last["t"] = loop.time()
        while loop.time() - last["t"] < tail_s:
            await asyncio.sleep(0.1)
        recv_task.cancel()

    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    gaps = [b - a for a, b in zip(recv_times, recv_times[1:])]
    in_window_out_s = in_window_samples / SR
    return (
        RunMetrics(
            ready_s=t_ready - t_connect,
            in_window_out_s=in_window_out_s,
            total_out_s=audio.shape[-1] / SR,
            rate=in_window_out_s / (len(pcm) / SR),
            max_gap_s=max(gaps) if gaps else 0.0,
            peak_gpu_mib=None,
        ),
        audio,
        "".join(text),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="ws://localhost:8124/api/chat")
    ap.add_argument("--input", required=True, help="24 kHz mono WAV streamed as the user turn")
    ap.add_argument("--runs", type=int, default=3, help="measured runs (after the discarded warmup run)")
    ap.add_argument("--no-warmup-run", action="store_true", help="do not run and discard an unmeasured first run")
    ap.add_argument("--tail", type=float, default=2.5, help="silence (s) that ends the drain after the input")
    ap.add_argument("--gpu-index", type=int, default=None, help="poll nvidia-smi on this GPU for peak memory")
    ap.add_argument("--out-dir", default="pplex_bench_out", help="reply WAVs are written here, one per run")
    args = ap.parse_args()

    wav, sr = sf.read(args.input, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        raise SystemExit(f"input must be {SR} Hz mono, got {sr} Hz (resample it first)")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[RunMetrics] = []
    total_runs = args.runs if args.no_warmup_run else args.runs + 1
    for i in range(total_runs):
        measured = args.no_warmup_run or i > 0
        poller = _GpuPoller(args.gpu_index) if (args.gpu_index is not None and measured) else None
        if poller:
            with poller:
                metrics, audio, reply_text = asyncio.run(_run_once(args.url, wav, args.tail))
            metrics.peak_gpu_mib = poller.peak_mib
        else:
            metrics, audio, reply_text = asyncio.run(_run_once(args.url, wav, args.tail))
        label = f"run {i}" if measured else "warmup (discarded)"
        gpu = f"  peak_gpu={metrics.peak_gpu_mib} MiB" if metrics.peak_gpu_mib is not None else ""
        print(
            f"{label}: ready={metrics.ready_s:.2f}s  out_in_window={metrics.in_window_out_s:.2f}s"
            f"  total_out={metrics.total_out_s:.2f}s  rate={metrics.rate:.3f}x  max_gap={metrics.max_gap_s:.2f}s{gpu}"
        )
        if reply_text:
            print(f"  text: {reply_text[:160]}")
        if measured:
            results.append(metrics)
            sf.write(out_dir / f"reply_run{i}.wav", audio, SR)
            (out_dir / f"reply_run{i}.txt").write_text(reply_text)

    rates = [m.rate for m in results]
    outs = [m.in_window_out_s for m in results]
    stddev = statistics.stdev(rates) if len(rates) > 1 else 0.0
    print(
        f"\nsummary over {len(results)} runs: rate mean={statistics.mean(rates):.3f}x stddev={stddev:.3f}"
        f"  out_in_window mean={statistics.mean(outs):.2f}s"
    )
    peaks = [m.peak_gpu_mib for m in results if m.peak_gpu_mib is not None]
    if peaks:
        print(f"peak GPU memory across runs: {max(peaks)} MiB")


if __name__ == "__main__":
    main()
