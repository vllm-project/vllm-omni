# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Frozen official scheduler and vocoder, measured with identical prepared inputs.

Run in the official environment. Streaming observation reuses its public
TTSVocoderManager in offline_send_result, without changing scheduler/sampling.
This is an offline driver, not a claim to measure the official HTTP server.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main():
    import dac
    import numpy as np
    import soundfile as sf
    import torch
    import zonos2.tokenizer.vocoder as vocoder
    from zonos2.message.tts import TTSSamplingParams
    from zonos2.tts.llm import TTSLLM

    from benchmarks.zonos2.protocol import PARAMS

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dac", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 4, 8), default=1)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cuda = torch.get_device_module("cuda")
    # No download: explicitly supply the exact same local DAC weights.
    vocoder._dac_model = dac.DAC.load(args.dac, strict=True).float().eval().to("cuda:0")
    import socket

    from zonos2.engine.config import EngineConfig

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        distributed_port = reservation.getsockname()[1]
    EngineConfig.distributed_addr = property(lambda config: f"tcp://127.0.0.1:{distributed_port}")
    model = TTSLLM(
        model_path=args.model,
        n_codebooks=9,
        codebook_size=1024,
        text_vocab=519,
        eoa_id=1024,
        audio_pad_id=1025,
        decode_audio=True,
        max_running_req=8,
        cuda_graph_bs=[1, 4, 8],
        memory_ratio=0.72,
        cache_type="naive",
    )
    bundles = [torch.load(path, weights_only=True) for path in sorted(args.inputs.glob("*.pt"))]
    rows, waves, gpu_steps, host_steps, dac_rows = [], [], [], [], []
    active: dict[int, dict] = {}
    original_send = model.offline_send_result
    original_forward = model.engine.forward_batch_tts

    def forward(batch):
        a, b = cuda.Event(enable_timing=True), cuda.Event(enable_timing=True)
        a.record(model.engine.stream)
        start = time.perf_counter()
        result = original_forward(batch)
        b.record(model.engine.stream)
        gpu_steps.append((a, b, batch.size))
        host_steps.append((time.perf_counter() - start) * 1000)
        return result

    model.engine.forward_batch_tts = forward

    original_incremental = model._vocoder._decode_incremental

    def incremental(uid, is_final):
        a, b = cuda.Event(enable_timing=True), cuda.Event(enable_timing=True)
        a.record()
        start = time.perf_counter()
        output = original_incremental(uid, is_final)
        b.record()
        b.synchronize()
        dac_rows.append(
            {"gpu_ms": a.elapsed_time(b), "host_ms": (time.perf_counter() - start) * 1000, "audio_bytes": len(output)}
        )
        return output

    model._vocoder._decode_incremental = incremental

    def send(reply):
        original_send(reply)
        now = time.perf_counter()
        for message in reply.data:
            if active[message.uid]["first_code"] is None:
                active[message.uid]["first_code"] = now
        chunks = model._vocoder.decode_frames(reply.data)
        for message, chunk in zip(reply.data, chunks, strict=True):
            status = active[message.uid]
            now = time.perf_counter()
            status["codes"].append(message.audio_codes)
            if chunk:
                if status["first_audio"] is None:
                    status["first_audio"] = now
                status["audio"].append(chunk)
            if message.finished:
                status.update(end=now, eos_frame=message.eos_frame)

    model.offline_send_result = send
    model.send_result = send

    def wave(batch, labels, round_id):
        nonlocal active
        start = time.perf_counter()
        active = {i: {"first_code": None, "first_audio": None, "codes": [], "audio": []} for i in range(len(batch))}
        sp = TTSSamplingParams(
            temperature=1.15,
            topk=106,
            top_p=0.0,
            min_p=0.18,
            max_tokens=1024,
            repetition_window=50,
            repetition_penalty=1.2,
            repetition_codebooks=8,
            seed=42,
        )
        model.generate(
            [bundle["base_frames"].tolist() for bundle in batch],
            sp,
            decode_audio=False,
            text_normalization=False,
            speaker_embedding=[b["speaker_embedding"] for b in batch],
        )
        end = time.perf_counter()
        outputs = []
        for i, (bundle, label) in enumerate(zip(batch, labels, strict=True)):
            status = active[i]
            audio = np.frombuffer(b"".join(status["audio"]), dtype=np.float32).copy()
            assert len(audio) and np.isfinite(audio).all()
            codes = torch.tensor(status["codes"], dtype=torch.int32)
            target = status["eos_frame"] if status["eos_frame"] is not None else len(codes)
            assert len(audio) == target * 512
            sf.write(args.out / f"{label}.wav", audio, 44100, subtype="FLOAT")
            torch.save(codes, args.out / f"{label}.pt")
            latency = status["end"] - start
            row = {
                "label": label,
                "case": bundle["id"],
                "round": round_id,
                "start": start,
                "end": status["end"],
                "ttfc_s": status["first_code"] - start,
                "ttfp_s": status["first_audio"] - start,
                "latency_s": latency,
                "duration_s": len(audio) / 44100,
                "rtf": latency / (len(audio) / 44100),
                "samples": len(audio),
                "frames": len(codes),
                "eos_frame": status["eos_frame"],
                "reached_cap": len(codes) >= 1024,
            }
            rows.append(row)
            outputs.append(row)
        return {
            "round": round_id,
            "requests": len(batch),
            "elapsed_s": end - start,
            "audio_s": sum(row["duration_s"] for row in outputs),
        }

    try:
        wave(bundles[:1], ["warmup_0"], -1)
        wave(bundles[:1], ["warmup_1"], -1)
        gpu_steps.clear()
        host_steps.clear()
        dac_rows.clear()
        for round_id in range(args.rounds):
            for offset in range(0, len(bundles), args.concurrency):
                batch = bundles[offset : offset + args.concurrency]
                entry = wave(batch, [f"r{round_id}_{b['id']}" for b in batch], round_id)
                entry["offset"] = offset
                waves.append(entry)
        cuda.synchronize()
        report = {
            "backend": "official",
            "driver": "offline scheduler + official streaming vocoder observer",
            "params": PARAMS,
            "concurrency": args.concurrency,
            "rows": rows,
            "waves": waves,
            "ar_gpu_ms": [a.elapsed_time(b) for a, b, n in gpu_steps],
            "ar_host_ms": host_steps,
            "ar_batch_sizes": [n for a, b, n in gpu_steps],
            "dac_timings": dac_rows,
            "torch": torch.__version__,
            "cuda_graph_bs": [1, 4, 8],
            "prefix_cache": "naive",
        }
        (args.out / "result.json").write_text(json.dumps(report, indent=2))
    finally:
        model.shutdown()
    print("OFFICIAL_BENCH_DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
