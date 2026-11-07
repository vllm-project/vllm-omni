# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Isolated DAC eager/compile/CUDA-graph A/B; replay actual codes for quality.

These are component probes, not production AR graph support. They preserve
sampling and OLA. Compile warmup/capture time is excluded and recorded.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main():
    import numpy as np
    import soundfile as sf
    import torch

    from benchmarks.zonos2.protocol import PARAMS, quantiles_ci
    from vllm_omni.model_executor.models.zonos2.zonos2_codec import DACStreamDecoder, LocalDAC

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", choices=("eager", "compile", "graph"), required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cuda = torch.get_device_module("cuda")
    dac = LocalDAC(device="cuda:0")
    codec = dac.load()

    def raw(codes):
        latent = codec.quantizer.from_codes(codes.clamp(0, 1023))[0]
        return codec.decode(latent).float()

    function = raw
    if args.variant == "compile":
        function = torch.compile(raw, fullgraph=False, dynamic=True, mode="default")
    graphs = {}
    setup = []

    @torch.inference_mode()
    def gpu(codes):
        if args.variant != "graph":
            return function(codes)
        frames = codes.shape[-1]
        if frames not in graphs:
            start = time.perf_counter()
            static = codes.clone()
            stream = cuda.Stream()
            with cuda.stream(stream):
                for _ in range(3):
                    raw(static)
            stream.synchronize()
            graph = cuda.CUDAGraph()
            with cuda.graph(graph, stream=stream):
                result = raw(static)
            graphs[frames] = (static, graph, result)
            setup.append({"frames": frames, "capture_s": time.perf_counter() - start})
        static, graph, result = graphs[frames]
        static.copy_(codes)
        graph.replay()
        return result

    @torch.inference_mode()
    def decode(codes):
        if codes.shape[-1] == 0:
            return torch.empty(0, dtype=torch.float32)
        output = gpu(codes.to(device="cuda:0", dtype=torch.long).unsqueeze(0))
        return output.squeeze(0).squeeze(0).cpu().clone()

    try:
        warm_start = time.perf_counter()
        for frames in (16, 20):
            codes = torch.arange(9 * frames, device="cuda:0").reshape(1, 9, frames) % 1024
            for _ in range(4):
                gpu(codes)
        cuda.synchronize()
        warmup_s = time.perf_counter() - warm_start
        timing = {}
        for frames in (16, 20):
            codes = torch.arange(9 * frames, device="cuda:0").reshape(1, 9, frames) % 1024
            wall, device = [], []
            for _ in range(100):
                a, b = cuda.Event(enable_timing=True), cuda.Event(enable_timing=True)
                a.record()
                start = time.perf_counter()
                gpu(codes)
                b.record()
                b.synchronize()
                wall.append((time.perf_counter() - start) * 1000)
                device.append(a.elapsed_time(b))
            timing[str(frames)] = {"wall_ms": quantiles_ci(wall), "device_interval_ms": quantiles_ci(device)}
        reference = json.loads((args.source / "result.json").read_text())
        rows = []
        for row in (item for item in reference["rows"] if item["round"] == 0):
            codes = torch.load(args.source / "record" / f"{row['request']}.pt", weights_only=True).long()
            boundary = row["eos_frame"] if row["eos_frame"] >= 0 else len(codes)
            stream = DACStreamDecoder(decode)
            pieces = []
            for end in range(24, len(codes), 16):
                target = min(end - 8, boundary)
                pieces.append(stream.push(row["case"], codes[:end], final=False, target=target, sequence=end))
            pieces.append(stream.push(row["case"], codes, final=True, target=boundary, sequence=len(codes) + 1))
            audio = torch.cat(pieces).numpy()
            baseline, rate = sf.read(args.source / f"{row['label']}.wav", dtype="float32")
            assert len(audio) == len(baseline) and np.isfinite(audio).all()
            error = float(np.abs(audio - baseline).max())
            mse = float(np.mean((audio.astype(np.float64) - baseline) ** 2))
            snr = float(10 * np.log10(np.mean(baseline.astype(np.float64) ** 2) / max(mse, 1e-30)))
            sf.write(args.out / f"{row['label']}.wav", audio, rate, subtype="FLOAT")
            rows.append(
                {**row, "max_wave_error": error, "wave_snr_db": snr, "numeric_gate_pass": error <= 1e-4 and snr >= 60}
            )
            stream.cleanup([row["case"]])
            assert not stream.states and not stream.closed
        (args.out / "result.json").write_text(
            json.dumps(
                {
                    "backend": f"dac_{args.variant}",
                    "variant": args.variant,
                    "scope": "DAC component only; actual native codes, unchanged OLA",
                    "params": PARAMS,
                    "rows": rows,
                    "timing": timing,
                    "warmup_s": warmup_s,
                    "graph_setup": setup,
                },
                indent=2,
            )
        )
        print("DAC_AB_DONE", args.variant, warmup_s, flush=True)
    except Exception as error:
        (args.out / "failure.json").write_text(
            json.dumps(
                {
                    "variant": args.variant,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "fallback_used": False,
                },
                indent=2,
            )
        )
        raise


if __name__ == "__main__":
    main()
