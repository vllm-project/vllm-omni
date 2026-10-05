# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark H3 forward-input preparation, excluding model/serving latency."""

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch

from vllm_omni.diffusion.models.minimax_h3.denoise_loop import MiniMaxH3DenoiseBranch
from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence_ref2va_blocks
from vllm_omni.platforms import current_omni_platform


def main() -> None:
    if not current_omni_platform.is_cuda():
        raise SystemExit("H3 timestep preparation microbench requires CUDA.")
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location("vllm_omni.diffusion.models.minimax_h3.baseline_denoise", a.baseline)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the baseline module")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    methods = {
        "baseline": baseline.MiniMaxH3DenoiseBranch.forward_kwargs,
        "candidate": MiniMaxH3DenoiseBranch.forward_kwargs,
    }
    torch.set_num_threads(4)
    report = {
        "torch": torch.__version__,
        "gpu": current_omni_platform.get_device_name(),
        "scope": "H3 forward_kwargs only, real CUDA; no DiT weights or serving throughput",
        "cases": [],
    }
    for latent_t, latent_h, latent_w in [(4, 32, 48), (31, 32, 48), (31, 96, 168), (76, 32, 48)]:
        packed = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=132,
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=1200,
            ref_blocks=[
                {"kind": "image", "latent_h": latent_h, "latent_w": latent_w},
                {"kind": "audio", "ref_audio_t": 100},
            ],
        )
        branch = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=torch.zeros(132, 8),
            token_tags=packed["token_tags"],
            device=current_omni_platform.get_torch_device(),
        )
        baseline_branch = baseline.MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=torch.zeros(132, 8),
            token_tags=packed["token_tags"],
            device=current_omni_platform.get_torch_device(),
        )
        branches = {"baseline": baseline_branch, "candidate": branch}
        kwargs = dict(t_video=0.31, t_audio=0.57, imgvid_cond_timestep=0.999, audio_ref_cond_timestep=1.0)
        kwargs.update(
            video_rows=torch.zeros(len(branch.img_pos), 96, device=branch.device),
            audio_rows=torch.zeros(len(branch.audio_pos), 32, device=branch.device),
        )
        expected = methods["baseline"](baseline_branch, **kwargs)
        actual = methods["candidate"](branch, **kwargs)
        for key in ("x", "audio_x", "unique_timesteps", "inverse_indices"):
            assert torch.equal(expected[key], actual[key]), key
        records = []
        for arm in ["baseline", "candidate", "candidate", "baseline"]:
            fn = methods[arm]
            for _ in range(20):
                fn(branches[arm], **kwargs)
            times = []
            for _ in range(30):
                current_omni_platform.synchronize()
                start = time.perf_counter()
                for _ in range(8):
                    fn(branches[arm], **kwargs)
                current_omni_platform.synchronize()
                times.append((time.perf_counter() - start) * 1000 / 8)
            records.append({"arm": arm, "times_ms": times, "median_ms": statistics.median(times)})
        # Tracing is outside the formal timing interval.
        counts = {}
        for arm, fn in methods.items():
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                fn(branches[arm], **kwargs)
                current_omni_platform.synchronize()
            counts[arm] = [
                {"key": e.key, "count": e.count}
                for e in prof.key_averages()
                if "nonzero" in e.key or "Synchronize" in e.key
            ]
        report["cases"].append(
            {
                "sequence": branch.seq_len,
                "latent_t": latent_t,
                "exact": True,
                "records": records,
                "profile_counts": counts,
            }
        )
        Path(a.output).write_text(json.dumps(report, indent=2))
        print(branch.seq_len, [(r["arm"], r["median_ms"]) for r in records], counts, flush=True)


if __name__ == "__main__":
    main()
