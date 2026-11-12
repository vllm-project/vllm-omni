#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU e2e: two independent CausVid requests share one cell-batch forward."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

DENOISE = 3
HISTORY = 6
LATENT_SHAPE = [1, 16, 3, 60, 104]
PROMPTS = (
    "A golden corgi runs on a sunny beach at golden hour.",
    "A red sailboat drifts across calm blue water under clear sky.",
)


def _write_deploy(repo: Path, out: Path) -> Path:
    cfg = OmegaConf.load(repo / "vllm_omni/deploy/causvid_wan_latest_kv.yaml")
    cfg.distributed_executor_backend = "mp"
    stage = cfg.stages[0]
    n_vis = len([x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",") if x != ""])
    if n_vis != 4:
        raise SystemExit(f"need 4 visible GPUs, got {n_vis}")
    stage.devices = ",".join(str(i) for i in range(4))
    stage.enforce_eager = True
    stage.max_num_seqs = 2
    stage.parallel_config.pipeline_parallel_size = 4
    stage.model_config.causvid_ckpt = "/data/models/CausVid/autoregressive_checkpoint/model.pt"
    stage.model_config.shift = 8.0
    stage.model_config.num_denoise_steps = DENOISE
    stage.model_config.ar_diffusion_stage_config.step_replicas = 4
    stage.model_config.ar_diffusion_stage_config.max_batch_size = 2
    stage.model_config.ar_diffusion_stage_config.max_history_chunks = HISTORY
    extra = stage.default_sampling_params.extra_args
    extra.num_chunks = 1
    extra.num_denoise_steps = DENOISE
    extra.kv_history_chunks = HISTORY
    extra.cell_schedule = "latest_kv"
    extra.shift = 8.0
    extra.reset = True
    path = out / "deploy.yaml"
    path.write_text(OmegaConf.to_yaml(cfg))
    return path


def _as_video_tensor(out) -> torch.Tensor:
    media = getattr(out, "images", None)
    if isinstance(media, (list, tuple)) and media:
        media = media[0]
    if media is None:
        media = getattr(out, "video", None)
    if media is None:
        raise RuntimeError(f"no video tensor on {type(out)!r}")
    tensor = media if isinstance(media, torch.Tensor) else torch.as_tensor(media)
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 5:
        tensor = tensor[0]
    if tensor.ndim == 4 and tensor.shape[0] in (1, 3):
        tensor = tensor.permute(1, 2, 3, 0).contiguous()
    return tensor


def _expect(ok: bool, message: str, failures: list[str]) -> None:
    print(("PASS " if ok else "FAIL ") + message, flush=True)
    if not ok:
        failures.append(message)


def main() -> None:
    repo = Path(os.environ["NOISY_KV_REPO"])
    out = Path(os.environ["NOISY_KV_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    print(f"[e2e] repo={repo} out={out} cvd={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    deploy = _write_deploy(repo, out)

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    t_load = time.perf_counter()
    omni = Omni(model="/data/models/Wan2.1-T2V-1.3B-Diffusers", deploy_config=str(deploy))
    print(f"[e2e] loaded {time.perf_counter() - t_load:.1f}s", flush=True)

    params = [
        OmniDiffusionSamplingParams(
            height=480,
            width=832,
            num_frames=13,
            seed=i,
            extra_args={
                "session_id": f"batch-{i}",
                "chunk_base": 0,
                "num_chunks": 1,
                "num_denoise_steps": DENOISE,
                "kv_history_chunks": HISTORY,
                "cell_schedule": "latest_kv",
                "shift": 8.0,
                "seed": i,
                "latent_shape": LATENT_SHAPE,
                "reset": True,
                "close_session": True,
                "stream_decode": False,
            },
        )
        for i in range(len(PROMPTS))
    ]

    failures: list[str] = []
    t0 = time.perf_counter()
    outputs = omni.generate(list(PROMPTS), params)
    elapsed = time.perf_counter() - t0
    print(f"[e2e] generate done in {elapsed:.2f}s n_out={len(outputs)}", flush=True)
    _expect(len(outputs) == 2, f"got 2 outputs, got {len(outputs)}", failures)

    rows = []
    for i, output in enumerate(outputs):
        frames = _as_video_tensor(output)
        std = float(frames.std()) if frames.numel() else 0.0
        row = {
            "i": i,
            "shape": list(frames.shape),
            "std": std,
            "finite": bool(torch.isfinite(frames).all()) if frames.numel() else False,
        }
        rows.append(row)
        print(f"[e2e] out[{i}] {row}", flush=True)
        _expect(row["finite"] and frames.numel() > 10000 and std > 1e-3, f"request {i} real video", failures)

    report = {"elapsed_s": elapsed, "rows": rows, "failures": failures}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    if failures:
        print("E2E_RESULT FAIL", flush=True)
        raise SystemExit(1)
    print("E2E_RESULT PASS", flush=True)


if __name__ == "__main__":
    main()
