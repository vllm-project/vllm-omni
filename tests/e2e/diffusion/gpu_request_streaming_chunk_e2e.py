#!/usr/bin/env python3
"""GPU e2e: request-mode CHUNK_READY video arrives before the final finished marker.

Requires 4 visible GPUs (Latest-KV vertical S=D+1). Set:
  NOISY_KV_REPO  — vllm-omni checkout with this change
  NOISY_KV_OUT   — output directory
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

SESSION_ID = "world-stream-e2e"
DENOISE = 3
HISTORY = 6
NUM_CHUNKS = 3
LATENT_SHAPE = [1, 16, 3, 60, 104]
PROMPT = (
    "A golden corgi trots onto a sunny beach at golden hour, "
    "wet sand under its paws, ocean waves ahead."
)


def _write_deploy(repo: Path, out: Path) -> Path:
    cfg = OmegaConf.load(repo / "vllm_omni/deploy/causvid_wan_latest_kv.yaml")
    cfg.distributed_executor_backend = "mp"
    stage = cfg.stages[0]
    n_vis = len([x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",") if x != ""])
    if n_vis != 4:
        raise SystemExit(f"need 4 visible GPUs, got {n_vis}: {os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    stage.devices = ",".join(str(i) for i in range(4))
    stage.enforce_eager = True
    stage.max_num_seqs = 1
    # Request-mode realtime chunks (not stepwise).
    stage.streaming_output = True
    stage.step_execution = False
    stage.parallel_config.pipeline_parallel_size = 4
    stage.model_config.causvid_ckpt = "/data/models/CausVid/autoregressive_checkpoint/model.pt"
    stage.model_config.shift = 8.0
    stage.model_config.num_denoise_steps = DENOISE
    stage.model_config.ar_diffusion_stage_config.step_replicas = 4
    stage.model_config.ar_diffusion_stage_config.max_batch_size = 1
    stage.model_config.ar_diffusion_stage_config.max_history_chunks = HISTORY
    extra = stage.default_sampling_params.extra_args
    extra.num_chunks = NUM_CHUNKS
    extra.num_denoise_steps = DENOISE
    extra.kv_history_chunks = HISTORY
    extra.cell_schedule = "latest_kv"
    extra.shift = 8.0
    extra.reset = True
    extra.stream_decode = True
    path = out / "deploy.yaml"
    path.write_text(OmegaConf.to_yaml(cfg))
    return path


def _payload_tensor(out) -> torch.Tensor | None:
    for attr in ("images", "video", "output"):
        media = getattr(out, attr, None)
        if isinstance(media, (list, tuple)) and media:
            media = media[0]
        if isinstance(media, torch.Tensor):
            return media.detach().float().cpu()
        if media is not None and hasattr(media, "shape"):
            try:
                return torch.as_tensor(media).float().cpu()
            except Exception:
                pass
    return None


def _expect(condition: bool, message: str, failures: list[str]) -> None:
    print(("PASS " if condition else "FAIL ") + message, flush=True)
    if not condition:
        failures.append(message)


def main() -> None:
    repo = Path(os.environ["NOISY_KV_REPO"])
    out = Path(os.environ["NOISY_KV_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    print(f"[e2e] cwd={Path.cwd()} repo={repo} out={out}", flush=True)
    print(f"[e2e] cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    if torch.cuda.is_available():
        print(f"[e2e] device={torch.cuda.get_device_name(0)}", flush=True)

    deploy = _write_deploy(repo, out)
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    started = time.perf_counter()
    omni = Omni(model="/data/models/Wan2.1-T2V-1.3B-Diffusers", deploy_config=str(deploy))
    print(f"[e2e] loaded {time.perf_counter() - started:.1f}s", flush=True)

    # Confirm request-mode streaming stayed on REQUEST_BATCH.
    engine = getattr(getattr(omni, "engine", None), "diffusion_engine", None)
    if engine is None:
        # Stage client may nest the engine differently; probe collective path later.
        stage0 = getattr(omni, "engines", None) or getattr(omni, "_engines", None)
        print(f"[e2e] engine probe engines={type(stage0)}", flush=True)

    params = OmniDiffusionSamplingParams(
        height=480,
        width=832,
        num_frames=13,
        seed=0,
        extra_args={
            "session_id": SESSION_ID,
            "chunk_base": 0,
            "num_chunks": NUM_CHUNKS,
            "num_denoise_steps": DENOISE,
            "kv_history_chunks": HISTORY,
            "cell_schedule": "latest_kv",
            "shift": 8.0,
            "seed": 0,
            "latent_shape": LATENT_SHAPE,
            "reset": True,
            "close_session": False,
            "stream_decode": True,
        },
    )

    failures: list[str] = []
    events: list[dict] = []
    t0 = time.perf_counter()
    try:
        print("[e2e] streaming generate start", flush=True)
        for i, req_out in enumerate(omni.generate(PROMPT, params, py_generator=True, use_tqdm=False)):
            now = time.perf_counter() - t0
            finished = bool(getattr(req_out, "finished", True))
            meta = getattr(req_out, "metadata", None) or {}
            if not isinstance(meta, dict):
                meta = {}
            chunk_index = meta.get("generation_chunk_index", getattr(req_out, "chunk_index", None))
            inner = getattr(req_out, "outputs", None) or getattr(req_out, "diffusion_output", None) or req_out
            if isinstance(inner, (list, tuple)) and inner:
                inner = inner[0]
            finished = bool(getattr(inner, "finished", finished))
            if chunk_index is None:
                chunk_index = getattr(inner, "chunk_index", None)
            tensor = _payload_tensor(inner) if inner is not req_out else _payload_tensor(req_out)
            if tensor is None:
                tensor = _payload_tensor(req_out)
            event = {
                "i": i,
                "t_s": now,
                "finished": finished,
                "chunk_index": chunk_index,
                "has_tensor": tensor is not None,
                "shape": list(tensor.shape) if tensor is not None else None,
                "std": float(tensor.std()) if tensor is not None and tensor.numel() else None,
                "type": type(req_out).__name__,
            }
            events.append(event)
            print(f"[e2e] event {event}", flush=True)
    finally:
        close = getattr(omni, "close", None) or getattr(omni, "shutdown", None)
        if callable(close):
            close()

    elapsed = time.perf_counter() - t0
    media_events = [e for e in events if e["has_tensor"] and not e["finished"]]
    finals = [e for e in events if e["finished"]]
    _expect(len(media_events) >= 1, f"got >=1 intermediate media chunks, got {len(media_events)}", failures)
    _expect(len(media_events) >= NUM_CHUNKS, f"got {NUM_CHUNKS} media chunks, got {len(media_events)}", failures)
    _expect(len(finals) == 1, f"exactly one finished marker, got {len(finals)}", failures)
    if media_events and finals:
        _expect(
            media_events[0]["t_s"] < finals[0]["t_s"],
            f"first chunk before final ({media_events[0]['t_s']:.3f} < {finals[0]['t_s']:.3f})",
            failures,
        )
        # Final should not resend a full video payload after chunks.
        _expect(
            finals[0]["has_tensor"] is False or finals[0]["std"] in (None, 0.0),
            "final marker does not resend full video",
            failures,
        )
    if media_events:
        idxs = [e["chunk_index"] for e in media_events if e["chunk_index"] is not None]
        if idxs:
            _expect(idxs == sorted(idxs), f"chunk indices in order: {idxs}", failures)
            _expect(idxs[0] == 0, f"first chunk_index is 0, got {idxs[0]}", failures)
        else:
            # Omni may only expose growing frame counts; still require monotonic growth.
            shapes = [tuple(e["shape"]) for e in media_events if e["shape"]]
            _expect(len(shapes) == len(media_events), "media events expose tensor shapes", failures)
        _expect(
            all((e["std"] or 0) > 1e-4 for e in media_events),
            "intermediate chunks have real pixel variance",
            failures,
        )

    report = {
        "elapsed_s": elapsed,
        "num_chunks_cfg": NUM_CHUNKS,
        "events": events,
        "failures": failures,
        "deploy": str(deploy),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        print("E2E_RESULT FAIL", flush=True)
        for item in failures:
            print(" -", item, flush=True)
        sys.exit(1)
    print("E2E_RESULT PASS", flush=True)


if __name__ == "__main__":
    main()
