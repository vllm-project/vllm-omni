# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pi0.5 S0 four-layer benchmark: model, pipeline, engine, and OpenPI E2E."""

from __future__ import annotations

import argparse
import copy
import json
import time
import uuid

import numpy as np
import torch

from vllm_omni.platforms import current_omni_platform

CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _robot_obs(views: int) -> dict:
    obs = {f"observation.images.{camera}": np.zeros((224, 224, 3), dtype=np.uint8) for camera in CAMERAS[:views]}
    obs["state"] = np.zeros(32, dtype=np.float32)
    return obs


def _summary(samples: list[float]) -> dict:
    values = torch.tensor(samples, dtype=torch.float64)
    return {
        "n": len(samples),
        "p50_ms": float(torch.quantile(values, 0.5)),
        "p95_ms": float(torch.quantile(values, 0.95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def _time_cuda(fn, iters: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    current_omni_platform.synchronize()
    samples = []
    for _ in range(iters):
        started = time.perf_counter()
        fn()
        current_omni_platform.synchronize()
        samples.append((time.perf_counter() - started) * 1e3)
    return samples


def _benchmark_engine(args: argparse.Namespace) -> None:
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    kwargs = {
        "model": args.model,
        "deploy_config": args.deploy_config,
        "enforce_eager": True,
        "log_stats": True,
    }
    if args.backend == "uni":
        kwargs["distributed_executor_backend"] = "uni"
    omni = Omni(**kwargs)
    try:
        pipeline = None
        if args.backend == "uni":
            client = omni.engine.stage_clients[0]
            pipeline = client._engine.executor.driver_worker.worker.model_runner.pipeline

        prompt = "pick up the red block and place it in the bin"
        results = {}
        raw_rows = []
        for views in args.views:
            robot_obs = _robot_obs(views)
            if args.backend == "uni":
                images, image_masks, lang_tokens, lang_masks = pipeline.processor.build_model_inputs(robot_obs)
                noise_generator = torch.Generator(device="cpu").manual_seed(42)
                noise = torch.randn(
                    (1, pipeline.config.chunk_size, pipeline.config.action_dim),
                    generator=noise_generator,
                    dtype=torch.float32,
                ).to(pipeline._device, pipeline._torch_dtype)
                sampling_generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(42)
                params = OmniDiffusionSamplingParams(
                    num_inference_steps=args.steps,
                    generator=sampling_generator,
                    extra_args={"robot_obs": robot_obs},
                )

                def model_only() -> None:
                    with torch.no_grad():
                        pipeline.model.sample_actions(
                            images=images,
                            image_masks=image_masks,
                            lang_tokens=lang_tokens,
                            lang_masks=lang_masks,
                            noise=noise.clone(),
                            num_steps=args.steps,
                        )

                def in_process_pipeline() -> None:
                    from vllm_omni.diffusion.request import OmniDiffusionRequest

                    request = OmniDiffusionRequest(
                        prompt=prompt,
                        sampling_params=copy.deepcopy(params),
                        request_id="bench-uni-pipeline",
                    )
                    with torch.no_grad():
                        pipeline.forward(request)
            else:
                params = OmniDiffusionSamplingParams(
                    num_inference_steps=args.steps,
                    seed=42,
                    extra_args={"robot_obs": robot_obs},
                )

            request_segments = []

            def engine_call():
                outputs = omni.generate(
                    prompt,
                    sampling_params_list=[copy.deepcopy(params)],
                    use_tqdm=False,
                )
                if outputs:
                    output = outputs[-1]
                    request_segments.append(
                        {
                            "request_id": getattr(output, "request_id", None),
                            "stage_durations": dict(getattr(output, "stage_durations", {}) or {}),
                            "metrics": dict(getattr(output, "metrics", {}) or {}),
                        }
                    )
                return outputs

            outputs = []
            for _ in range(args.warmup):
                outputs = engine_call()
            current_omni_platform.synchronize()
            engine_latencies = []
            segments = []
            for index in range(args.iters):
                before = len(request_segments)
                started = time.perf_counter()
                outputs = engine_call()
                current_omni_platform.synchronize()
                latency = (time.perf_counter() - started) * 1e3
                engine_latencies.append(latency)
                if len(request_segments) > before:
                    segment = request_segments[-1]
                    segments.append(segment)
                    raw_rows.append(
                        {
                            "views": views,
                            "iteration": index,
                            "layer": f"engine_{args.backend}_ms",
                            "latency_ms": latency,
                            **segment,
                        }
                    )

            layers = {f"engine_{args.backend}_ms": _summary(engine_latencies)}
            if args.backend == "uni":
                layers["model_only_ms"] = _summary(_time_cuda(model_only, args.iters, args.warmup))
                layers["pipeline_ms"] = _summary(_time_cuda(in_process_pipeline, args.iters, args.warmup))
            final_output = outputs[-1] if outputs else None
            results[str(views)] = {
                "backend": args.backend,
                "views": views,
                "steps": args.steps,
                "layers": layers,
                "request_segments": segments,
                "stage_durations": dict(getattr(final_output, "stage_durations", {}) or {}),
                "peak_memory_mb": getattr(final_output, "peak_memory_mb", 0.0),
            }
            if pipeline is not None:
                results[str(views)]["dtype"] = str(pipeline._torch_dtype)

        result = {"backend": args.backend, "views": results}
        rendered = json.dumps(result, indent=2)
        print(rendered)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as handle:
                handle.write(rendered + "\n")
        if args.raw_out:
            with open(args.raw_out, "w", encoding="utf-8") as handle:
                for row in raw_rows:
                    handle.write(json.dumps(row) + "\n")
    finally:
        omni.close()


def _benchmark_e2e(args: argparse.Namespace) -> None:
    import websockets.sync.client as websockets_client
    from openpi_client import msgpack_numpy

    from tests.helpers.runtime import (
        PI0_CAMERA_KEYS,
        PI0_OPENPI_PATH,
        _pi0_decode_action_response,
        pi0_make_dummy_obs,
    )

    uri = f"ws://{args.host}:{args.port}{PI0_OPENPI_PATH}"
    packer = msgpack_numpy.Packer()
    conn = websockets_client.connect(uri, compression=None, max_size=None, ping_interval=300, ping_timeout=3600)
    try:
        metadata = msgpack_numpy.unpackb(conn.recv())
        summary = {"metadata": metadata, "views": {}}
        for views in (1, 2, 3):
            session_id = f"pi05-e2e-{views}view-{uuid.uuid4().hex[:8]}"
            latencies = []
            action_means = []
            for _ in range(args.requests):
                observation = pi0_make_dummy_obs(prompt="pick up the red block", session_id=session_id)
                for camera in PI0_CAMERA_KEYS[views:]:
                    observation.pop(camera, None)
                observation["seed"] = 42
                observation["sampling_params"] = {"num_inference_steps": args.steps}
                observation["endpoint"] = "infer"

                started = time.perf_counter()
                conn.send(packer.pack(observation))
                actions = np.asarray(_pi0_decode_action_response(conn.recv()), dtype=np.float32)
                latencies.append(time.perf_counter() - started)
                if actions.shape != (50, 32):
                    raise AssertionError(f"views={views} shape={actions.shape}")
                if not np.isfinite(actions).all():
                    raise AssertionError(f"views={views} non-finite actions")
                action_means.append(float(actions.mean()))

            summary["views"][str(views)] = {
                "requests": args.requests,
                "num_inference_steps": args.steps,
                "latency_s": latencies,
                "mean_latency_s": float(np.mean(latencies)),
                "action_mean": float(np.mean(action_means)),
                "action_shape": [50, 32],
            }
            print(json.dumps(summary["views"][str(views)]), flush=True)
        print(json.dumps(summary, indent=2))
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("uni", "mp", "e2e"), required=True)
    parser.add_argument("--model", default="lerobot/pi05_base")
    parser.add_argument("--deploy-config", default="vllm_omni/deploy/pi05.yaml")
    parser.add_argument("--views", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--out")
    parser.add_argument("--raw-out")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--requests", type=int, default=30)
    args = parser.parse_args()
    args.backend = args.mode
    if args.mode == "e2e":
        _benchmark_e2e(args)
    else:
        _benchmark_engine(args)


if __name__ == "__main__":
    main()
