# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ABot evidence: public typed sessions, stepwise, and offline controls.

Run each phase in a fresh process. Baselines do not enable a torch profiler or
stage synchronizations. RPC snapshots run outside measured next_chunk calls.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path


class ReviewEvidenceWorker:
    def review_snapshot(self):
        import torch

        runner = self.model_runner
        pipeline = runner.pipeline
        cache = getattr(runner, "kv_cache", None)
        states = pipeline._streaming_decode_states
        if pipeline._vae_backend == "wan":
            from vllm_omni.diffusion.models.abot_world.pipeline_abot_world import _wan_decode_cache_bytes

            vae_budget = _wan_decode_cache_bytes(pipeline.vae.decoder, 32, 52, pipeline.vae.dtype)
        else:
            vae_budget = pipeline.vae.persistent_state_bytes(32, 52, pipeline.vae.dtype)
        return {
            "budget_ok": all(state.nbytes() <= vae_budget for state in states.values()),
            "gpu": torch.cuda.get_device_name(),
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "allocated_bytes": torch.accelerator.memory_allocated(),
            "reserved_bytes": torch.accelerator.memory_reserved(),
            "peak_allocated_bytes": torch.accelerator.max_memory_allocated(),
            "decode_states": {
                key: {
                    "bytes": state.nbytes(),
                    "latent_frames": state.frames_decoded,
                    "chunks": state.chunks_decoded,
                    "cache_dtypes": sorted({str(x.dtype) for x in state.feat_map if torch.is_tensor(x)}),
                    "cache_tensors": [
                        {"shape": list(x.shape), "dtype": str(x.dtype), "bytes": x.numel() * x.element_size()}
                        for x in state.feat_map
                        if torch.is_tensor(x)
                    ],
                }
                for key, state in states.items()
            },
            "vae_state_budget_bytes_per_session": vae_budget,
            "model_owned_state_bytes_per_session": (
                pipeline.ar_diffusion_kv_cache_spec().model_owned_state_bytes_per_session
            ),
            "pool": None
            if cache is None
            else {
                key: getattr(cache, key)
                for key in (
                    "memory_budget_bytes",
                    "model_owned_state_reserved_bytes",
                    "session_capacity",
                    "num_blocks_total",
                    "scratch_reserved_bytes",
                    "cross_attention_reserved_bytes",
                )
            },
            "pool_storages": []
            if cache is None
            else [
                [tensor.data_ptr(), tensor.untyped_storage().nbytes()]
                for layer_pools in cache._kv_pools
                for tensor in layer_pools
            ],
        }


PROMPT = "The camera moves slowly forward through the scene."


def frames_of(output):
    import torch

    if output.error:
        raise RuntimeError(output.error)
    values = output.images
    if not values:
        values = output.multimodal_output.get("video")
    if isinstance(values, list):
        assert len(values) == 1, (len(values), output.multimodal_output.keys())
        values = values[0]
    assert isinstance(values, torch.Tensor), type(values)
    if values.ndim == 5:
        assert values.shape[0] == 1
        values = values[0]
    assert values.ndim == 4, values.shape
    if values.shape[1] == 3:
        return values.cpu()  # TCHW
    if values.shape[0] == 3:
        return values.permute(1, 0, 2, 3).cpu()
    raise AssertionError(values.shape)


def summarize(rows):
    values = sorted(row["wall_ms"] for row in rows[1:])
    return {
        "first_chunk_ms": rows[0]["wall_ms"],
        "steady_median_ms": statistics.median(values) if values else None,
        "steady_mean_ms": statistics.mean(values) if values else None,
        "steady_generated_fps": 12000 / statistics.mean(values) if values else None,
        "steady_p95_ms": values[min(len(values) - 1, int(len(values) * 0.95))] if values else None,
        "max_tick_ms": max(row["wall_ms"] for row in rows),
        "max_tick_index": max(rows, key=lambda row: row["wall_ms"])["tick"],
        "sum_tick_wall_ms": sum(row["wall_ms"] for row in rows),
    }


def validate_frames(frames, index):
    import torch

    assert frames.shape == (9 if index == 0 else 12, 3, 512, 832), frames.shape
    assert torch.isfinite(frames).all(), f"nonfinite tick {index}"
    assert frames.min() >= 0 and frames.max() <= 1
    return {"shape": list(frames.shape), "mean": frames.mean().item(), "std": frames.std().item()}


async def main(args):
    import torch

    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    result = {
        "args": vars(args),
        "status": "running",
        "torch": torch.__version__,
        "measurement": "CPU-ready video; includes request/compute/decode/return; excludes MP4/network transport",
        "enforce_eager": True,
        "profiler": False,
        "max_num_seqs": 1,
        "snapshots": {},
        "events": {
            "cuda_graph_capture": "disabled: enforce_eager=True, warmup_cudagraph=False",
            "kv_pool_growth": "compare real fixed pool storage identities and sizes; page reuse is not growth",
        },
    }
    import vllm_omni

    root = Path(vllm_omni.__file__).parent
    result["source_hashes"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for pattern in ("diffusion/models/abot_world/*.py", "experimental/ar_diffusion/**/*.py")
        for path in root.glob(pattern)
    }
    result["input_sha256"] = hashlib.sha256(Path(args.image).read_bytes()).hexdigest()
    engine = None
    manager = None
    active = set()

    def save():
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    def sampling(ticks=1):
        return OmniDiffusionSamplingParams(
            height=512,
            width=832,
            num_frames=12 * ticks - 3,
            num_inference_steps=4,
            max_sequence_length=512,
            seed=42,
            output_type="pt",
            extra_args={"flow_shift": 5.0},
        )

    async def snapshot(label):
        values = await engine.collective_rpc("review_snapshot", stage_ids=[0], timeout=120)
        result["snapshots"][label] = values
        save()

        def payloads(obj):
            if isinstance(obj, dict):
                if "budget_ok" in obj:
                    yield obj
                else:
                    for value in obj.values():
                        yield from payloads(value)
            elif isinstance(obj, list):
                for value in obj:
                    yield from payloads(value)

        worker_values = list(payloads(values))
        assert worker_values, f"No successful worker snapshot: {values}"
        assert all(value["budget_ok"] for value in worker_values), f"VAE budget underestimated: {values}"
        initial = list(payloads(result["snapshots"]["engine_ready"]))
        assert [v["pool_storages"] for v in worker_values] == [v["pool_storages"] for v in initial], (
            f"KV pool storage grew or moved at {label}"
        )
        if label.endswith("closed") or label.startswith("after_run_"):
            assert all(not value["decode_states"] for value in worker_values), f"VAE state leaked: {label}"
        return values

    try:
        kwargs = {
            "model": args.model,
            "model_class_name": "ABotWorldCausalPipeline",
            "enforce_eager": True,
            "tensor_parallel_size": 1,
            "max_num_seqs": 1,
            "worker_extension_cls": "benchmarks.diffusion.abot_review_evidence.ReviewEvidenceWorker",
            "model_config": {
                "abot_vae": args.vae,
                "ar_diffusion_height": 512,
                "ar_diffusion_width": 832,
                "ar_diffusion_kv_config": {"gpu_memory_fraction": 0.15, "warmup_cudagraph": False},
            },
        }
        if args.phase != "offline":
            kwargs["engine_backend"] = "vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine"
        if args.phase == "stepwise":
            kwargs.update(step_execution=True, diffusion_streaming_output=True)
        engine = AsyncOmni(**kwargs)
        await snapshot("engine_ready")
        prompt = {"prompt": PROMPT, "multi_modal_data": {"image": args.image}}
        if args.phase in {"stepwise", "offline"}:
            ticks = args.ticks if args.phase == "stepwise" else 10
            result["runs"] = []
            for rep in range(args.repeats + 1):
                start = last = time.perf_counter()
                deliveries = []
                async for output in engine.generate(prompt, sampling(ticks), request_id=f"{args.phase}-{rep}"):
                    if not output.images and output.multimodal_output.get("video") is None:
                        continue
                    frames = frames_of(output)
                    now = time.perf_counter()
                    deliveries.append((frames, (now - last) * 1000))
                    last = now
                total_ms = (time.perf_counter() - start) * 1000
                if args.phase == "offline":
                    assert len(deliveries) == 1
                    clip = deliveries[0][0]
                    assert clip.shape == (117, 3, 512, 832), clip.shape
                    assert torch.isfinite(clip).all()
                    assert clip.min() >= 0 and clip.max() <= 1
                    torch.save(clip, out / f"clip-{rep}.pt")
                    rows = [{"tick": 0, "wall_ms": deliveries[0][1], "frames": 117}]
                else:
                    assert len(deliveries) == ticks, len(deliveries)
                    rows = [
                        {"tick": i, "wall_ms": ms, **validate_frames(frames, i)}
                        for i, (frames, ms) in enumerate(deliveries)
                    ]
                    if rep == 1:
                        for i, (frames, _) in enumerate(deliveries[:10]):
                            torch.save(frames, out / f"tick-{i:03d}.pt")
                result["runs"].append(
                    {"warmup": rep == 0, "rows": rows, "total_ms": total_ms, "summary": summarize(rows)}
                )
                await snapshot(f"after_run_{rep}")
                print(json.dumps({"phase": args.phase, "rep": rep, "summary": summarize(rows)}), flush=True)
        else:
            from vllm_omni.diffusion.models.abot_world.actions import ABotCameraControlReducer
            from vllm_omni.experimental.ar_diffusion.consumer import ARDiffusionOmniTickConsumer
            from vllm_omni.experimental.ar_diffusion.session import (
                ARDiffusionSessionEvent,
                ARDiffusionSessionManager,
                ARDiffusionWorkerLifecycle,
            )
            from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionControlInput

            consumer = ARDiffusionOmniTickConsumer(
                engine,
                prompt_provider=lambda tick: {**prompt, "prompt": tick.prompt},
                sampling_params_list=[sampling()],
                diffusion_stage_id=0,
            )
            manager = ARDiffusionSessionManager(
                tick_consumer=consumer,
                lifecycle=ARDiffusionWorkerLifecycle(engine, stage_ids=[0], timeout=120),
                max_pending_events=8,
                control_reducer_factory=ABotCameraControlReducer,
            )

            async def create(sid):
                value = await manager.create_session(sid)
                active.add(sid)
                return value

            async def close(sid):
                await manager.close_session(sid)
                active.remove(sid)

            async def tick(session, index, action=False):
                controls = (
                    ARDiffusionControlInput(
                        track="camera",
                        schema="abot.camera_actions.v1",
                        data={"mode": "script", "frames": [["a"], [], []] if action else [[], [], []]},
                    ),
                )
                start = time.perf_counter()
                await session.accept_event(
                    ARDiffusionSessionEvent(
                        event_id=index,
                        prompt=PROMPT if index == 0 else None,
                        controls=controls,
                    )
                )
                output = await session.next_chunk()
                frames = frames_of(output)
                elapsed_ms = (time.perf_counter() - start) * 1000
                assert consumer.chunk_metadata(output).chunk_index == index
                return frames, {
                    "tick": index,
                    "wall_ms": elapsed_ms,
                    "finished": bool(output.finished),
                    "runtime_events": [],
                    "event_stall_ms": None,
                }

            if args.phase == "typed":
                result["runs"] = []
                for rep in range(args.repeats + 1):
                    sid = f"typed-{rep}"
                    session = await create(sid)
                    rows, tensors = [], []
                    for index in range(args.ticks):
                        frames, row = await tick(session, index)
                        rows.append(row)
                        tensors.append(frames)
                    # Validation, IO, and memory RPCs are deliberately outside timing.
                    for index, frames in enumerate(tensors):
                        rows[index].update(validate_frames(frames, index))
                        if rep == 1 and index < 10:
                            torch.save(frames, out / f"tick-{index:03d}.pt")
                    await snapshot(f"run_{rep}_live")
                    await close(sid)
                    await snapshot(f"run_{rep}_closed")
                    result["runs"].append({"warmup": rep == 0, "rows": rows, "summary": summarize(rows)})
                    print(json.dumps({"phase": "typed", "rep": rep, "summary": summarize(rows)}), flush=True)
                    save()
            else:
                result["comparisons"] = []
                controls = {}
                for name in ("a", "b"):
                    session = await create(f"solo-{name}")
                    controls[name] = [(await tick(session, i, action=name == "b"))[0] for i in range(3)]
                    await close(f"solo-{name}")
                a, b = await create("a"), await create("b")
                for index in range(3):
                    for name, session in (("a", a), ("b", b)):
                        frames, row = await tick(session, index, action=name == "b")
                        exact = torch.equal(frames, controls[name][index])
                        row.update(
                            session=name,
                            exact_vs_solo=exact,
                            max_abs_error=(frames - controls[name][index]).abs().max().item(),
                            **validate_frames(frames, index),
                        )
                        result["comparisons"].append(row)
                        await snapshot(f"{name}{index}")
                        assert exact, row
                await close("a")
                await snapshot("a_closed_b_live")
                await close("b")
                await snapshot("both_closed")
        result["status"] = "passed"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if manager is not None:
            for sid in tuple(active):
                await manager.close_session(sid)
        if engine is not None:
            engine.shutdown()
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vae", choices=("wan", "taew2_2"), required=True)
    parser.add_argument("--phase", choices=("typed", "dual", "stepwise", "offline"), required=True)
    parser.add_argument("--ticks", type=int, default=35)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.ticks < 2 or args.repeats < 1:
        parser.error("--ticks must be at least 2; --repeats must be at least 1")
    asyncio.run(main(args))
