# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""torchrun harness for the native PR8282 Wan chunk executor and noisy KV.

The Omni diffusion frontend does not implement a multinode executor. This runs
its actual DiT model, scheduler, PP coordinator and cache with external ranks.
Pre-encoded conditioning and CUDA completion events isolate DiT throughput.
"""

import argparse
import hashlib
import json
import os
import shutil
import socket
import statistics
import time
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--condition", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--groups", type=int, choices=[1, 2], default=2)
    p.add_argument("--steps", type=int, choices=[4], default=4)
    p.add_argument("--chunks", type=int, default=128)
    p.add_argument("--skip", type=int, default=64)
    p.add_argument("--measure", type=int, default=32)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument(
        "--variants", nargs="+", choices=["baseline", "cached", "hybrid", "hybrid_fused"], default=["baseline"]
    )
    p.add_argument("--expected-sha", default=None)
    args = p.parse_args()
    rank, world, local_rank = [int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")]
    if rank == 0:
        root = Path(__file__).resolve().parents[2]
        files = list((root / "benchmarks/ar_diffusion").glob("*.py"))
        files.append(root / "benchmarks/ar_diffusion/provenance.json")
        files += [
            root / name
            for name in (
                "vllm_omni/diffusion/models/waveserve_wan/transformer.py",
                "vllm_omni/diffusion/models/wan2_2/wan2_2_transformer.py",
                "vllm_omni/experimental/ar_diffusion/chunk_executor.py",
                "vllm_omni/experimental/ar_diffusion/chunk_schedule.py",
                "vllm_omni/experimental/ar_diffusion/kv_cache/noisy.py",
                "vllm_omni/experimental/ar_diffusion/kv_cache/paged_attention.py",
            )
        ]
        manifest = {}
        for file in files:
            if not file.is_file():
                continue
            relative = file.relative_to(root)
            dst = args.out / "source_snapshot" / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and dst.read_bytes() != file.read_bytes():
                raise RuntimeError(f"result directory contains different source: {relative}")
            shutil.copy2(file, dst)
            manifest[str(relative)] = hashlib.sha256(file.read_bytes()).hexdigest()
        (args.out / "source_snapshot/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if world != (args.steps + 1) * args.groups:
        p.error("world must equal (steps + clean) times groups")
    if args.warmup < 0 or args.repeat < 1 or args.measure < 1:
        p.error("warmup must be nonnegative; repeat and measure must be positive")
    if args.skip < 1 or args.skip + args.measure > args.chunks:
        p.error("completion window must fit the request")
    torch.set_num_threads(8)
    from vllm_omni.platforms import current_omni_platform

    current_omni_platform.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    from vllm.config.vllm import set_current_vllm_config
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm_omni.diffusion.models.waveserve_wan.pipeline_waveserve_wan import FlowEuler, _LatentChunkAdapter
    from vllm_omni.diffusion.models.waveserve_wan.transformer import StageWanTransformer
    from vllm_omni.experimental.ar_diffusion.chunk_executor import (
        ARDiffusionChunkContext,
        ChunkRunSpec,
        ChunkTopology,
        run_chunk_pipeline,
    )
    from vllm_omni.experimental.ar_diffusion.chunk_schedule import ChunkSchedule, Ordering, build_chunk_plan
    from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import ARDiffusionNoisyKVSpec, NoisyKVCache, NoisyKVState

    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_model_parallel,
        get_pp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config

    od = OmniDiffusionConfig(
        model=str(args.model),
        dtype=torch.bfloat16,
        model_class_name="WaveServeWanPipeline",
        num_gpus=world,
        enforce_eager=True,
        parallel_config=DiffusionParallelConfig(pipeline_parallel_size=world),
    )
    config = create_diffusion_vllm_config(device, od)
    with (
        set_current_vllm_config(config),
        set_current_diffusion_config(od),
        set_forward_context(vllm_config=config, omni_diffusion_config=od),
    ):
        init_distributed_environment(world_size=world, rank=rank, local_rank=local_rank)
        initialize_model_parallel(pipeline_parallel_size=world)
        with set_default_torch_dtype(torch.bfloat16):
            model = StageWanTransformer(
                layer_groups=args.groups,
                pp_rank=rank,
                transformer_config=json.loads((args.model / "transformer/config.json").read_text()),
            )
        model.to(device=device, dtype=torch.bfloat16)
        if (model.num_layers, model.num_heads, model.head_dim) != (30, 12, 128):
            raise ValueError("this benchmark contract requires Wan 2.1 1.3B geometry")
        files = sorted((args.model / "transformer").glob("*.safetensors"))

        def weights():
            for file in files:
                with safe_open(file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        yield key, f.get_tensor(key)

        model.load_weights(weights())
        model.eval().requires_grad_(False)
        shape = (1, 16, 3, 60, 104)
        tokens = model.seq_len_for_latent(shape)
        kv_spec = ARDiffusionNoisyKVSpec(model.local_num_layers, 12, 128, tokens, tokens, 6)
        prompt = torch.load(args.condition, map_location="cpu", weights_only=True)["text"].to(device)
        pp = get_pp_group()
        sampler = FlowEuler(args.steps, shift=5.0)
        plan = build_chunk_plan(
            ChunkSchedule(args.chunks, args.steps, args.steps + 1, args.groups, Ordering.INTERLEAVED, 6)
        )
        from benchmarks.ar_diffusion.hybrid_kv import HybridNoisyKVState, native_hybrid_attention
        from benchmarks.ar_diffusion.native_optimizations import optimized

        kv_group = dist.new_group(ranks=list(range(world)), backend="nccl")
        reference = args.expected_sha
        groups = []
        for slot in range(plan.num_slots):
            task = plan.task(slot, rank)
            assert task is None or task[1] == rank // args.groups, "condition cache requires a fixed stage"
        producer = args.steps * args.groups - 1
        for group, variant in enumerate(args.variants):
            rows = []
            hybrid = variant in ("hybrid", "hybrid_fused")
            torch.accelerator.reset_peak_memory_stats()
            kv = (
                None
                if hybrid
                else NoisyKVCache(
                    kv_spec,
                    dtype=torch.bfloat16,
                    device=device,
                    layer_groups=args.groups,
                    max_batch_size=1,
                    stages=args.steps + 1,
                )
            )
            with ExitStack() as scopes:
                math_variant = "fused" if variant == "hybrid_fused" else "cached" if variant == "hybrid" else variant
                caches = scopes.enter_context(optimized(model, math_variant))
                for index in range(args.warmup + args.repeat):
                    for module in caches:
                        module.clear()
                    if kv is not None:
                        kv.transport.bytes_sent = kv.transport.bytes_received = 0
                    state = (
                        HybridNoisyKVState(kv_spec, model, kv_group, device, torch.bfloat16)
                        if hybrid
                        else NoisyKVState(kv)
                    )
                    ctx = ARDiffusionChunkContext(
                        ChunkRunSpec(ChunkTopology(args.steps + 1, args.groups), 1, rank, pp), state
                    )
                    request = f"benchmark-{index}"
                    ctx.enqueue(request, plan, chunk_tokens=tokens)
                    events, latents = [], []

                    def finished(chunk, latent):
                        assert chunk == len(events)
                        event = torch.cuda.Event(enable_timing=True)
                        event.record()
                        events.append(event)
                        latents.append(latent.detach())

                    adapter = _LatentChunkAdapter(
                        model,
                        sampler=sampler,
                        prompt_embeds=prompt,
                        latent_shape=shape,
                        seed=0,
                        device=device,
                        dtype=torch.bfloat16,
                        on_finished=finished,
                    )
                    current_omni_platform.synchronize()
                    dist.barrier()
                    start = time.perf_counter()
                    with ExitStack() as forward:
                        forward.enter_context(torch.inference_mode())
                        if hybrid:
                            forward.enter_context(native_hybrid_attention(state))
                        run_chunk_pipeline(ctx=ctx, adapter=adapter)
                    current_omni_platform.synchronize()
                    dist.barrier()
                    seconds = time.perf_counter() - start
                    assert not state._chunk_tokens, "KV state retained a completed request"
                    if kv is not None:
                        assert not kv.pool.keys, "native cache retained versions after request"
                    if rank == producer:
                        assert len(events) == len(latents) == args.chunks
                        milliseconds = [events[0].elapsed_time(e) for e in events]
                        fps = (
                            args.measure
                            * 12000
                            / (milliseconds[args.skip + args.measure - 1] - milliseconds[args.skip - 1])
                        )
                        latent = torch.cat(latents, dim=2).contiguous()
                        assert torch.isfinite(latent).all()
                        digest = hashlib.sha256(latent.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                        if reference is None:
                            reference = digest
                        assert reference == digest, (reference, digest)
                        row = dict(
                            group=group,
                            variant=variant,
                            repeat=index,
                            warmup=index < args.warmup,
                            steady_fps=fps,
                            gpu_ms=milliseconds,
                            request_seconds=seconds,
                            latent_sha256=digest,
                            latent_shape=list(latent.shape),
                        )
                        args.out.mkdir(parents=True, exist_ok=True)
                        (args.out / f"g{group}-{variant}-{index}.json").write_text(json.dumps(row, indent=2) + "\n")
                        rows.append(row)
                        print(
                            f"OMNI_RESULT variant={variant} groups={args.groups} repeat={index} "
                            f"warmup={row['warmup']} fps={fps:.6f} sha={digest}",
                            flush=True,
                        )
                    dist.barrier()
            ranks = [None] * world
            dist.all_gather_object(
                ranks,
                dict(
                    rank=rank,
                    host=socket.gethostname(),
                    peak_allocated=torch.accelerator.max_memory_allocated(),
                    kv_capacity=state.capacity if hybrid else kv.capacity,
                    kv_reserved_bytes=state.reserved_bytes if hybrid else kv.reserved_bytes,
                    kv_pool="hybrid_ring" if hybrid else "native_version_pool",
                    kv_sent_bytes=ctx.kv.bytes_sent,
                    kv_received_bytes=ctx.kv.bytes_received,
                ),
            )
            if rank == producer:
                samples = [r["steady_fps"] for r in rows if not r["warmup"]]
                result = dict(
                    implementation="native vLLM-Omni PR8282 chunk executor; external torchrun harness",
                    boundary="pure DiT CUDA completion, preencoded text, no VAE/codec/frontend",
                    steps=args.steps,
                    groups=args.groups,
                    k=30 // args.groups,
                    dit_gpus=world,
                    history=6,
                    q=3,
                    chunks=args.chunks,
                    skip=args.skip,
                    measure=args.measure,
                    width=832,
                    height=480,
                    variant=variant,
                    conditioning_sha256=hashlib.sha256(args.condition.read_bytes()).hexdigest(),
                    torch=torch.__version__,
                    fps_samples=samples,
                    median_fps=statistics.median(samples),
                    latent_sha256=reference,
                    ranks=ranks,
                )
                groups.append(result)
                (args.out / "summary.json").write_text(json.dumps(dict(groups=groups), indent=2) + "\n")
            # Drop native pool references before measuring the next variant.
            del ctx, state, kv
        if rank == producer:
            (args.out / "completed.txt").write_text("all full latent parity and repeat hashes passed\n")
        destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
