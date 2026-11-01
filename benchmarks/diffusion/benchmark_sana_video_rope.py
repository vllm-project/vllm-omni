# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Same-engine eager/current-dispatch SANA RoPE A/B, with raw receipts.

Run with a pinned local SANA-Video 2B snapshot on one NVIDIA GPU. The eager
arm restores the pre-PR two-call expression; the fused arm calls the actual
production support predicate and custom op. Warmups and file hashing are
excluded from the client latency. All decoded outputs must be bit-identical.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

BASE = "9a1a05facdf1f93e87b56be1177a2030492eef27"
MODEL_REVISION = "fed3bce411c58a0f688a31afe8f52e61acc2b15f"
MODULE_PATH = "vllm_omni/diffusion/models/sana_video/transformer_sana_video.py"


class RopeABWorkerExtension:
    """Install symmetric counters and CUDA events inside the real stage worker."""

    def configure_rope_ab(self, mode: str):
        import vllm_omni.diffusion.models.sana_video.transformer_sana_video as module

        if mode not in ("eager", "fused"):
            raise ValueError(mode)
        if not hasattr(self, "_rope_ab"):
            native_pair = module.apply_interleaved_rotary_emb_pair
            native_fused = module.fused_interleaved_rope
            self._rope_ab = {}

            def counted_pair(query, key, cos, sin):
                state = self._rope_ab
                state["pair_calls"] += 1
                state["query_shape"] = list(query.shape)
                state["table_dtype"] = str(cos.dtype)
                if state["mode"] == "eager":
                    return (
                        module.apply_interleaved_rotary_emb(query, cos, sin),
                        module.apply_interleaved_rotary_emb(key, cos, sin),
                    )
                return native_pair(query, key, cos, sin)

            def counted_fused(*args):
                self._rope_ab["fused_calls"] += 1
                return native_fused(*args)

            module.apply_interleaved_rotary_emb_pair = counted_pair
            module.fused_interleaved_rope = counted_fused
            transformer = self.model_runner.pipeline.transformer

            def before_forward(_module, _args):
                start = torch.Event(enable_timing=True)
                stop = torch.Event(enable_timing=True)
                self._rope_ab["transformer_events"].append((start, stop))
                start.record()

            def after_forward(_module, _args, _output):
                self._rope_ab["transformer_events"][-1][1].record()

            transformer.register_forward_pre_hook(before_forward)
            transformer.register_forward_hook(after_forward)
            pipeline = self.model_runner.pipeline
            native_diffuse = pipeline.diffuse

            def capture_latents(*args, **kwargs):
                output = native_diffuse(*args, **kwargs)
                self._rope_ab["latents"] = output.detach()
                return output

            pipeline.diffuse = capture_latents
            vae = pipeline.vae
            native_decode = vae.decode

            def timed_decode(*args, **kwargs):
                start = torch.Event(enable_timing=True)
                stop = torch.Event(enable_timing=True)
                self._rope_ab["decode_events"].append((start, stop))
                start.record()
                output = native_decode(*args, **kwargs)
                stop.record()
                return output

            vae.decode = timed_decode
        torch.accelerator.synchronize()
        self._rope_ab.pop("latents", None)
        self._rope_ab.update(mode=mode, pair_calls=0, fused_calls=0, transformer_events=[], decode_events=[])
        torch.accelerator.reset_peak_memory_stats()
        return {"mode": mode, "pipeline": type(self.model_runner.pipeline).__name__}

    def collect_rope_ab(self):
        torch.accelerator.synchronize()
        state = self._rope_ab
        return {
            "mode": state["mode"],
            "latent_sha256": hashlib.sha256(
                state["latents"].cpu().contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
            "latent_shape": list(state["latents"].shape),
            "latent_dtype": str(state["latents"].dtype),
            "pair_calls": state["pair_calls"],
            "fused_calls": state["fused_calls"],
            "query_shape": state["query_shape"],
            "table_dtype": state["table_dtype"],
            "transformer_ms": sum(start.elapsed_time(stop) for start, stop in state["transformer_events"]),
            "vae_decode_ms": sum(start.elapsed_time(stop) for start, stop in state["decode_events"]),
            "peak_allocated_bytes": torch.accelerator.max_memory_allocated(),
            "peak_reserved_bytes": torch.accelerator.max_memory_reserved(),
        }


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def verify_eager_reference() -> dict:
    """Prove the eager arm's helper is the baseline's original nested function."""
    sources = [git("show", f"{ref}:{MODULE_PATH}") for ref in (BASE, "HEAD")]
    names = ("apply_rotary_emb", "apply_interleaved_rotary_emb")
    normalized = []
    for source, name in zip(sources, names):
        node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == name)
        node.name = "eager_rope"
        node.returns = None
        if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            node.body.pop(0)
        for item in ast.walk(node):
            if isinstance(item, ast.Name) and item.id == "out":
                item.id = "output"
        normalized.append(ast.dump(node, include_attributes=False))
    if normalized[0] != normalized[1]:
        raise RuntimeError("The head eager helper differs from the pinned baseline expression")
    return {
        "baseline": BASE,
        "head": git("rev-parse", "HEAD"),
        "normalized_eager_sha256": hashlib.sha256(normalized[0].encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local snapshot at the pinned model revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=1, help="Unmeasured full-shape requests per arm")
    parser.add_argument(
        "--vllm-029-compat",
        action="store_true",
        help="Ignore the unused gelu_and_mul_sparse IR priority on vLLM builds without that config field",
    )
    args = parser.parse_args()
    if args.steps < 1 or args.pairs < 2 or args.warmups < 1:
        parser.error("steps/warmups must be positive and pairs must be >=2")
    git("diff", "--exit-code", "HEAD", "--", "vllm_omni")
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = verify_eager_reference()
    model_root = Path(args.model).resolve()
    manifest_path = model_root / "validated_revision.json"
    model_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if model_root.name != MODEL_REVISION and model_manifest.get("revision") != MODEL_REVISION:
        parser.error("Use the pinned HF snapshot directory, or a directory with validated_revision.json")
    receipt = {
        **provenance,
        "model_revision": MODEL_REVISION,
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "request": {
            "height": 480,
            "width": 832,
            "num_frames": 81,
            "fps": 16,
            "seed": 42,
            "guidance_scale": 6.0,
            "motion_score": 30,
            "use_resolution_binning": False,
            "output_type": "np",
            "prompt": "A cinematic tracking shot of a sailboat crossing the ocean at sunset",
            "negative_prompt": "blur, distortion, abrupt cuts",
        },
        "model_snapshot": str(model_root),
        "weight_sha256": {
            entry["rfilename"]: entry["lfs"]["sha256"] for entry in model_manifest.get("files", []) if entry.get("lfs")
        },
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name) for name in ("torch", "vllm", "triton", "diffusers", "transformers")
        },
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        "steps": args.steps,
        "warmups_per_arm": args.warmups,
        "measured_pairs": args.pairs,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "samples": [],
    }
    if args.vllm_029_compat:
        from dataclasses import fields

        from vllm.config.kernel import IrOpPriorityConfig

        if "gelu_and_mul_sparse" in {field.name for field in fields(IrOpPriorityConfig)}:
            parser.error("The requested vLLM 0.29 compatibility adapter is unnecessary on this build")
        native_with_default = IrOpPriorityConfig.with_default

        def compatible_with_default(cls, default, **kwargs):
            kwargs.pop("gelu_and_mul_sparse", None)
            return native_with_default(default, **kwargs)

        IrOpPriorityConfig.with_default = classmethod(compatible_with_default)
        receipt["runtime_compatibility"] = "Ignore unsupported gelu_and_mul_sparse IR configuration (unused by SANA)"

    from vllm_omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    omni = Omni(
        model=args.model,
        model_class_name="SanaVideoPipeline",
        dtype="bfloat16",
        distributed_executor_backend="uni",
        enforce_eager=True,
        cache_backend="none",
        max_num_seqs=1,
        worker_extension_cls="benchmarks.diffusion.benchmark_sana_video_rope.RopeABWorkerExtension",
    )
    reference = None
    reference_latent_sha256 = None

    def rpc(method, *values):
        result = omni.engine.collective_rpc(method=method, args=values, stage_ids=[0], timeout=120)
        while isinstance(result, list):
            if len(result) != 1:
                raise RuntimeError(f"Unexpected RPC result: {result}")
            result = result[0]
        return result

    def run(mode, pair, warmup):
        nonlocal reference, reference_latent_sha256
        rpc("configure_rope_ab", mode)
        sampling = OmniDiffusionSamplingParams(
            height=480,
            width=832,
            num_frames=81,
            fps=16,
            num_inference_steps=args.steps,
            guidance_scale=6.0,
            seed=42,
            output_type="np",
            extra_args={"motion_score": 30, "use_resolution_binning": False},
        )
        start = time.perf_counter()
        result = omni.generate(
            {
                "prompt": "A cinematic tracking shot of a sailboat crossing the ocean at sunset",
                "negative_prompt": "blur, distortion, abrupt cuts",
            },
            sampling,
            use_tqdm=False,
        )[0]
        elapsed = time.perf_counter() - start
        stats = rpc("collect_rope_ab")
        video = np.asarray(result.images)
        while video.ndim > 4 and video.shape[0] == 1:
            video = video[0]
        if video.shape != (81, 480, 832, 3) or not np.isfinite(video).all():
            raise RuntimeError(f"Invalid video: {video.shape}, {video.dtype}")
        if reference is None:
            reference = video.copy()
            reference_latent_sha256 = stats["latent_sha256"]
            np.save(args.output / "reference.npy", reference)
        equal = reference.dtype == video.dtype and np.array_equal(reference, video)
        if stats["latent_sha256"] != reference_latent_sha256:
            raise RuntimeError("Final latent differs from the eager reference")
        if not equal:
            raise RuntimeError("Decoded output differs from the eager reference")
        expected_calls = args.steps * 20
        if stats["pair_calls"] != expected_calls or stats["fused_calls"] != (expected_calls if mode == "fused" else 0):
            raise RuntimeError(f"Unexpected RoPE dispatch: {stats}")
        row = {
            **stats,
            "pair": pair,
            "warmup": warmup,
            "client_s": elapsed,
            "output_shape": list(video.shape),
            "output_dtype": str(video.dtype),
            "output_sha256": hashlib.sha256(video.tobytes()).hexdigest(),
            "bit_equal": equal,
        }
        receipt["samples"].append(row)
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    try:
        for index in range(args.warmups):
            for mode in ("eager", "fused"):
                run(mode, index, True)
        for index in range(args.pairs):
            for mode in ("eager", "fused") if index % 2 == 0 else ("fused", "eager"):
                run(mode, index, False)
        receipt["summary"] = {}
        for mode in ("eager", "fused"):
            rows = [r for r in receipt["samples"] if r["mode"] == mode and not r["warmup"]]
            receipt["summary"][mode] = {
                field: {
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "stdev": statistics.stdev(values),
                }
                for field in (
                    "client_s",
                    "transformer_ms",
                    "vae_decode_ms",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                )
                for values in [[r[field] for r in rows]]
            }
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt["summary"], indent=2), flush=True)
    finally:
        omni.close()
        if args.vllm_029_compat:
            IrOpPriorityConfig.with_default = native_with_default


if __name__ == "__main__":
    main()
