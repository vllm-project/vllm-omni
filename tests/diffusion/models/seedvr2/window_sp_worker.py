# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""torchrun worker for the SeedVR2 window-aligned sequence-parallel path.

Cases
-----
``transport``
    Pure Plan A redistribution on synthetic hidden rows: every layout
    transition of the 3B schedule (regular <-> shifted, both directions) must
    move rows bit-exactly, including uneven splits and ranks without windows.
``toy-block``
    A small joint-video/text window attention block run for four layers
    (A -> B -> A -> B) against a single-rank oracle, checking video *and* text
    agreement and the global window-mean text reduction.
``seedvr2``
    The real 3B checkpoint: the SP=N result must match the SP=1 result of the
    same port within the frozen fixture tolerance.

The worker is meant to be launched with ``torchrun``; it never spawns pytest
itself.  Each rank writes ``rank<rank>.json`` plus a merged ``report.json`` into
``--report-dir``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist


@dataclass
class Report:
    case: str
    world_size: int
    rank: int
    status: str = "ok"
    error: str | None = None
    metrics: dict = field(default_factory=dict)

    def write(self, report_dir: Path) -> None:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f"rank{self.rank}.json").write_text(
            json.dumps(
                {
                    "case": self.case,
                    "world_size": self.world_size,
                    "rank": self.rank,
                    "status": self.status,
                    "error": self.error,
                    **self.metrics,
                },
                indent=2,
                sort_keys=True,
            )
        )


# ---------------------------------------------------------------------------
# transport case
# ---------------------------------------------------------------------------


def run_transport(args: argparse.Namespace, rank: int, world_size: int) -> Report:
    from vllm_omni.diffusion.models.seedvr2.window_sp import WindowLayoutManager

    report = Report(case="transport", world_size=world_size, rank=rank)
    group = dist.new_group(list(range(world_size)))
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)

    token_grid = (args.frames, args.height // 2, args.width // 2)
    manager = WindowLayoutManager(
        token_grid,
        group=group,
        world_size=world_size,
        rank=rank,
        num_layers=args.num_layers,
        methods=("720pwin_by_size_bysize", "720pswin_by_size_bysize"),
    )
    num_tokens = token_grid[0] * token_grid[1] * token_grid[2]
    features = args.transport_features
    torch.manual_seed(args.seed)
    canonical = torch.arange(num_tokens, dtype=torch.int64, device=device).unsqueeze(1).repeat(1, features)
    canonical = canonical + torch.arange(features, device=device) * num_tokens

    first = manager.layer_layout(0)
    hidden = manager.rank_plan(first).global_token_ids.to(device)
    hidden = canonical.index_select(0, hidden).to(torch.float32)
    current = first.key

    checked = 0
    transitions = 0
    for layer in range(1, args.num_layers):
        layout = manager.layer_layout(layer)
        hidden = manager.ensure_layout(hidden, current, layout.key)
        current = layout.key
        transitions += 1
        # After every transition the local rows must equal the canonical rows of
        # this rank's windows, bit for bit.
        expected_ids = manager.rank_plan(layout).global_token_ids.to(device)
        expected = canonical.index_select(0, expected_ids).to(torch.float32)
        if hidden.shape != expected.shape or not torch.equal(hidden, expected):
            report.status = "fail"
            report.error = f"row mismatch after transition {layer} (rank {rank})"
            break
        checked += 1

    gather = torch.tensor([1 if report.status == "ok" else 0, checked, transitions], dtype=torch.int64, device=device)
    gathered = [torch.zeros_like(gather) for _ in range(world_size)]
    dist.all_gather(gathered, gather, group=group)
    report.metrics.update(
        {
            "transitions_checked": checked,
            "schedule_transitions": transitions,
            "synchronized": bool(all(int(g[2]) == transitions for g in gathered)),
            "rows_per_rank": [int(manager.rank_plan(manager.layer_layout(0)).global_token_ids.numel())],
        }
    )
    dist.destroy_process_group(group)
    return report


# ---------------------------------------------------------------------------
# toy block case
# ---------------------------------------------------------------------------


def run_toy_block(args: argparse.Namespace, rank: int, world_size: int) -> Report:
    from vllm_omni.diffusion.models.seedvr2.na_ops import SeedVR2WindowRuntime
    from vllm_omni.diffusion.models.seedvr2.nadit import NaMMSRTransformerBlock

    report = Report(case="toy-block", world_size=world_size, rank=rank)
    group = dist.new_group(list(range(world_size)))
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)

    vid_dim, txt_dim, heads, head_dim = 64, 64, 4, 24
    text_len = 5
    layers = 4
    token_grid = (args.frames, args.height // 2, args.width // 2)
    num_tokens = token_grid[0] * token_grid[1] * token_grid[2]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    txt = torch.randn(text_len, txt_dim, generator=generator).to(device)
    canonical = torch.randn(num_tokens, vid_dim, generator=generator).to(device)

    def build_model():
        torch.manual_seed(args.seed)
        model = NaMMSRTransformerBlock(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            emb_dim=6 * vid_dim,
            heads=heads,
            head_dim=head_dim,
            expand_ratio=2,
            norm_eps=1e-5,
            qk_bias=False,
            mlp_type="swiglu",
            shared_weights=False,
            rope_dim=head_dim,
            is_last_layer=False,
            use_varlen_kernel=False,
        ).to(device=device, dtype=torch.float64)
        return model

    model = build_model()
    emb = torch.randn(1, 6 * vid_dim, generator=generator).to(device=device, dtype=torch.float64)

    # Single-rank oracle on the same block weights.
    oracle_runtime = SeedVR2WindowRuntime(
        token_grid, text_len=text_len, group=None, world_size=1, rank=0, num_layers=layers
    )
    vid_oracle = canonical.to(torch.float64)
    txt_oracle = txt.to(torch.float64)
    with torch.no_grad():
        for layer in range(layers):
            layout = oracle_runtime.layout_for_layer(layer)
            vid_oracle = oracle_runtime.ensure_layout(
                vid_oracle, oracle_runtime.layout_for_layer(max(layer - 1, 0)).key, layout.key
            )
            ctx = oracle_runtime.context(layout, device)
            vid_oracle, txt_oracle = model(vid_oracle, txt_oracle, emb, ctx, oracle_runtime)

    # Distributed run.
    runtime = SeedVR2WindowRuntime(
        token_grid, text_len=text_len, group=group, world_size=world_size, rank=rank, num_layers=layers
    )
    first = runtime.layout_for_layer(0)
    vid_local = runtime.local_rows_for(canonical.to(torch.float64), first)
    txt_local = txt.to(torch.float64)
    with torch.no_grad():
        current = first.key
        for layer in range(layers):
            layout = runtime.layout_for_layer(layer)
            vid_local = runtime.ensure_layout(vid_local, current, layout.key)
            current = layout.key
            ctx = runtime.context(layout, device)
            vid_local, txt_local = model(vid_local, txt_local, emb, ctx, runtime)
    vid_distributed = runtime.to_canonical_rows(vid_local, runtime.layout_for_layer(layers - 1))

    diff = (vid_distributed - vid_oracle).abs()
    txt_diff = (txt_local - txt_oracle).abs()
    report.metrics.update(
        {
            "video_max_abs_error": float(diff.max().item()),
            "video_rel_l2": float(
                (torch.linalg.vector_norm(vid_distributed - vid_oracle) / torch.linalg.vector_norm(vid_oracle)).item()
            ),
            "text_max_abs_error": float(txt_diff.max().item()),
            "layout_transitions": runtime.stats["layout_transitions"],
            "network_transitions": runtime.stats["network_transitions"],
            "text_all_reduces": runtime.stats["text_all_reduces"],
            "tolerance": 1e-9,
        }
    )
    if report.metrics["video_max_abs_error"] > 1e-9 or report.metrics["text_max_abs_error"] > 1e-9:  # noqa: PLR2004
        report.status = "fail"
        report.error = "toy block mismatch against the single-rank oracle"
    dist.destroy_process_group(group)
    return report


# ---------------------------------------------------------------------------
# seedvr2 case
# ---------------------------------------------------------------------------


#: The released checkpoint stores the RoPE table one module deeper than this
#: port registers it.  The mapping is deliberately restricted to that suffix.
ROPE_BUFFER_KEY_SUFFIX = ".rope.rope.freqs"
ROPE_BUFFER_KEY_NORMALIZED = ".rope.freqs"


def normalize_reference_state_dict(state: dict, *, num_layers: int | None = None):
    """Normalize reference checkpoint keys into this port's layout.

    Only the RoPE buffer suffix is rewritten; every other key must already match.
    Returns ``(normalized_state, stats)`` and raises on a collision instead of
    silently letting one tensor overwrite another.
    """
    normalized: dict = {}
    stats = {"rope_buffer_keys_normalized": 0, "truncated_block_keys": 0}
    for key, value in state.items():
        new_key = key
        if key.endswith(ROPE_BUFFER_KEY_SUFFIX):
            new_key = key[: -len(ROPE_BUFFER_KEY_SUFFIX)] + ROPE_BUFFER_KEY_NORMALIZED
            stats["rope_buffer_keys_normalized"] += 1
        if num_layers is not None and new_key.startswith("blocks."):
            index = int(new_key.split(".")[1])
            if index >= num_layers:
                stats["truncated_block_keys"] += 1
                continue
        if new_key in normalized:
            raise RuntimeError(
                f"checkpoint key collision after normalization: {new_key!r} produced by more than one source key"
            )
        normalized[new_key] = value
    return normalized, stats


def _load_port_model(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    *,
    model_factory=None,
    state_loader=None,
):
    """Build the port, load the released checkpoint, and fail on any mismatch.

    ``model_factory`` / ``state_loader`` exist so the checkpoint contract can be
    unit-tested with a tiny fixture instead of a full 3B model.
    """
    from safetensors.torch import load_file

    from vllm_omni.diffusion.models.seedvr2.nadit import SEEDVR2_3B_CONFIG, SeedVR2NaDiT

    full_layers = int(SEEDVR2_3B_CONFIG["num_layers"])
    truncated = args.num_layers != full_layers
    if truncated and not args.allow_truncated_layers:
        raise RuntimeError(
            f"--num-layers {args.num_layers} != {full_layers}: full-depth loading is the acceptance path. "
            "Pass --allow-truncated-layers to load a development fixture instead."
        )

    if model_factory is None:

        def model_factory():
            model_kwargs = dict(SEEDVR2_3B_CONFIG)
            model_kwargs["num_layers"] = args.num_layers
            model_kwargs["use_varlen_kernel"] = args.varlen
            return SeedVR2NaDiT(**model_kwargs)

    if state_loader is None:
        state_loader = load_file

    model = model_factory()
    state = state_loader(args.ckpt)
    normalized, stats = normalize_reference_state_dict(state, num_layers=args.num_layers if truncated else None)
    missing, unexpected = model.load_state_dict(normalized, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint does not match the port: "
            f"{len(missing)} missing key(s) {list(missing)[:5]}, "
            f"{len(unexpected)} unexpected key(s) {list(unexpected)[:5]}"
        )
    model = model.to(device=device, dtype=dtype).eval()
    return model, missing, unexpected, stats


def run_seedvr2(args: argparse.Namespace, rank: int, world_size: int) -> Report:
    report = Report(case="seedvr2", world_size=world_size, rank=rank)
    group = dist.new_group(list(range(world_size)))
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    dtype = getattr(torch, args.dtype)

    model, missing, unexpected, load_stats = _load_port_model(args, device, dtype)
    report.metrics["missing_keys"] = len(missing)
    report.metrics["unexpected_keys"] = len(unexpected)
    report.metrics["checkpoint_key_normalization"] = load_stats
    report.metrics["checkpoint_layers"] = args.num_layers

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    frames, height, width = args.frames, args.height, args.width
    vid = torch.randn(frames * height * width, 33, generator=generator).to(device=device, dtype=dtype)
    txt = torch.randn(args.text_len, 5120, generator=generator).to(device=device, dtype=dtype)
    vid_shape = torch.tensor([[frames, height, width]], dtype=torch.long, device=device)
    txt_shape = torch.tensor([[args.text_len]], dtype=torch.long, device=device)
    timestep = torch.tensor([1000.0], device=device, dtype=dtype)

    token_grid = model.token_grid_for(vid_shape)
    num_tokens = token_grid[0] * token_grid[1] * token_grid[2]

    def timed(fn):
        torch.accelerator.synchronize(device)
        start = time.perf_counter()
        out = fn()
        torch.accelerator.synchronize(device)
        return out, (time.perf_counter() - start) * 1000.0

    def measure(runtime=None):
        samples = []
        output = None
        snapshot = None
        previous = dict(runtime.stats) if runtime is not None else {}
        with torch.no_grad():
            for index in range(args.warmup + args.iterations):
                output, elapsed = timed(
                    lambda: model(
                        vid=vid,
                        txt=txt,
                        vid_shape=vid_shape,
                        txt_shape=txt_shape,
                        timestep=timestep,
                        runtime=runtime,
                    ).vid_sample
                )
                if runtime is not None:
                    # Counters accumulate over the whole session; report the last
                    # single forward so the numbers describe exactly one pass.
                    snapshot = {key: value - previous.get(key, 0) for key, value in runtime.stats.items()}
                    previous = dict(runtime.stats)
                if index >= args.warmup:
                    samples.append(elapsed)
        samples.sort()
        median = samples[len(samples) // 2]
        p95 = samples[min(len(samples) - 1, int(round(0.95 * (len(samples) - 1))))]
        return output, median, p95, snapshot

    single, single_ms, single_p95, _ = measure(None)

    runtime = model.build_runtime(
        token_grid,
        text_len=args.text_len,
        group=group if world_size > 1 else None,
        world_size=world_size,
        rank=rank,
    )
    model.reset_attention_stats()
    distributed, distributed_ms, distributed_p95, stats = measure(runtime)
    attention = model.attention_path_summary()

    diff = (distributed.float() - single.float()).abs()
    ref_norm = torch.linalg.vector_norm(single.float()).clamp_min(1e-12)
    rel_l2 = float((torch.linalg.vector_norm(distributed.float() - single.float()) / ref_norm).item())
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    finite = bool(torch.isfinite(distributed).all().item())

    per_rank = torch.tensor(
        [int(runtime.manager.rank_plan(runtime.layout_for_layer(0)).global_token_ids.numel())],
        dtype=torch.int64,
        device=device,
    )
    counts = [torch.zeros_like(per_rank) for _ in range(world_size)]
    dist.all_gather(counts, per_rank, group=group) if world_size > 1 else None
    timing = torch.tensor([single_ms, distributed_ms], dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(timing, op=dist.ReduceOp.MAX, group=group)

    peak = torch.accelerator.max_memory_allocated(device)
    peaks = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]
    if world_size > 1:
        dist.all_gather(peaks, torch.tensor([peak], dtype=torch.int64, device=device), group=group)

    atol, rtol = (float(v) for v in args.tolerance.split(","))
    ok = finite and bool(torch.allclose(distributed.float(), single.float(), atol=atol, rtol=rtol))
    report.metrics.update(
        {
            "post_patch_shape": list(token_grid),
            "video_tokens": num_tokens,
            "text_tokens": args.text_len,
            "sp_size": world_size,
            "windows_per_layout": {str(layer): int(runtime.layout_for_layer(layer).num_windows) for layer in (0, 1)},
            "per_rank_tokens": [int(c) for c in counts[rank].tolist()] if world_size > 1 else [num_tokens],
            "layout_transition_count": stats["layout_transitions"],
            "network_a2a_count": stats["network_transitions"],
            "local_reorder_count": stats["local_reorders"],
            "text_all_reduce_count": stats["text_all_reduces"],
            "logical_remote_video_bytes": stats["remote_video_rows"]
            * model.vid_dim
            * torch.empty(0, dtype=dtype).element_size(),
            "logical_text_reduce_tensor_bytes": args.text_len * model.vid_dim * 4 * stats["text_all_reduces"],
            "forward_ms_sp1": single_ms,
            "forward_ms_sp1_p95": single_p95,
            "forward_ms_spN": distributed_ms,
            "forward_ms_spN_p95": distributed_p95,
            "measured_iterations": args.iterations,
            "warmup_iterations": args.warmup,
            "speedup_sp1_over_spN": single_ms / distributed_ms if distributed_ms > 0 else None,
            "scaling_efficiency": (single_ms / distributed_ms) / world_size if distributed_ms > 0 else None,
            "max_abs_error": max_abs,
            "mean_abs_error": mean_abs,
            "rel_l2": rel_l2,
            "finite": finite,
            "tolerance_atol": atol,
            "tolerance_rtol": rtol,
            "peak_allocated_bytes": int(peak),
            "peak_allocated_bytes_all_ranks": [int(p.item()) for p in peaks],
            "dtype": args.dtype,
            "attention_path_requested": "packed_varlen" if args.varlen else "grouped_sdpa",
            "attention_layers_per_resolved_path": attention["layers_per_path"],
            "attention_backend_names": attention["backend_names"],
            "varlen_fallback_reasons": attention["varlen_fallback_reasons"],
            "packed_varlen_calls": attention["packed_varlen_calls"],
            "grouped_sdpa_calls": attention["grouped_sdpa_calls"],
            "no_local_windows_calls": attention["no_local_windows_calls"],
        }
    )
    if not ok:
        report.status = "fail"
        report.error = f"SP={world_size} output differs from SP=1 (max_abs={max_abs:.3e}, rel_l2={rel_l2:.3e})"
    if world_size > 1:
        dist.destroy_process_group(group)
    return report


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=("transport", "toy-block", "seedvr2"))
    parser.add_argument("--seed", type=int, default=7723)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--ckpt", default="/models/seedvr2_ema_3b_fp16.safetensors")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--text-len", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--transport-features", type=int, default=8)
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--tolerance", default="2e-2,2e-2", help="atol,rtol for the seedvr2 case")
    parser.add_argument("--varlen", action="store_true", help="request the packed-varlen attention kernel")
    parser.add_argument(
        "--allow-truncated-layers",
        action="store_true",
        help="development only: load a checkpoint truncated to --num-layers instead of the full depth",
    )
    parser.add_argument("--warmup", type=int, default=1, help="warmup iterations for the seedvr2 case")
    parser.add_argument("--iterations", type=int, default=3, help="measured iterations for the seedvr2 case")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    dist.init_process_group(backend="nccl")
    if args.case == "transport":
        report = run_transport(args, rank, world_size)
    elif args.case == "toy-block":
        report = run_toy_block(args, rank, world_size)
    else:
        report = run_seedvr2(args, rank, world_size)
    report.write(Path(args.report_dir))
    dist.barrier()
    if rank == 0:
        merged = {}
        for other in range(world_size):
            path = Path(args.report_dir) / f"rank{other}.json"
            if path.exists():
                merged[f"rank{other}"] = json.loads(path.read_text())
        (Path(args.report_dir) / "report.json").write_text(json.dumps(merged, indent=2, sort_keys=True))
        statuses = {entry["status"] for entry in merged.values()}
        print(f"[window-sp] case={args.case} world_size={world_size} statuses={sorted(statuses)}")
        for entry in merged.values():
            interesting = {
                key: value
                for key, value in entry.items()
                if key
                in (
                    "max_abs_error",
                    "rel_l2",
                    "video_max_abs_error",
                    "text_max_abs_error",
                    "layout_transition_count",
                    "network_a2a_count",
                    "text_all_reduce_count",
                    "forward_ms_sp1",
                    "forward_ms_spN",
                    "error",
                )
            }
            print(f"  rank{entry['rank']}: {interesting}")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
