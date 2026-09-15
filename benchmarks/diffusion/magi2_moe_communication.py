# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Component profiler for MAGI-2's multi-head MoE layer under sequence parallelism.

Refs https://github.com/vllm-project/vllm-omni/issues/7085 (M3). Breaks one
``Magi2MultiHeadMoELayer`` step into the pieces M3 asks about -- the sequence-size
exchange, head dispatch/undispatch, TP reductions, routed expert compute and the
shared-expert branch -- so the relative cost is measured before any overlap or
kernel work is proposed.

The layer is built from the released MAGI-2 Preview geometry with random weights.
Routing decisions therefore differ from a real checkpoint, which changes expert
load balance but not the collective sizes, which depend only on token counts and
head geometry.

Usage:

    torchrun --standalone --nnodes=1 --nproc-per-node=4 \\
        benchmarks/diffusion/magi2_moe_communication.py --tokens 7500

    # TP layout instead of the released SP-only layout
    torchrun --standalone --nnodes=1 --nproc-per-node=4 \\
        benchmarks/diffusion/magi2_moe_communication.py --layout tp
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Callable

import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.models.magi2.configuration_magi2 import (
    Magi2MHCConfig,
    Magi2MoEConfig,
    Magi2PreviewConfig,
)
from vllm_omni.diffusion.models.magi2.modeling_magi2 import (
    Magi2MultiHeadMoELayer,
    Modality,
    ModalityDispatcher,
)
from vllm_omni.diffusion.models.magi2.parallel import ep_dispatch, ep_undispatch
from vllm_omni.platforms import current_omni_platform

# The released Preview transformer runs MoE in layers 2..37.
_MOE_LAYERS_PER_FORWARD = 36


def _released_config() -> Magi2PreviewConfig:
    """Return the released MAGI-2 Preview geometry."""

    return Magi2PreviewConfig(
        num_layers=40,
        hidden_size=3072,
        head_dim=128,
        num_query_groups=24,
        params_dtype=torch.bfloat16,
        mhc=Magi2MHCConfig(),
        moe=Magi2MoEConfig(
            num_heads=12,
            num_experts=256,
            top_k=6,
            expert_intermediate_size=1280,
            shared_expert_intermediate_size=1280,
            modality_shared_expert_intermediate_size=1280,
        ),
    )


def _measure(
    operation: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
) -> list[float]:
    """Return per-iteration milliseconds measured with CUDA events."""

    for _ in range(warmup):
        operation()
    current_omni_platform.synchronize()

    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        current_omni_platform.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def _rank_max_median(samples: list[float], device: torch.device) -> tuple[float, float, float]:
    """Reduce per-rank median/min/max to the slowest rank, which paces the step."""

    values = torch.tensor(
        [statistics.median(samples), min(samples), max(samples)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return float(values[0]), float(values[1]), float(values[2])


def _build_inputs(
    config: Magi2PreviewConfig,
    tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, ModalityDispatcher]:
    generator = torch.Generator(device="cpu").manual_seed(4242)
    hidden = torch.randn(
        tokens,
        config.hidden_size,
        dtype=torch.float32,
        generator=generator,
    ).to(device=device, dtype=config.params_dtype)
    # A representative packed mix: mostly video tokens with audio and text.
    modality = torch.full((tokens,), int(Modality.VIDEO), dtype=torch.int64)
    modality[: max(tokens // 20, 1)] = int(Modality.TEXT)
    modality[tokens // 2 : tokens // 2 + max(tokens // 20, 1)] = int(Modality.AUDIO)
    return hidden, ModalityDispatcher(modality.to(device), 3)


def _async_ep_dispatch(
    tensor: torch.Tensor,
    group,
    sequence_split_sizes: list[int],
):
    """Async variant of ``ep_dispatch`` used only by the overlap probe.

    Mirrors ``vllm_omni.diffusion.models.magi2.parallel.ep_dispatch`` but returns
    the work handle so the caller can run independent compute before waiting.
    """

    sequence, heads, dim = tensor.shape
    local_heads = heads // group.world_size
    send = tensor.contiguous().view(sequence, group.world_size, local_heads, dim).permute(1, 0, 2, 3).contiguous()
    output = torch.empty(
        (sum(sequence_split_sizes), local_heads, dim),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    row_width = local_heads * dim
    handle = dist.all_to_all_single(
        output.view(-1),
        send.view(-1),
        output_split_sizes=[size * row_width for size in sequence_split_sizes],
        input_split_sizes=[sequence * row_width] * group.world_size,
        group=group.group,
        async_op=True,
    )
    return handle, output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens",
        type=int,
        default=7500,
        help="Local tokens per rank after the Ulysses split.",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Unmeasured warmup iterations.")
    parser.add_argument("--iterations", type=int, default=30, help="Measured iterations.")
    parser.add_argument(
        "--layout",
        choices=("sp", "tp"),
        default="sp",
        help="Released SP-only MoE-head layout, or the TP head layout.",
    )
    parser.add_argument("--json", type=str, default="", help="Optional path for machine-readable results.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method="env://",
        local_rank=local_rank,
    )
    tensor_parallel_size = world_size if args.layout == "tp" else 1
    sequence_parallel_size = 1 if args.layout == "tp" else world_size
    initialize_model_parallel(
        data_parallel_size=1,
        cfg_parallel_size=1,
        sequence_parallel_size=sequence_parallel_size,
        ulysses_degree=sequence_parallel_size,
        ring_degree=1,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=1,
    )

    try:
        config = _released_config()
        layer = Magi2MultiHeadMoELayer(config).to(device)
        hidden, dispatcher = _build_inputs(config, args.tokens, device)
        split_sizes = [args.tokens] * world_size

        with torch.no_grad():
            normalized = layer.pre_norm(hidden, dispatcher)
            routed_input = layer.split_linear(normalized)
            routed_output = layer.moe_mlp(routed_input, sequence_split_sizes=split_sizes)

            def whole_layer() -> object:
                return layer(hidden, dispatcher, sequence_split_sizes=split_sizes)

            def shared_branch() -> object:
                return layer._shared_experts(normalized, dispatcher)

            def routed_branch() -> object:
                return layer.merge_linear(layer.moe_mlp(routed_input, sequence_split_sizes=split_sizes))

            def moe_with_metadata() -> object:
                return layer.moe_mlp(routed_input, sequence_split_sizes=split_sizes)

            def moe_with_collective() -> object:
                return layer.moe_mlp(routed_input)

            measurements: dict[str, Callable[[], object]] = {
                "whole layer": whole_layer,
                "routed branch (split+MoE+merge)": routed_branch,
                "shared-expert branch": shared_branch,
                "MoE with reused metadata": moe_with_metadata,
                "MoE with per-layer collective": moe_with_collective,
                "  split_linear": lambda: layer.split_linear(normalized),
                "  merge_linear": lambda: layer.merge_linear(routed_output),
            }

            moe = layer.moe_mlp
            if moe.ep_group.world_size > 1 and not moe.ep_group.replicated_sequence:
                # SP-only layout: the routed branch pays two token all-to-all
                # collectives around the local expert compute.
                heads = routed_input.view(-1, moe.num_heads, moe.d_head)
                dispatched = ep_dispatch(heads, moe.ep_group, split_sizes)
                measurements["  ep_dispatch"] = lambda: ep_dispatch(heads, moe.ep_group, split_sizes)
                measurements["  local expert compute"] = lambda: moe._local_forward(dispatched)
                measurements["  ep_undispatch"] = lambda: ep_undispatch(dispatched, moe.ep_group, split_sizes)

                # M3 overlap probe: the shared branch depends only on
                # ``normalized``, so it can run while the dispatch all-to-all
                # is in flight. Measured here before proposing a model change.
                def serialized_branches() -> object:
                    routed = ep_dispatch(heads, moe.ep_group, split_sizes)
                    routed = moe._local_forward(routed)
                    routed = ep_undispatch(routed, moe.ep_group, split_sizes)
                    return layer._shared_experts(normalized, dispatcher), routed

                def overlapped_branches() -> object:
                    handle, pending = _async_ep_dispatch(heads, moe.ep_group, split_sizes)
                    shared = layer._shared_experts(normalized, dispatcher)
                    handle.wait()
                    routed = moe._local_forward(pending)
                    routed = ep_undispatch(routed, moe.ep_group, split_sizes)
                    return shared, routed

                measurements["branches serialized (today)"] = serialized_branches
                measurements["branches overlapped (probe)"] = overlapped_branches

            tp_group = layer.merge_linear.tp_group
            if tp_group.world_size > 1:
                # Row-parallel ``merge_linear`` reduces the full activation.
                reduction_buffer = torch.empty_like(normalized)

                def tp_reduction() -> object:
                    return dist.all_reduce(reduction_buffer, group=tp_group.group)

                measurements["  TP reduction (merge_linear)"] = tp_reduction

            results: dict[str, tuple[float, float, float]] = {}
            for name, operation in measurements.items():
                samples = _measure(operation, warmup=args.warmup, iterations=args.iterations)
                results[name] = _rank_max_median(samples, device)

        if rank == 0:
            print(f"\nMAGI-2 Preview MoE layer, layout={args.layout}, world_size={world_size}")
            print(f"local tokens/rank={args.tokens}, warmup={args.warmup}, iterations={args.iterations}")
            print(f"{'component':<34}{'median ms':>12}{'min ms':>10}{'max ms':>10}")
            for name, (median, minimum, maximum) in results.items():
                print(f"{name:<34}{median:>12.4f}{minimum:>10.4f}{maximum:>10.4f}")

            metadata_saving = results["MoE with per-layer collective"][0] - results["MoE with reused metadata"][0]
            shared_median = results["shared-expert branch"][0]
            routed_median = results["routed branch (split+MoE+merge)"][0]
            print(
                f"\nper-layer metadata exchange saved: {metadata_saving:.4f} ms"
                f"  ({metadata_saving * _MOE_LAYERS_PER_FORWARD:.2f} ms over {_MOE_LAYERS_PER_FORWARD} MoE layers)"
            )
            print(
                f"shared branch is {shared_median / routed_median:.1%} of the routed branch; "
                "they are serialized today and carry no data dependency"
            )
            dispatch = results.get("  ep_dispatch")
            undispatch = results.get("  ep_undispatch")
            if dispatch and undispatch:
                collectives = dispatch[0] + undispatch[0]
                print(
                    f"routed-branch token collectives: {collectives:.4f} ms/layer "
                    f"({collectives * _MOE_LAYERS_PER_FORWARD:.2f} ms over {_MOE_LAYERS_PER_FORWARD} MoE layers); "
                    f"shared branch could hide up to {min(shared_median, collectives):.4f} ms/layer"
                )
            serialized = results.get("branches serialized (today)")
            overlapped = results.get("branches overlapped (probe)")
            if serialized and overlapped:
                delta = serialized[0] - overlapped[0]
                print(
                    f"overlap probe: {serialized[0]:.4f} -> {overlapped[0]:.4f} ms/layer "
                    f"({delta:+.4f} ms, {delta / serialized[0]:+.2%}); "
                    f"{delta * _MOE_LAYERS_PER_FORWARD:+.2f} ms over {_MOE_LAYERS_PER_FORWARD} MoE layers"
                )
            if args.json:
                with open(args.json, "w") as handle:
                    json.dump(
                        {
                            "layout": args.layout,
                            "world_size": world_size,
                            "tokens_per_rank": args.tokens,
                            "iterations": args.iterations,
                            "components": {
                                name: {"median_ms": median, "min_ms": minimum, "max_ms": maximum}
                                for name, (median, minimum, maximum) in results.items()
                            },
                        },
                        handle,
                        indent=2,
                    )
    finally:
        destroy_distributed_env()


if __name__ == "__main__":
    main()
