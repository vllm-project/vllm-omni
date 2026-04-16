# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared CLI arguments for diffusion entrypoints."""

import argparse


def add_stage_configs_path_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to the stage configs file. If not specified, the stage configs will be loaded from the model.",
    )


def add_tensor_parallel_size_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallelism size (default: 1).",
    )


def add_diffusion_sequence_parallel_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--usp",
        "--ulysses-degree",
        dest="ulysses_degree",
        type=int,
        default=None,
        help="Ulysses Sequence Parallelism degree for diffusion models. "
        "Equivalent to setting DiffusionParallelConfig.ulysses_degree.",
    )
    parser.add_argument(
        "--ulysses-mode",
        type=str,
        default="strict",
        choices=["strict", "advanced_uaa"],
        help="Ulysses sequence-parallel mode for diffusion models. "
        "'strict' keeps the original divisibility requirements; "
        "'advanced_uaa' enables the experimental UAA path for uneven sequence/head shapes.",
    )
    parser.add_argument(
        "--ring",
        "--ring-degree",
        dest="ring_degree",
        type=int,
        default=None,
        help="Ring Sequence Parallelism degree for diffusion models. "
        "Equivalent to setting DiffusionParallelConfig.ring_degree.",
    )


def add_diffusion_cfg_parallel_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cfg-parallel-size",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of devices for CFG parallel computation for diffusion models. "
        "Equivalent to setting DiffusionParallelConfig.cfg_parallel_size.",
    )


def add_diffusion_vae_patch_parallel_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--vae-patch-parallel-size",
        type=int,
        default=1,
        help="VAE Patch Parallelism degree for diffusion models. "
        "Distributes VAE decode workload across multiple ranks by splitting the latent spatially. "
        "Equivalent to setting DiffusionParallelConfig.vae_patch_parallel_size.",
    )


def add_diffusion_vae_memory_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--vae-use-slicing",
        action="store_true",
        help="Enable VAE slicing for memory optimization (useful for mitigating OOM issues).",
    )
    parser.add_argument(
        "--vae-use-tiling",
        action="store_true",
        help="Enable VAE tiling for memory optimization (useful for mitigating OOM issues).",
    )


def add_diffusion_weight_loading_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--disable-multithread-weight-load",
        action="store_false",
        dest="enable_multithread_weight_load",
        default=True,
        help="Disable multi-threaded safetensors loading (default: enabled with 4 threads).",
    )
    parser.add_argument(
        "--num-weight-load-threads",
        type=int,
        default=4,
        help="Number of threads for parallel weight loading (default: 4).",
    )


def add_diffusion_cpu_offload_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help="Enable CPU offloading for diffusion models.",
    )
