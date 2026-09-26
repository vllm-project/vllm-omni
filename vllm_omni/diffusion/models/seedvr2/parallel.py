# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SeedVR2 uses the regular SP group for model-owned window routing."""

from vllm_omni.diffusion.data import DiffusionParallelConfig


def validate_seedvr2_parallel_config(parallel: DiffusionParallelConfig) -> None:
    if (
        parallel.sequence_parallel_size != parallel.ulysses_degree
        or parallel.ring_degree != 1
        or parallel.allgather_degree != 1
        or parallel.ulysses_mode != "strict"
        or parallel.ulysses_a2a_permute
    ):
        raise ValueError(
            "SeedVR2 sequence parallelism requires pure ulysses_degree with ring_degree=allgather_degree=1, "
            "ulysses_mode='strict', and ulysses_a2a_permute=False; "
            "SeedVR2 owns the window attention exchange."
        )
