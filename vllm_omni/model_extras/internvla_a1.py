# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

INTERNVLA_A1_EXTRA_BODY_PARAMS = frozenset(
    {
        "num_steps",
        "decode_image",
    }
)
INTERNVLA_A1_EXTRA_OUTPUT_PARAMS = frozenset(
    {
        "decoded",
    }
)


def load_eval_context(
    *,
    model_dir: str | Path,
    dataset_dir: str | Path,
    device: str = "cuda",
    dtype: str = "bfloat16",
    compile_model: bool = False,
    attn_implementation: str = "eager",
    enable_regional_compile: bool = False,
    **_: Any,
) -> dict[str, Any]:
    """Load InternVLA-A1 config, train metadata, and A2D open-loop dataset."""
    from vllm_omni.diffusion.models.internvla_a1 import (
        InternVLAA1Config,
        InternVLAA1TrainMetadata,
    )
    from vllm_omni.model_extras.internvla_a1_dataset import (
        A2DOpenLoopDataset,
        tensor_dtype,
    )

    model_path = Path(model_dir)
    config = InternVLAA1Config.from_pretrained(model_path)
    config.device = device
    config.dtype = dtype
    config.compile_model = compile_model
    config.attn_implementation = attn_implementation
    config.enable_regional_compile = enable_regional_compile

    train_meta = InternVLAA1TrainMetadata.from_pretrained(model_path)
    with open(model_path / "stats.json", encoding="utf-8") as f:
        train_stats = json.load(f)["a2d"]

    dataset = A2DOpenLoopDataset(
        dataset_dir,
        config=config,
        train_stats=train_stats,
    )
    return {
        "dataset": dataset,
        "config": config,
        "train_meta": train_meta,
        "processor_model_name": train_meta.processor_model_name,
        "device": device,
        "dtype": dtype,
        "torch_dtype": tensor_dtype(dtype),
    }


def build_observations(
    *,
    dataset: Any,
    config: Any,
    index: int,
    seed: int,
    device: str,
    dtype: Any,
    **_: Any,
) -> dict[str, Any]:
    """Build robot observation tensors and shared noise for one sample index."""

    from vllm_omni.diffusion.models.internvla_a1.config import OBS_STATE
    from vllm_omni.model_extras.internvla_a1_dataset import (
        collate_open_loop_samples,
        make_shared_noise,
        tensor_dtype,
    )

    sample = dataset.get_sample(index)
    torch_dtype = tensor_dtype(dtype) if isinstance(dtype, str) else dtype
    batch_inputs, metadata = collate_open_loop_samples([sample], device=device, dtype=torch_dtype)
    noise = make_shared_noise(
        seed,
        index,
        (
            batch_inputs[OBS_STATE].shape[0],
            config.chunk_size,
            config.max_action_dim,
        ),
        device,
    )
    return {
        "batch_inputs": batch_inputs,
        "noise": noise,
        "sample": sample,
        "metadata": metadata,
    }


def process_actions(
    *,
    pred: Any,
    dataset: Any,
    sample: Any | None = None,
    index: int | None = None,
    seed: int | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Truncate predicted actions to the physical action dim and summarize."""
    import torch

    from vllm_omni.model_extras.internvla_a1_dataset import tensor_sha256

    pred = pred[:, :, : dataset.physical_action_dim].to(torch.float32).cpu()
    result: dict[str, Any] = {
        "pred": pred,
        "shape": list(pred.shape),
        "mean": float(pred.mean().item()),
        "std": float(pred.std().item()),
        "action_sha256": tensor_sha256(pred),
        "first_action_prefix": pred[0, 0, :8].tolist(),
    }
    if sample is not None:
        result.update(
            {
                "index": sample.index if index is None else index,
                "episode_index": sample.episode_index,
                "task": sample.task,
            }
        )
    if seed is not None:
        result["seed"] = seed
    return result


def run_open_loop(
    *,
    policy: Any,
    dataset: Any,
    config: Any,
    train_meta: Any,
    run_sample_actions: Any,
    num_episodes: int,
    seed: int,
    device: str,
    dtype: Any,
    output_dir: str | Path,
    skip_plots: bool = False,
    mode: str = "vllm_registry",
    **_: Any,
) -> dict[str, Any]:
    """Run open-loop GT evaluation using InternVLA A2D helpers."""
    from vllm_omni.model_extras.internvla_a1_dataset import (
        collate_open_loop_samples,
        run_open_loop_evaluation,
        tensor_dtype,
    )

    torch_dtype = tensor_dtype(dtype) if isinstance(dtype, str) else dtype
    return run_open_loop_evaluation(
        mode=mode,
        policy=policy,
        config=config,
        dataset=dataset,
        train_meta=train_meta,
        collate_samples=collate_open_loop_samples,
        run_sample_actions=run_sample_actions,
        num_episodes=num_episodes,
        seed=seed,
        device=device,
        dtype=torch_dtype,
        output_dir=output_dir,
        skip_plots=skip_plots,
    )


__all__ = [
    "INTERNVLA_A1_EXTRA_BODY_PARAMS",
    "INTERNVLA_A1_EXTRA_OUTPUT_PARAMS",
    "build_observations",
    "load_eval_context",
    "process_actions",
    "run_open_loop",
]
