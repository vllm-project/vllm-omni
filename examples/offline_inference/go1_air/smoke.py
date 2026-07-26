#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal GO-1-Air offline-inference smoke test.

Loads (or stubs) the policy, runs a single forward pass on a fake batch,
and asserts the action chunk shape matches ``[1, chunk_size, action_dim]``.

Two modes:

* ``GO1_AIR_MODEL_DIR`` is set and points at a checkpoint directory
  containing ``config.json`` and ``model.safetensors[.index.json]`` —
  weights are loaded.
* Otherwise a no-checkpoint stub policy is constructed so the pipeline
  plumbing can still be exercised end-to-end (zero actions returned).

This script is intentionally tiny; the full open-loop evaluation harness
lives in a follow-up PR (mirroring ``examples/offline_inference/internvla_a1``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vllm_omni.diffusion.data import OmniDiffusionConfig  # noqa: E402
from vllm_omni.diffusion.models.go1_air import (  # noqa: E402
    Go1AirPipeline,
    get_go1_air_actions,
)
from vllm_omni.diffusion.models.go1_air.config import (  # noqa: E402
    OBS_IMAGES,
    OBS_STATE,
    OBS_TASK,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest  # noqa: E402
from vllm_omni.inputs.data import OmniDiffusionSamplingParams  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GO-1-Air smoke test.")
    parser.add_argument(
        "--model-dir",
        default=os.getenv("GO1_AIR_MODEL_DIR"),
        help="Path to a GO-1-Air checkpoint directory (optional).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16" if torch.cuda.is_available() else "float32",
    )
    parser.add_argument(
        "--tiny-config",
        action="store_true",
        help="Use a tiny CPU-friendly config for repository tests.",
    )
    return parser.parse_args()


def tiny_model_config() -> dict:
    return {
        "action_chunk_size": 2,
        "force_image_size": 28,
        "downsample_ratio": 0.5,
        "img_context_token_id": 3,
        "pad_token_id": 0,
        "vision_config": {
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "patch_size": 14,
            "image_size": 28,
            "qkv_bias": True,
        },
        "llm_config": {
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 64,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        "action_config": {
            "hidden_size": 8,
            "input_hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 64,
            "action_dim": 4,
            "state_dim": 4,
            "state_token_num": 1,
            "pad_token_id": 0,
        },
        "noise_scheduler_config": {
            "num_inference_timesteps": 1,
            "num_train_timesteps": 10,
            "beta_schedule": "squaredcos_cap_v2",
            "prediction_type": "sample",
        },
    }


def build_fake_batch(*, image_size: int, state_dim: int, device: str, dtype: torch.dtype):
    return {
        OBS_STATE: torch.zeros((1, state_dim), device=device, dtype=dtype),
        OBS_TASK: ["pick up the red block"],
        f"{OBS_IMAGES}.image0": torch.zeros(
            (1, 1, 3, image_size, image_size),
            device=device,
            dtype=dtype,
        ),
        f"{OBS_IMAGES}.image0_mask": torch.ones((1,), device=device, dtype=torch.bool),
    }


def main() -> int:
    args = parse_args()

    od_config = OmniDiffusionConfig(
        model_class_name="Go1AirPipeline",
        model=args.model_dir or "",
        dtype=args.dtype,
        model_config=tiny_model_config() if args.tiny_config else {},
        custom_pipeline_args={
            "device": args.device,
            "dtype": args.dtype,
        },
    )

    pipeline = Go1AirPipeline(od_config=od_config)
    cfg = pipeline.config
    print(f"[smoke] runtime_mode={pipeline.runtime_mode()}")
    print(f"[smoke] action chunk={cfg.chunk_size} action_dim={cfg.max_action_dim}")

    torch_dtype = getattr(torch, cfg.dtype)
    batch = build_fake_batch(
        image_size=cfg.image_resolution[0],
        state_dim=cfg.max_state_dim,
        device=cfg.device,
        dtype=torch_dtype,
    )

    sampling_params = OmniDiffusionSamplingParams(extra_args={"batch_inputs": batch})
    req = OmniDiffusionRequest(
        prompts=[""],
        sampling_params=sampling_params,
        request_id="go1-air-smoke",
    )

    output = pipeline.forward(req)
    if output.error:
        print(f"[smoke] FAIL: {output.error}")
        return 1

    actions = get_go1_air_actions(output)
    assert isinstance(actions, torch.Tensor), f"expected tensor, got {type(actions)}"
    expected = (1, cfg.chunk_size, cfg.max_action_dim)
    assert tuple(actions.shape) == expected, f"shape {tuple(actions.shape)} != {expected}"
    print(f"[smoke] OK action shape={tuple(actions.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
