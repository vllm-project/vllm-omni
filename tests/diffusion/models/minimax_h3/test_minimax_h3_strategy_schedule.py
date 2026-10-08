# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full-forward capture, compilation and replay with CPU or real CUDA providers."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from vllm_omni.diffusion.attention.strategy import AttentionOperation, ForwardStrategyPlan
from vllm_omni.diffusion.data import AttentionConfig
from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def test_minimax_recipe_schedule_and_dense_refiner():
    config = AttentionConfig(
        **json.loads(
            (Path(__file__).resolve().parents[4] / "recipes/attention/minimax-h3-subblock-strategy.json").read_text()
        )
    )
    strategy = config.strategy
    operations = [
        AttentionOperation(f"{role}.{i}", i, f"minimax_h3.{role}", "self", component="transformer")
        for role, count in (("dit", 50), ("token_refiner", 2))
        for i in range(count)
    ]
    strategy.validate_inventory(operations)
    for total in (4, 20, 21, 35, 50):
        for step in range(total):
            layout = next(
                index for end, index in ForwardStrategyPlan.from_strategy(strategy).resolve_steps(total) if step < end
            )
            for operation in operations:
                selected = strategy.assignments(operation)[layout]
                assert (getattr(selected, "name", None) == "block_sparse") == (
                    step >= 20 and operation.role == "minimax_h3.dit"
                )
                if operation.role == "minimax_h3.token_refiner":
                    assert selected.backend == "FLASH_ATTN"
    restored = AttentionConfig(**asdict(config))
    assert ForwardStrategyPlan.from_strategy(restored.strategy).resolve_steps(35) == ForwardStrategyPlan.from_strategy(
        strategy
    ).resolve_steps(35)


def test_minimax_exception_clears_strategy_progress():
    from tests.diffusion.models.minimax_h3.test_minimax_h3_step_execution import _make_branch
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import minimax_h3_denoise_loop

    branch, video, audio = _make_branch(text_len=3, latent_t=2, latent_h=2, latent_w=2, audio_t=3, seed=7)
    context = ForwardContext()

    def interrupted_model(**kwargs):
        assert context.denoise_step_idx == 0 and context.total_denoise_steps == 2
        context.attention_strategy_schedule = (object(), 2, ((2, 0),))
        raise RuntimeError("interrupted")

    with override_forward_context(context), pytest.raises(RuntimeError, match="interrupted"):
        minimax_h3_denoise_loop(
            model=interrupted_model,
            positive=branch,
            initial_video_rows=video,
            initial_audio_rows=audio,
            keyframe_cond_rows=None,
            sigmas_video=[1.0, 0.5, 0.0],
            sigmas_audio=[1.0, 0.5, 0.0],
            device=torch.device("cpu"),
        )
    assert context.denoise_step_idx is None and context.total_denoise_steps is None
    assert context.attention_strategy_schedule is None
