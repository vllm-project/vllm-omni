# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Phase 5 CFG bypass in SenseNovaU1Pipeline.

Verifies that prepare_encode sets do_true_cfg=False and that per-request
CFG is handled internally via state.extra["p"].cfg_scale.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

HIDDEN_DIM = 32
PATCH_SIZE = 16
IMAGE_SIZE = (64, 64)
TOKEN_H = IMAGE_SIZE[1] // PATCH_SIZE
TOKEN_W = IMAGE_SIZE[0] // PATCH_SIZE
NUM_TOKENS = TOKEN_H * TOKEN_W
OUTPUT_DIM = 3 * PATCH_SIZE ** 2


def _make_pipeline():
    """Create a SenseNovaU1Pipeline stub without loading weights."""
    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
        SenseNovaU1Pipeline,
    )

    pipeline = object.__new__(SenseNovaU1Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.patch_size = PATCH_SIZE
    pipeline.merge_size = 1
    pipeline._step_states = {}
    pipeline.top_cfg = SimpleNamespace(
        add_noise_scale_embedding=False,
        noise_scale_max_value=1.0,
        use_pixel_head=False,
        t_eps=0.02,
    )

    mock_lm = MagicMock()
    mock_lm.config = SimpleNamespace(hidden_size=HIDDEN_DIM)
    pipeline.language_model = mock_lm

    pipeline.fm_modules = nn.ModuleDict({
        "timestep_embedder": nn.Identity(),
        "fm_head": nn.Linear(HIDDEN_DIM, OUTPUT_DIM, bias=False),
    })

    return pipeline


def _make_fake_state(cfg_scale=4.0):
    """Create a mock DiffusionRequestState with minimal fields."""
    sampling = SimpleNamespace(
        height=IMAGE_SIZE[1],
        width=IMAGE_SIZE[0],
        num_inference_steps=4,
        seed=42,
        extra_args={"cfg_scale": cfg_scale},
    )
    state = SimpleNamespace(
        sched_req_id="test_req",
        req_id="test_req",
        prompts=["test prompt"],
        sampling=sampling,
        extra={},
    )
    return state


class TestPrepareCFGBypass:

    def test_prepare_encode_sets_do_true_cfg_false(self) -> None:
        pipeline = _make_pipeline()
        state = _make_fake_state(cfg_scale=7.5)

        parsed_p = SimpleNamespace(
            first_prompt="test prompt",
            prompt="test prompt",
            extra_args={},
            image_size=IMAGE_SIZE,
            num_steps=4,
            cfg_scale=7.5,
            img_cfg_scale=1.0,
            cfg_norm="none",
            timestep_shift=3.0,
            cfg_interval=(0.0, 1.0),
            batch_size=1,
            seed=42,
            think_mode=False,
            t_eps=0.02,
        )
        ns = SimpleNamespace(
            grid_h=TOKEN_H,
            grid_w=TOKEN_W,
            grid_hw=(TOKEN_H, TOKEN_W),
            token_h=TOKEN_H,
            token_w=TOKEN_W,
            timesteps=torch.linspace(1.0, 0.0, 5),
            noise_scale=1.0,
            merge_size=1,
            image_prediction=torch.randn(1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0]),
        )
        caches = {
            "cond": MagicMock(),
            "uncond": MagicMock(),
            "idx_cond": torch.zeros(3, NUM_TOKENS, dtype=torch.long),
            "idx_uncond": torch.zeros(3, NUM_TOKENS, dtype=torch.long),
        }

        pipeline._parse_request_from_state = MagicMock(return_value=parsed_p)
        pipeline._init_noise_and_schedule = MagicMock(return_value=ns)
        pipeline._extract_input_images = MagicMock(return_value=None)
        pipeline._build_t2i_caches = MagicMock(return_value=caches)

        pipeline.prepare_encode(state)

        assert hasattr(state, "do_true_cfg")
        assert state.do_true_cfg is False

    def test_prepare_encode_stores_cfg_scale_in_extra(self) -> None:
        pipeline = _make_pipeline()
        state = _make_fake_state(cfg_scale=5.0)

        parsed_p = SimpleNamespace(
            first_prompt="test prompt",
            prompt="test prompt",
            extra_args={},
            image_size=IMAGE_SIZE,
            num_steps=4,
            cfg_scale=5.0,
            img_cfg_scale=1.0,
            cfg_norm="none",
            timestep_shift=3.0,
            cfg_interval=(0.0, 1.0),
            batch_size=1,
            seed=42,
            think_mode=False,
            t_eps=0.02,
        )
        ns = SimpleNamespace(
            grid_h=TOKEN_H,
            grid_w=TOKEN_W,
            grid_hw=(TOKEN_H, TOKEN_W),
            token_h=TOKEN_H,
            token_w=TOKEN_W,
            timesteps=torch.linspace(1.0, 0.0, 5),
            noise_scale=1.0,
            merge_size=1,
            image_prediction=torch.randn(1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0]),
        )
        caches = {
            "cond": MagicMock(),
            "uncond": MagicMock(),
            "idx_cond": torch.zeros(3, NUM_TOKENS, dtype=torch.long),
            "idx_uncond": torch.zeros(3, NUM_TOKENS, dtype=torch.long),
        }

        pipeline._parse_request_from_state = MagicMock(return_value=parsed_p)
        pipeline._init_noise_and_schedule = MagicMock(return_value=ns)
        pipeline._extract_input_images = MagicMock(return_value=None)
        pipeline._build_t2i_caches = MagicMock(return_value=caches)

        pipeline.prepare_encode(state)

        assert "p" in state.extra
        assert state.extra["p"].cfg_scale == 5.0

    def test_cfg_scale_one_still_sets_do_true_cfg_false(self) -> None:
        """Even with cfg_scale=1 (no CFG needed), do_true_cfg is False."""
        pipeline = _make_pipeline()
        state = _make_fake_state(cfg_scale=1.0)

        parsed_p = SimpleNamespace(
            first_prompt="test prompt",
            prompt="test prompt",
            extra_args={},
            image_size=IMAGE_SIZE,
            num_steps=4,
            cfg_scale=1.0,
            img_cfg_scale=1.0,
            cfg_norm="none",
            timestep_shift=3.0,
            cfg_interval=(0.0, 1.0),
            batch_size=1,
            seed=42,
            think_mode=False,
            t_eps=0.02,
        )
        ns = SimpleNamespace(
            grid_h=TOKEN_H,
            grid_w=TOKEN_W,
            grid_hw=(TOKEN_H, TOKEN_W),
            token_h=TOKEN_H,
            token_w=TOKEN_W,
            timesteps=torch.linspace(1.0, 0.0, 5),
            noise_scale=1.0,
            merge_size=1,
            image_prediction=torch.randn(1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0]),
        )
        caches = {}

        pipeline._parse_request_from_state = MagicMock(return_value=parsed_p)
        pipeline._init_noise_and_schedule = MagicMock(return_value=ns)
        pipeline._extract_input_images = MagicMock(return_value=None)
        pipeline._build_t2i_caches = MagicMock(return_value=caches)

        pipeline.prepare_encode(state)

        assert state.do_true_cfg is False
        assert state.extra["p"].cfg_scale == 1.0


class TestCFGHandledInternally:

    def test_batched_denoise_uses_extra_cfg_scale_not_do_true_cfg(self) -> None:
        """_batched_denoise_step reads cfg_scale from extra['p'], not do_true_cfg."""
        from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
            SenseNovaU1Pipeline,
        )

        pipeline = _make_pipeline()

        timesteps = torch.linspace(1.0, 0.0, 5)[:4]
        for req_id, cfg in [("a", 4.0), ("b", 1.0)]:
            pipeline._step_states[req_id] = {
                "p": SimpleNamespace(
                    batch_size=1,
                    image_size=IMAGE_SIZE,
                    cfg_scale=cfg,
                    cfg_norm="none",
                    cfg_interval=(0.0, 1.0),
                ),
                "ns": SimpleNamespace(
                    grid_h=TOKEN_H,
                    grid_w=TOKEN_W,
                    grid_hw=(TOKEN_H, TOKEN_W),
                    token_h=TOKEN_H,
                    token_w=TOKEN_W,
                    timesteps=timesteps,
                    noise_scale=1.0,
                    merge_size=1,
                ),
                "caches": {
                    "cond": MagicMock(),
                    "uncond": MagicMock(),
                    "idx_cond": torch.zeros(3, NUM_TOKENS, dtype=torch.long),
                    "idx_uncond": torch.zeros(3, NUM_TOKENS, dtype=torch.long),
                },
                "_image_prediction": torch.randn(1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0]),
                "_current_step_index": 0,
            }

        pipeline._prepare_single_embeds = MagicMock(
            side_effect=lambda extra: (
                torch.randn(1, NUM_TOKENS, HIDDEN_DIM),
                torch.zeros(3, NUM_TOKENS, dtype=torch.long),
            )
        )

        predict_v_calls = []

        def mock_predict_v(embeds, indexes, data, kv_key):
            predict_v_calls.append((kv_key, len(data)))
            return [torch.randn(1, NUM_TOKENS, OUTPUT_DIM) for _ in data]

        pipeline._batched_predict_v = mock_predict_v

        class _StubBatch:
            req_ids = ["a", "b"]

        pipeline._batched_denoise_step(_StubBatch())

        assert predict_v_calls[0] == ("cond", 2)
        assert predict_v_calls[1] == ("uncond", 1)
