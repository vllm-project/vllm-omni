# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SenseNovaU1Pipeline batching methods.

Tests denoise_step dispatch, _prepare_single_embeds, _batched_predict_v,
and _batched_denoise_step without loading model weights.
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
TOKEN_H = IMAGE_SIZE[1] // PATCH_SIZE  # 4
TOKEN_W = IMAGE_SIZE[0] // PATCH_SIZE  # 4
NUM_TOKENS = TOKEN_H * TOKEN_W  # 16
OUTPUT_DIM = 3 * PATCH_SIZE ** 2  # 768


# ============================================================
# Stubs
# ============================================================


class _FnModule(nn.Module):
    """Wrap a callable as an nn.Module (for use inside ModuleDict)."""
    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def forward(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _MockCacheLayer:
    def __init__(self, prefix_len):
        self.flash_prefix_len = prefix_len
        self.flash_k_cache = torch.randn(1, prefix_len + 64, 2, 8)
        self.flash_v_cache = torch.randn(1, prefix_len + 64, 2, 8)


class _MockDynamicCache:
    def __init__(self, num_layers, prefix_len):
        self.layers = [_MockCacheLayer(prefix_len) for _ in range(num_layers)]


class _StubInputBatch:
    def __init__(self, req_ids):
        self.req_ids = req_ids


def _make_step_state(
    req_id,
    *,
    image_size=IMAGE_SIZE,
    cfg_scale=4.0,
    cfg_norm="none",
    cfg_interval=(0.0, 1.0),
    num_steps=4,
    step_index=0,
    prefix_len=10,
    batch_size=1,
    noise_scale=1.0,
    merge_size=1,
):
    """Build a fake _step_states entry for one request."""
    token_h = image_size[1] // PATCH_SIZE
    token_w = image_size[0] // PATCH_SIZE
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)[:num_steps]

    p = SimpleNamespace(
        batch_size=batch_size,
        image_size=image_size,
        cfg_scale=cfg_scale,
        cfg_norm=cfg_norm,
        cfg_interval=cfg_interval,
    )
    ns = SimpleNamespace(
        grid_h=token_h,
        grid_w=token_w,
        grid_hw=(token_h, token_w),
        token_h=token_h,
        token_w=token_w,
        timesteps=timesteps,
        noise_scale=noise_scale,
        merge_size=merge_size,
    )
    caches = {
        "cond": _MockDynamicCache(1, prefix_len),
        "uncond": _MockDynamicCache(1, prefix_len),
        "idx_cond": torch.zeros(3, token_h * token_w, dtype=torch.long),
        "idx_uncond": torch.zeros(3, token_h * token_w, dtype=torch.long),
    }
    image_prediction = torch.randn(batch_size, 3, image_size[1], image_size[0])

    return {
        "p": p,
        "ns": ns,
        "caches": caches,
        "_image_prediction": image_prediction,
        "_current_step_index": step_index,
    }


def _make_pipeline():
    """Create a SenseNovaU1Pipeline stub without loading weights."""
    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
        SenseNovaU1Pipeline,
    )

    pipeline = object.__new__(SenseNovaU1Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.patch_size = PATCH_SIZE
    pipeline._step_states = {}
    pipeline.top_cfg = SimpleNamespace(
        add_noise_scale_embedding=False,
        noise_scale_max_value=1.0,
        use_pixel_head=False,
        t_eps=0.02,
    )

    # Stub language_model with forward_varlen
    mock_lm = MagicMock()
    pipeline.language_model = mock_lm

    # Stub fm_modules
    pipeline.fm_modules = nn.ModuleDict({
        "timestep_embedder": nn.Identity(),
        "fm_head": nn.Linear(HIDDEN_DIM, OUTPUT_DIM, bias=False),
    })

    # Stub _extract_feature to return identity-like output
    pipeline._extract_feature = lambda x, gen_model=False, grid_hw=None: torch.randn(
        x.shape[0], HIDDEN_DIM
    )

    return pipeline


# ============================================================
# denoise_step dispatch tests
# ============================================================


class TestDenoiseStepDispatch:

    def test_single_request_calls_fast_path(self) -> None:
        pipeline = _make_pipeline()
        state = _make_step_state("req_0")
        pipeline._step_states["req_0"] = state

        v_pred = torch.randn(1, NUM_TOKENS, OUTPUT_DIM)
        pipeline._step_denoise_single = MagicMock(return_value=v_pred)
        pipeline._batched_denoise_step = MagicMock()

        ib = _StubInputBatch(["req_0"])
        result = pipeline.denoise_step(ib)

        pipeline._step_denoise_single.assert_called_once()
        pipeline._batched_denoise_step.assert_not_called()
        assert result is v_pred

    def test_multiple_requests_calls_batched(self) -> None:
        pipeline = _make_pipeline()
        for i in range(3):
            pipeline._step_states[f"req_{i}"] = _make_step_state(f"req_{i}")

        expected = torch.randn(3, NUM_TOKENS, OUTPUT_DIM)
        pipeline._batched_denoise_step = MagicMock(return_value=expected)

        ib = _StubInputBatch(["req_0", "req_1", "req_2"])
        result = pipeline.denoise_step(ib)

        pipeline._batched_denoise_step.assert_called_once_with(ib)
        assert result is expected

    def test_missing_state_raises_valueerror(self) -> None:
        pipeline = _make_pipeline()
        ib = _StubInputBatch(["nonexistent"])

        with pytest.raises(ValueError, match="No step state found"):
            pipeline.denoise_step(ib)


# ============================================================
# _prepare_single_embeds tests
# ============================================================


class TestPrepareSingleEmbeds:

    def test_adds_timestep_embedding(self) -> None:
        pipeline = _make_pipeline()
        state = _make_step_state("req", step_index=0)

        extract_output = torch.ones(1, NUM_TOKENS, HIDDEN_DIM) * 2.0
        pipeline._extract_feature = MagicMock(return_value=extract_output.view(NUM_TOKENS, HIDDEN_DIM))

        ts_output = torch.ones(NUM_TOKENS, HIDDEN_DIM) * 3.0
        pipeline.fm_modules["timestep_embedder"] = _FnModule(lambda x: ts_output)

        embeds, indexes = pipeline._prepare_single_embeds(state)

        assert embeds.shape == (1, NUM_TOKENS, HIDDEN_DIM)
        expected = torch.ones(1, NUM_TOKENS, HIDDEN_DIM) * 5.0  # 2 + 3
        torch.testing.assert_close(embeds, expected)

    def test_adds_noise_scale_embedding_when_configured(self) -> None:
        pipeline = _make_pipeline()
        pipeline.top_cfg.add_noise_scale_embedding = True
        state = _make_step_state("req", step_index=0, noise_scale=0.5)

        extract_output = torch.zeros(NUM_TOKENS, HIDDEN_DIM)
        pipeline._extract_feature = MagicMock(return_value=extract_output)

        ts_output = torch.ones(NUM_TOKENS, HIDDEN_DIM)
        pipeline.fm_modules["timestep_embedder"] = _FnModule(lambda x: ts_output)

        ns_output = torch.ones(NUM_TOKENS, HIDDEN_DIM) * 0.5
        pipeline.fm_modules["noise_scale_embedder"] = _FnModule(lambda x: ns_output)

        embeds, _ = pipeline._prepare_single_embeds(state)

        # 0 + 1 + 0.5 = 1.5
        expected_val = 1.5
        torch.testing.assert_close(embeds, torch.full((1, NUM_TOKENS, HIDDEN_DIM), expected_val))

    def test_returns_correct_indexes(self) -> None:
        pipeline = _make_pipeline()
        state = _make_step_state("req")
        expected_indexes = state["caches"]["idx_cond"]

        pipeline._extract_feature = MagicMock(return_value=torch.randn(NUM_TOKENS, HIDDEN_DIM))
        pipeline.fm_modules["timestep_embedder"] = _FnModule(
            lambda x: torch.zeros(NUM_TOKENS, HIDDEN_DIM)
        )

        _, indexes = pipeline._prepare_single_embeds(state)
        assert indexes is expected_indexes


# ============================================================
# _batched_predict_v tests
# ============================================================


class TestBatchedPredictV:

    def test_packs_embeds_correctly(self) -> None:
        pipeline = _make_pipeline()

        embeds_list = [
            torch.ones(1, NUM_TOKENS, HIDDEN_DIM) * 1.0,
            torch.ones(1, NUM_TOKENS, HIDDEN_DIM) * 2.0,
        ]
        indexes_list = [
            torch.zeros(3, NUM_TOKENS, dtype=torch.long),
            torch.zeros(3, NUM_TOKENS, dtype=torch.long),
        ]

        captured = {}

        def mock_forward_varlen(**kwargs):
            captured.update(kwargs)
            total_s = kwargs["inputs_embeds"].shape[1]
            return SimpleNamespace(
                hidden_states=torch.randn(1, total_s, HIDDEN_DIM)
            )

        pipeline.language_model.forward_varlen = mock_forward_varlen

        per_req_data = [
            _make_step_state("a", prefix_len=5),
            _make_step_state("b", prefix_len=8),
        ]

        pipeline._batched_predict_v(embeds_list, indexes_list, per_req_data, "cond")

        packed = captured["inputs_embeds"]
        assert packed.shape == (1, NUM_TOKENS * 2, HIDDEN_DIM)
        torch.testing.assert_close(packed[0, :NUM_TOKENS], torch.ones(NUM_TOKENS, HIDDEN_DIM) * 1.0)
        torch.testing.assert_close(packed[0, NUM_TOKENS:], torch.ones(NUM_TOKENS, HIDDEN_DIM) * 2.0)

    def test_builds_cu_seqlens_correctly(self) -> None:
        pipeline = _make_pipeline()

        prefix_lens = [3, 5]
        embeds_list = [torch.randn(1, NUM_TOKENS, HIDDEN_DIM) for _ in prefix_lens]
        indexes_list = [torch.zeros(3, NUM_TOKENS, dtype=torch.long) for _ in prefix_lens]

        captured = {}

        def mock_forward_varlen(**kwargs):
            captured.update(kwargs)
            total_s = kwargs["inputs_embeds"].shape[1]
            return SimpleNamespace(hidden_states=torch.randn(1, total_s, HIDDEN_DIM))

        pipeline.language_model.forward_varlen = mock_forward_varlen

        per_req_data = [
            _make_step_state("a", prefix_len=prefix_lens[0]),
            _make_step_state("b", prefix_len=prefix_lens[1]),
        ]

        pipeline._batched_predict_v(embeds_list, indexes_list, per_req_data, "cond")

        cu_q = captured["cu_seqlens_q"]
        cu_k = captured["cu_seqlens_k"]
        assert cu_q.tolist() == [0, NUM_TOKENS, NUM_TOKENS * 2]
        assert cu_k.tolist() == [0, 3 + NUM_TOKENS, 3 + NUM_TOKENS + 5 + NUM_TOKENS]

    def test_returns_per_request_results(self) -> None:
        pipeline = _make_pipeline()

        q_lens = [NUM_TOKENS, NUM_TOKENS]
        embeds_list = [torch.randn(1, q, HIDDEN_DIM) for q in q_lens]
        indexes_list = [torch.zeros(3, q, dtype=torch.long) for q in q_lens]

        def mock_forward_varlen(**kwargs):
            total_s = kwargs["inputs_embeds"].shape[1]
            return SimpleNamespace(hidden_states=torch.randn(1, total_s, HIDDEN_DIM))

        pipeline.language_model.forward_varlen = mock_forward_varlen

        per_req_data = [
            _make_step_state("a", prefix_len=5),
            _make_step_state("b", prefix_len=8),
        ]

        results = pipeline._batched_predict_v(embeds_list, indexes_list, per_req_data, "cond")

        assert len(results) == 2
        for r in results:
            assert r.shape[0] == 1


# ============================================================
# _batched_denoise_step tests
# ============================================================


class TestBatchedDenoiseStep:

    def _setup_pipeline_for_batch(self, pipeline, req_ids, cfg_scales=None, step_index=0):
        if cfg_scales is None:
            cfg_scales = [4.0] * len(req_ids)

        for i, req_id in enumerate(req_ids):
            pipeline._step_states[req_id] = _make_step_state(
                req_id, cfg_scale=cfg_scales[i], step_index=step_index,
            )

        # Mock _prepare_single_embeds to return known tensors
        pipeline._prepare_single_embeds = MagicMock(
            side_effect=lambda extra: (
                torch.randn(1, NUM_TOKENS, HIDDEN_DIM),
                torch.zeros(3, NUM_TOKENS, dtype=torch.long),
            )
        )

    def test_cond_only_when_no_cfg(self) -> None:
        pipeline = _make_pipeline()
        req_ids = ["a", "b"]
        self._setup_pipeline_for_batch(pipeline, req_ids, cfg_scales=[1.0, 1.0])

        predict_v_calls = []

        def mock_predict_v(embeds, indexes, data, kv_key):
            predict_v_calls.append(kv_key)
            return [torch.randn(1, NUM_TOKENS, OUTPUT_DIM) for _ in data]

        pipeline._batched_predict_v = mock_predict_v

        ib = _StubInputBatch(req_ids)
        result = pipeline._batched_denoise_step(ib)

        assert predict_v_calls == ["cond"]
        assert result.shape[0] == 2

    def test_uncond_only_for_cfg_requests(self) -> None:
        pipeline = _make_pipeline()
        req_ids = ["a", "b", "c"]
        self._setup_pipeline_for_batch(pipeline, req_ids, cfg_scales=[4.0, 1.0, 7.5])

        predict_v_calls = []
        predict_v_data_lens = []

        def mock_predict_v(embeds, indexes, data, kv_key):
            predict_v_calls.append(kv_key)
            predict_v_data_lens.append(len(data))
            return [torch.randn(1, NUM_TOKENS, OUTPUT_DIM) for _ in data]

        pipeline._batched_predict_v = mock_predict_v

        ib = _StubInputBatch(req_ids)
        pipeline._batched_denoise_step(ib)

        assert predict_v_calls == ["cond", "uncond"]
        assert predict_v_data_lens[0] == 3  # cond for all
        assert predict_v_data_lens[1] == 2  # uncond only for a and c

    def test_cfg_combination_standard(self) -> None:
        pipeline = _make_pipeline()
        req_ids = ["a"]
        self._setup_pipeline_for_batch(pipeline, req_ids, cfg_scales=[4.0])

        cond_v = torch.ones(1, NUM_TOKENS, OUTPUT_DIM) * 2.0
        uncond_v = torch.ones(1, NUM_TOKENS, OUTPUT_DIM) * 1.0

        call_idx = {"n": 0}

        def mock_predict_v(embeds, indexes, data, kv_key):
            call_idx["n"] += 1
            if kv_key == "cond":
                return [cond_v]
            return [uncond_v]

        pipeline._batched_predict_v = mock_predict_v
        pipeline._prepare_single_embeds = MagicMock(
            return_value=(torch.randn(1, NUM_TOKENS, HIDDEN_DIM), torch.zeros(3, NUM_TOKENS, dtype=torch.long))
        )
        pipeline._apply_cfg_norm = staticmethod(lambda v, c, norm: v)

        ib = _StubInputBatch(req_ids)
        result = pipeline._batched_denoise_step(ib)

        # v_pred = uncond + cfg_scale * (cond - uncond) = 1 + 4*(2-1) = 5
        expected = torch.ones(1, NUM_TOKENS, OUTPUT_DIM) * 5.0
        torch.testing.assert_close(result, expected)

    def test_cfg_interval_respected(self) -> None:
        pipeline = _make_pipeline()
        req_ids = ["a"]

        state = _make_step_state("a", cfg_scale=4.0, cfg_interval=(0.5, 1.0), step_index=3, num_steps=4)
        # timestep at step 3 should be near 0, below cfg_interval[0]=0.5
        pipeline._step_states["a"] = state

        predict_v_calls = []

        def mock_predict_v(embeds, indexes, data, kv_key):
            predict_v_calls.append(kv_key)
            return [torch.randn(1, NUM_TOKENS, OUTPUT_DIM) for _ in data]

        pipeline._batched_predict_v = mock_predict_v
        pipeline._prepare_single_embeds = MagicMock(
            return_value=(torch.randn(1, NUM_TOKENS, HIDDEN_DIM), torch.zeros(3, NUM_TOKENS, dtype=torch.long))
        )

        ib = _StubInputBatch(req_ids)
        pipeline._batched_denoise_step(ib)

        # timestep at step_index=3 with num_steps=4 is 0.25, below cfg_interval[0]=0.5
        # so uncond should be skipped
        assert predict_v_calls == ["cond"]

    def test_concatenates_v_preds_in_order(self) -> None:
        pipeline = _make_pipeline()
        req_ids = ["a", "b", "c"]
        self._setup_pipeline_for_batch(pipeline, req_ids, cfg_scales=[1.0, 1.0, 1.0])

        def mock_predict_v(embeds, indexes, data, kv_key):
            return [torch.ones(1, NUM_TOKENS, OUTPUT_DIM) * (i + 1) for i in range(len(data))]

        pipeline._batched_predict_v = mock_predict_v

        ib = _StubInputBatch(req_ids)
        result = pipeline._batched_denoise_step(ib)

        assert result.shape == (3, NUM_TOKENS, OUTPUT_DIM)
        torch.testing.assert_close(result[0], torch.ones(NUM_TOKENS, OUTPUT_DIM) * 1.0)
        torch.testing.assert_close(result[1], torch.ones(NUM_TOKENS, OUTPUT_DIM) * 2.0)
        torch.testing.assert_close(result[2], torch.ones(NUM_TOKENS, OUTPUT_DIM) * 3.0)

    def test_raises_on_missing_state(self) -> None:
        pipeline = _make_pipeline()
        ib = _StubInputBatch(["missing_req"])

        with pytest.raises(ValueError, match="No step state found"):
            pipeline._batched_denoise_step(ib)

    def test_mixed_cfg_produces_different_results(self) -> None:
        pipeline = _make_pipeline()
        req_ids = ["a", "b"]
        self._setup_pipeline_for_batch(pipeline, req_ids, cfg_scales=[4.0, 1.0])

        cond_vals = [
            torch.ones(1, NUM_TOKENS, OUTPUT_DIM) * 2.0,
            torch.ones(1, NUM_TOKENS, OUTPUT_DIM) * 2.0,
        ]
        uncond_val = torch.ones(1, NUM_TOKENS, OUTPUT_DIM) * 1.0

        def mock_predict_v(embeds, indexes, data, kv_key):
            if kv_key == "cond":
                return cond_vals[:len(data)]
            return [uncond_val]

        pipeline._batched_predict_v = mock_predict_v
        pipeline._apply_cfg_norm = staticmethod(lambda v, c, norm: v)

        ib = _StubInputBatch(req_ids)
        result = pipeline._batched_denoise_step(ib)

        # req_a: uncond + 4*(cond - uncond) = 1 + 4*1 = 5
        # req_b: cond only = 2
        torch.testing.assert_close(result[0], torch.ones(NUM_TOKENS, OUTPUT_DIM) * 5.0)
        torch.testing.assert_close(result[1], torch.ones(NUM_TOKENS, OUTPUT_DIM) * 2.0)
