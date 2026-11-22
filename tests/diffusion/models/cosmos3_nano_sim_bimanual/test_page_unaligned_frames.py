# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Frames the KV page size does not divide.

The default 720x1280 request is 23 x 40 + 4 = 924 tokens per frame, and the CUDA
and ROCm kernels page by multiples of 16, so the pool pages at 16 tokens while
still evicting whole 924-token frames.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from test_cookbook import manifest
from test_sampling_history import methods
from vllm.v1.attention.backend import MultipleOf

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.geometry import Cosmos3NanoSimBimanualGeometry
from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVCache, estimate_ar_diffusion_kv_cache_memory
from vllm_omni.experimental.ar_diffusion.kv_cache.config import ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.runner import paging_block_size

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PAGE = 16
DEFAULT = Cosmos3NanoSimBimanualGeometry(height=720, width=1280)  # 924 tokens per frame
SMALL = Cosmos3NanoSimBimanualGeometry(height=480, width=832)  # 394 tokens per frame
ALIGNED = Cosmos3NanoSimBimanualGeometry(height=128, width=224)  # 4 x 7 + 4 = 32 tokens per frame


def pipeline():
    pipe = methods()()
    pipe.manifest = replace(manifest(), window_frames=2, sink_frames=1)
    pipe._MAIN_BRANCH = "main"
    pipe._SESSION_CAPACITY = 1
    pipe.transformer = Mock(num_hidden_layers=2, num_kv_heads_local=1, head_dim=8)
    pipe._ar_diffusion_kv_state = None
    return pipe


def bound_state(pipe, geometry):
    """Build the pool the way the runner does: frame-sized chunks on kernel-legal pages."""
    spec = pipe._kv_spec_for_geometry(geometry)
    page = paging_block_size(spec.tokens_per_frame, [MultipleOf(PAGE)])
    config = ARDiffusionKVConfig(
        enable=True,
        chunk_size=spec.tokens_per_frame,
        window_chunks=spec.window_frames,
        sink_chunks=spec.sink_frames,
    )
    estimate = estimate_ar_diffusion_kv_cache_memory(config, spec, torch.float32, block_size=page)
    cache = ARDiffusionKVCache(
        config,
        num_layers=spec.num_layers,
        num_kv_heads=spec.num_kv_heads,
        head_size=spec.head_size,
        dtype=torch.float32,
        block_size=page,
        max_model_len=spec.max_model_len,
        available_bytes=estimate.required_bytes,
        kv_branches=spec.kv_branches,
        session_capacity=spec.session_capacity,
        cross_attention_lengths=spec.cross_attention_lengths,
        cross_attention_kv_heads=spec.cross_attention_kv_heads,
        frames_per_block=spec.frames_per_block,
        max_scratch_frames_per_branch=spec.max_scratch_frames_per_branch,
        max_scratch_tokens_per_branch=spec.max_scratch_tokens_per_branch,
        model_owned_state_bytes_per_session=spec.model_owned_state_bytes_per_session,
        device=torch.device("cpu"),
    )
    return SimpleNamespace(kv_cache=cache)


@pytest.mark.parametrize(("geometry", "tokens_per_frame"), [(DEFAULT, 924), (SMALL, 394)])
def test_bound_pool_is_checked_against_frame_size_not_page_size(geometry, tokens_per_frame):
    pipe = pipeline()
    state = bound_state(pipe, geometry)
    assert state.kv_cache.block_size == PAGE != tokens_per_frame
    assert state.kv_cache.spec.chunk_size == tokens_per_frame
    pipe._validate_bound_kv_geometry(state, geometry)


def test_pool_for_another_resolution_is_still_rejected():
    pipe = pipeline()
    state = bound_state(pipe, DEFAULT)
    with pytest.raises(RuntimeError, match=r"tokens_per_frame=expected 394, got 924"):
        pipe._validate_bound_kv_geometry(state, SMALL)


@pytest.mark.parametrize(
    ("geometry", "batched"),
    [(DEFAULT, False), (SMALL, False), (ALIGNED, True)],
)
def test_clean_commit_batches_only_page_aligned_frames(geometry, batched):
    pipe = pipeline()
    pipe._ar_diffusion_kv_state = bound_state(pipe, geometry)
    assert pipe._can_batch_clean_commit() is batched


def test_dense_history_always_batches():
    assert pipeline()._can_batch_clean_commit() is True
