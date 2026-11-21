# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Bit-exactness tests for the audex causal speech decoder CUDA-graph runner.

The runner replays graphed steady-state chunks (full emit width, complete
lookahead window) against a static shared KV workspace, keeping the session's
stock ``CausalCodecDecoderCache`` authoritative via copy-in/copy-out. Every
test compares a graphed session against an eager session on the same model
and frame schedule and requires bit-identical output — including utterances
that cross KV bucket boundaries, mixed graph/eager fallback past the frame
cap, and concurrent interleaved sessions.
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.audex.speech_decoder import (
    AudexCausalSpeechDecoderConfig,
    AudexCausalSpeechDecoderModel,
)
from vllm_omni.model_executor.models.audex.speech_decoder.graph_runner import (
    AudexCausalSpeechDecoderGraphRunner,
)

pytestmark = [pytest.mark.core_model]

CHUNK_FRAMES = 5
BURST = 15  # frames per push, like the 300 ms connector window at 50 fps


def _real_model() -> AudexCausalSpeechDecoderModel:
    torch.manual_seed(0)
    config = AudexCausalSpeechDecoderConfig()
    return AudexCausalSpeechDecoderModel(config).to(device="cuda", dtype=torch.float32).eval()


def _codes(n_frames: int, levels: int = 65536, seed: int = 7) -> list[list[int]]:
    gen = torch.Generator().manual_seed(seed)
    return [[int(c)] for c in torch.randint(0, levels, (n_frames,), generator=gen).tolist()]


def _run_session(model, frames: list[list[int]], *, graph_runner=None) -> torch.Tensor:
    session = model.create_session(chunk_frames=CHUNK_FRAMES, return_numpy=False, graph_runner=graph_runner)
    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for i in range(0, len(frames), BURST):
            for _sr, chunk in session.push(frames[i : i + BURST]):
                chunks.append(chunk.reshape(-1))
        for _sr, chunk in session.flush():
            chunks.append(chunk.reshape(-1))
    return torch.cat(chunks).to(torch.float32)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("n_frames", [9, 23, 45, 130, 500])
def test_graphed_session_bit_exact_vs_eager(n_frames: int) -> None:
    model = _real_model()
    runner = AudexCausalSpeechDecoderGraphRunner(model, emit_frames=CHUNK_FRAMES)
    frames = _codes(n_frames)

    eager = _run_session(model, frames)
    graphed = _run_session(model, frames, graph_runner=runner)

    assert graphed.numel() == n_frames * model.config.hop_length
    assert torch.equal(eager, graphed)
    # the graph path must actually have engaged (not silently fallen back)
    assert runner._graphs, "no graphs were captured"


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_graph_runner_falls_back_past_frame_cap_bit_exact() -> None:
    model = _real_model()
    # cap at one bucket: chunks up to position 64 replay, the rest run eager
    runner = AudexCausalSpeechDecoderGraphRunner(model, emit_frames=CHUNK_FRAMES, max_frames=64, bucket_stride=64)
    frames = _codes(100)

    eager = _run_session(model, frames)
    mixed = _run_session(model, frames, graph_runner=runner)

    assert runner.bucket_for(64) == 64
    assert runner.bucket_for(65) is None
    assert torch.equal(eager, mixed)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_interleaved_sessions_bit_exact() -> None:
    model = _real_model()
    runner = AudexCausalSpeechDecoderGraphRunner(model, emit_frames=CHUNK_FRAMES)
    frames_a = _codes(75, seed=1)
    frames_b = _codes(75, seed=2)

    ref_a = _run_session(model, frames_a)
    ref_b = _run_session(model, frames_b)

    # two sessions alternating pushes through the shared runner/workspace
    session_a = model.create_session(chunk_frames=CHUNK_FRAMES, return_numpy=False, graph_runner=runner)
    session_b = model.create_session(chunk_frames=CHUNK_FRAMES, return_numpy=False, graph_runner=runner)
    chunks_a: list[torch.Tensor] = []
    chunks_b: list[torch.Tensor] = []
    with torch.inference_mode():
        for i in range(0, 75, BURST):
            for _sr, chunk in session_a.push(frames_a[i : i + BURST]):
                chunks_a.append(chunk.reshape(-1))
            for _sr, chunk in session_b.push(frames_b[i : i + BURST]):
                chunks_b.append(chunk.reshape(-1))
        for _sr, chunk in session_a.flush():
            chunks_a.append(chunk.reshape(-1))
        for _sr, chunk in session_b.flush():
            chunks_b.append(chunk.reshape(-1))

    assert torch.equal(ref_a, torch.cat(chunks_a).to(torch.float32))
    assert torch.equal(ref_b, torch.cat(chunks_b).to(torch.float32))


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_runner_ignored_when_emit_width_mismatches() -> None:
    model = _real_model()
    runner = AudexCausalSpeechDecoderGraphRunner(model, emit_frames=CHUNK_FRAMES + 1)
    frames = _codes(40)

    ref = _run_session(model, frames)
    session = model.create_session(chunk_frames=CHUNK_FRAMES, return_numpy=False, graph_runner=runner)
    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for i in range(0, len(frames), BURST):
            for _sr, chunk in session.push(frames[i : i + BURST]):
                chunks.append(chunk.reshape(-1))
        for _sr, chunk in session.flush():
            chunks.append(chunk.reshape(-1))

    assert torch.equal(ref, torch.cat(chunks).to(torch.float32))
    assert not runner._graphs  # never engaged


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_runner_rejects_non_cuda_and_bad_config() -> None:
    model = _real_model()
    with pytest.raises(ValueError, match="CUDA device"):
        AudexCausalSpeechDecoderGraphRunner(model, emit_frames=CHUNK_FRAMES, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="emit_frames must be positive"):
        AudexCausalSpeechDecoderGraphRunner(model, emit_frames=0)
    with pytest.raises(ValueError, match="max_frames must be a multiple"):
        AudexCausalSpeechDecoderGraphRunner(model, emit_frames=CHUNK_FRAMES, max_frames=100, bucket_stride=64)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_bucket_math() -> None:
    model = _real_model()
    runner = AudexCausalSpeechDecoderGraphRunner(model, emit_frames=CHUNK_FRAMES)
    assert runner.bucket_for(5) == 64
    assert runner.bucket_for(64) == 64
    assert runner.bucket_for(65) == 128
    assert runner.bucket_for(1024) == 1024
    assert runner.bucket_for(1025) is None
    assert math.prod(model.config.codebook_levels) == model.config.codebook_size
