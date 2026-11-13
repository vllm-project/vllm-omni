# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass, field, replace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.zonos3 import (
    Zonos3ChunkConfig,
    Zonos3RVQStageInputProcessor,
    zonos3_rvq_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CONFIG = Zonos3ChunkConfig(codebook_sizes=(4096,) + (1024,) * 9, eoa_id=4096, pad_id=4097)


def delayed(frames: torch.Tensor) -> torch.Tensor:
    """Independent fixture: place frame f's codebook j at row f+j."""
    rows = torch.full((len(frames) + 10, 10), CONFIG.pad_id, dtype=torch.int64)
    for f, frame in enumerate(frames):
        for j in range(10):
            rows[f + j, j] = frame[j]
    for j in range(10):
        rows[len(frames) + j, j] = CONFIG.eoa_id
    return rows


@pytest.mark.parametrize("n", [0, 1, 2, 9, 10, 17, 100])
@pytest.mark.parametrize("publication_rows", [1, 3, 13, 128])
def test_reconstruction_ramp_and_final_flush(n, publication_rows):
    frames = torch.arange(n * 10).reshape(n, 10) % 1024
    rows = delayed(frames)
    state = Zonos3RVQStageInputProcessor(CONFIG)
    chunks = []
    for start in range(0, len(rows), publication_rows):
        p = state.push("r", rows[start : start + publication_rows])
        if p is not None:
            chunks.append(p)
        assert len(state.rows) <= 9 + max(CONFIG.chunk_ramp + (CONFIG.chunk_size,)) + publication_rows
    chunks.append(state.push("r", None, finished=True))
    assert torch.equal(torch.cat([p.codes.audio for p in chunks]), frames)
    assert [p.meta.chunk_seq for p in chunks] == list(range(len(chunks)))
    assert [bool(p.meta.finished) for p in chunks] == [False] * (len(chunks) - 1) + [True]
    assert state.emitted == n and state.done and not state.rows
    assert all(p.meta.request_id == "r" for p in chunks)


def test_exact_ramp_thresholds_with_one_row_per_publication():
    frames = torch.arange(700).reshape(70, 10) % 1024
    state = Zonos3RVQStageInputProcessor(CONFIG)
    sizes = []
    for row in delayed(frames):
        p = state.push("r", row[None])
        if p is not None:
            sizes.append(len(p.codes.audio))
    final = state.push("r", None, finished=True)
    assert sizes == [1, 2, 4, 8, 15, 15, 15]
    assert len(final.codes.audio) == 10


@pytest.mark.parametrize("first_codebooks", [1, 10])
@pytest.mark.parametrize("silence_frames", [1, 3, 12])
def test_silence_priming_counts_only_generated_frames_toward_first_chunk(first_codebooks, silence_frames):
    # The prompt supplies only the first S delayed rows. Subsequent AR rows
    # complete their residuals; dropping this prefix loses decoder context.
    frames = torch.arange((silence_frames + 20) * 10).reshape(-1, 10) % 1024
    rows = delayed(frames)
    state = Zonos3RVQStageInputProcessor(replace(CONFIG, first_chunk_codebooks=first_codebooks))
    state.prime(rows[:silence_frames])
    chunks = []
    first_publication = None
    for index, row in enumerate(rows[silence_frames:]):
        payload = state.push("primed", row[None])
        if payload is not None:
            if first_publication is None:
                first_publication = index + 1
            chunks.append(payload)
    chunks.append(state.push("primed", None, finished=True))
    assert first_publication == first_codebooks
    assert chunks[0].meta.ref_context_size == silence_frames
    assert all(chunk.meta.ref_context_size is None for chunk in chunks[1:])
    assert len(chunks[0].codes.audio) == silence_frames + 1
    combined = torch.cat([chunk.codes.audio for chunk in chunks])
    assert torch.equal(combined[:, 0], frames[:, 0])
    assert torch.equal(combined[silence_frames + 1 :], frames[silence_frames + 1 :])
    if first_codebooks == 10:
        assert torch.equal(combined, frames)
    else:
        # Partial frames intentionally omit unavailable residuals once, and
        # those frames must never be decoded again with more codebooks.
        assert (combined[silence_frames, 1:] == -1).all()


@pytest.mark.parametrize("codebooks", [1, 2, 4, 9])
def test_partial_first_chunk_never_replays_frames(codebooks):
    frames = torch.arange(200).reshape(20, 10)
    rows = delayed(frames)
    state = Zonos3RVQStageInputProcessor(replace(CONFIG, first_chunk_codebooks=codebooks))
    first = state.push("r", rows[:codebooks])
    assert torch.equal(first.codes.audio[0, :codebooks], frames[0, :codebooks])
    assert (first.codes.audio[0, codebooks:] == -1).all()
    state.push("r", rows[codebooks:9])
    tail = state.push("r", rows[max(codebooks, 9) :], finished=True)
    assert tail.meta.num_processed_tokens == 1
    assert torch.equal(tail.codes.audio, frames[1:])


def test_max_token_truncation_flushes_only_complete_frames():
    frames = torch.arange(100).reshape(10, 10)
    state = Zonos3RVQStageInputProcessor(CONFIG)
    p = state.push("r", delayed(frames)[:13], finished=True)
    assert torch.equal(p.codes.audio, frames[:4])


def test_partial_first_chunk_followed_by_immediate_truncation():
    state = Zonos3RVQStageInputProcessor(replace(CONFIG, first_chunk_codebooks=1))
    rows = delayed(torch.arange(30).reshape(3, 10))
    state.push("r", rows[:1])
    p = state.push("r", None, finished=True)
    assert p.codes.audio.shape == (0, 10)
    assert p.meta.finished and state.emitted == 1


def test_terminal_eoa_requires_completed_ar_tail_drain():
    frames = torch.arange(30).reshape(3, 10)
    rows = delayed(frames)
    state = Zonos3RVQStageInputProcessor(CONFIG)
    # EOA is first seen at row three (book zero), but terminal arrives before
    # the ten delayed tail markers that the GPU sampler promises to emit.
    with pytest.raises(ValueError, match="Tail Drain"):
        state.push("r", rows[:4], finished=True, drain_complete=True)


def test_non_terminal_exact_boundary_is_followed_by_transport_terminal():
    frames = torch.arange(20).reshape(2, 10)
    state = Zonos3RVQStageInputProcessor(CONFIG)
    rows = delayed(frames)
    result = state.push("r", rows, finished=True, drain_complete=True)
    assert result is not None and result.meta.codec_streaming and result.meta.finished


def test_cancel_does_not_flush_and_closes_state():
    state = Zonos3RVQStageInputProcessor(CONFIG)
    state.push("r", delayed(torch.zeros((4, 10), dtype=torch.int64))[:8])
    assert state.push("r", None, cancelled=True) is None
    assert not state.rows
    with pytest.raises(RuntimeError, match="closed"):
        state.push("r", None, finished=True)


@pytest.mark.parametrize("bad", [torch.zeros((1, 9), dtype=torch.int64), torch.zeros((1, 10))])
def test_reject_invalid_materialized_rows(bad):
    with pytest.raises(ValueError, match="integer delayed RVQ"):
        Zonos3RVQStageInputProcessor(CONFIG).push("r", bad)


def test_reject_missing_or_invalid_complete_residual():
    rows = delayed(torch.zeros((2, 10), dtype=torch.int64))
    rows[9, 9] = CONFIG.pad_id
    with pytest.raises(ValueError, match="Invalid RVQ code"):
        Zonos3RVQStageInputProcessor(CONFIG).push("r", rows[:10])


def test_reject_non_cpu_payload_without_materializing_it():
    with pytest.raises(ValueError, match="CPU-materialized"):
        Zonos3RVQStageInputProcessor(CONFIG).push("r", torch.empty((1, 10), device="meta", dtype=torch.int64))


@pytest.mark.parametrize("concurrency", [1, 4, 8, 16, 32])
def test_mixed_request_progress(concurrency):
    # CPU producer-state validation, not neural decoder batching evidence.
    frames = [torch.full((i + 3, 10), i, dtype=torch.int64) for i in range(concurrency)]
    rows = [delayed(f) for f in frames]
    states = [Zonos3RVQStageInputProcessor(CONFIG) for _ in frames]
    result = [[] for _ in frames]
    for step in range(max(map(len, rows))):
        # Reverse batch order on alternating steps, removing completed requests.
        order = range(concurrency) if step % 2 else reversed(range(concurrency))
        for i in order:
            if step >= len(rows[i]):
                continue
            payload = states[i].push(str(i), rows[i][step : step + 1], finished=step == len(rows[i]) - 1)
            if payload is not None:
                result[i].append(payload.codes.audio)
    for i, parts in enumerate(result):
        assert torch.equal(torch.cat(parts), frames[i])


@dataclass
class Connector:
    config: dict = field(
        default_factory=lambda: {
            "extra": {
                "codebook_sizes": CONFIG.codebook_sizes,
                "eoa_id": CONFIG.eoa_id,
                "audio_pad_id": CONFIG.pad_id,
            }
        }
    )


@dataclass
class Manager:
    connector: Connector = field(default_factory=Connector)
    request_payload: dict[str, dict[str, object]] = field(default_factory=dict)


@dataclass
class Request:
    external_req_id: str
    aborted: bool = False
    finished: bool = False
    output_stopped: bool = False
    terminal_output: bool = False

    def is_finished(self) -> bool:
        return self.finished


@pytest.mark.parametrize("flat_config", [False, True])
def test_native_hook_independent_requests_terminal_cancel_and_identity_reuse(flat_config):
    manager = Manager()
    if flat_config:
        # ConnectorFactory passes spec.extra directly to the constructor.
        manager.connector.config = manager.connector.config["extra"]
    a, b = Request("external-a"), Request("external-b")
    frames = torch.arange(200).reshape(20, 10)
    rows = delayed(frames)
    for request in (a, b):
        assert zonos3_rvq_async_chunk(manager, {"codes": {"audio": rows[:9]}}, request) is None
    a.aborted = True
    aborted = zonos3_rvq_async_chunk(manager, None, a, True)
    assert aborted.meta.aborted is True and aborted.meta.transport_terminal
    assert aborted.codes.audio.shape == (0, 10)
    assert a.external_req_id not in manager.request_payload
    p = zonos3_rvq_async_chunk(
        manager,
        {"codes": {"audio": rows[9:]}, "meta": {"codec_eos_valid": torch.tensor([True])}},
        b,
        True,
    )
    assert torch.equal(p.codes.audio, frames)
    assert p.meta.finished and p.meta.request_id == b.external_req_id
    assert b.external_req_id not in manager.request_payload
