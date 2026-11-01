# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contract and boundary cases for the Noisy KV session, from the outside in.

Every expectation here comes from the written contract
(``docs/design/feature/noisy_kv.md``) or from a plain invariant about the KV
view, never from the code under test: a continuation may only read blocks that
were written, a session may not claim versions it released, the retention
window is the last H committed chunks, and the offset is the number of chunks
that actually finished.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.experimental.ar_diffusion.cell_executor import (
    ARDiffusionCellContext,
    CellAdapter,
    CellRunSpec,
    CellTopology,
    run_cell_pipeline,
)
from vllm_omni.experimental.ar_diffusion.cell_schedule import CellSchedule, Ordering, build_cell_plan
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import (
    ARDiffusionNoisyKVSpec,
    NoisyKVCache,
    NoisyKVState,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.session import (
    NoisyKVSession,
    NoisyKVSessionRegistry,
    NoisySessionError,
    NoisySessionState,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

MARKER = 7.0


class _ViewRecorder(CellAdapter):
    """Writes a marker into every chunk it runs and records what it read."""

    def __init__(self, *, fail_chunk: int | None = None) -> None:
        self.fail_chunk = fail_chunk
        self.views: list[dict] = []

    def forward(self, tasks, kv_contexts, *, hidden):
        del hidden
        for (_req, cell), ctx_list in zip(tasks, kv_contexts):
            view = ctx_list[0] if isinstance(ctx_list, list) else ctx_list
            chunk, _step, _block = cell
            if self.fail_chunk is not None and chunk >= self.fail_chunk:
                raise RuntimeError("cell forward failed")
            block_size = int(view.block_size)
            history_blocks = int(view.seq_len) // block_size - len(view.video_slots) // block_size
            history_blocks = max(0, history_blocks)
            table = view.block_table[0, : int(view.seq_len) // block_size].long()
            # ``block_table`` holds block ids; a block covers ``block_size`` tokens,
            # so the history content is the concatenation of those token ranges.
            tokens = [
                token
                for block in table[:history_blocks].tolist()
                for token in range(block * block_size, (block + 1) * block_size)
            ]
            history_values = (
                view.key_pool[torch.tensor(tokens, dtype=torch.long, device=view.key_pool.device)].reshape(-1).tolist()
                if tokens
                else []
            )
            self.views.append(
                {
                    "cell": cell,
                    "seq_len": int(view.seq_len),
                    "block_size": block_size,
                    "table": table.tolist(),
                    "history_blocks": history_blocks,
                    "history_values": history_values,
                }
            )
            slot = view.video_slots.long()
            view.key_pool[slot] = torch.full_like(view.key_pool[slot], MARKER)
            view.value_pool[slot] = torch.full_like(view.value_pool[slot], MARKER)
        return torch.zeros(1, 1, 1)

    def pack_activation(self, output):
        return {"hidden_states": output}

    def unpack_activation(self, payload):
        return payload["hidden_states"]


def _cache(*, max_chunk_tokens: int = 4, block_size: int = 4, history: int = 1) -> NoisyKVCache:
    return NoisyKVCache(
        ARDiffusionNoisyKVSpec(
            num_layers=1,
            num_kv_heads=1,
            head_size=1,
            block_size=block_size,
            max_chunk_tokens=max_chunk_tokens,
            max_history_chunks=history,
        ),
        dtype=torch.float32,
        device=torch.device("cpu"),
        layer_groups=1,
        max_batch_size=1,
        session_retention_chunks=max(1, history),
    )


def _plan(*, chunks: int, history: int = 1, steps: int = 1):
    return build_cell_plan(
        CellSchedule(
            chunks=chunks,
            num_denoise_steps=steps,
            step_replicas=1,
            layer_groups=1,
            ordering=Ordering.SELF_FORCING,
            kv_history_chunks=history,
        )
    )


def _run(
    state: NoisyKVState,
    session: NoisyKVSession | None,
    req: str,
    *,
    chunks: int,
    chunk_tokens: int,
    history: int = 1,
    chunk_start: int | None = None,
    fail_chunk: int | None = None,
) -> _ViewRecorder:
    plan = _plan(chunks=chunks, history=history)
    ctx = ARDiffusionCellContext(
        spec=CellRunSpec(topology=CellTopology(step_replicas=1, layer_groups=1), rank=0),
        kv=state,
        chunk_start=0 if session is None else (session.committed_chunks if chunk_start is None else chunk_start),
        session=session,
    )
    assert ctx.admit(req, plan, chunk_tokens=chunk_tokens, slot=0)
    recorder = _ViewRecorder(fail_chunk=fail_chunk)
    run_cell_pipeline(ctx=ctx, adapter=recorder)
    return recorder


def test_history_blocks_hold_only_what_the_producing_chunk_wrote():
    """A continuation may only read blocks the earlier chunk actually wrote.

    Request A runs 4-token chunks (one block each). Request B continues the same
    session with 8-token chunks (two blocks each). The roadmap requires the plan
    to match the request configuration; if the session reuses history produced
    under a different shape, the reader must not consume an unwritten block.
    """
    cache = _cache(max_chunk_tokens=8)
    state = NoisyKVState(cache)
    session = NoisyKVSession(session_id="s1")

    _run(state, session, "A", chunks=2, chunk_tokens=4)
    assert session.committed_chunks == 2

    # The roadmap requires the plan to match the request configuration, so a
    # shape change inside one session is refused instead of reused.
    with pytest.raises(NoisySessionError):
        _run(state, session, "B", chunks=1, chunk_tokens=8)
    assert session.committed_chunks == 2, "the refused request still advanced the session"


def test_retention_window_keeps_exactly_the_last_h_committed_chunks():
    """The window is H chunks, so chunk 0 must be gone after H+1 chunks."""
    history = 2
    cache = _cache(history=history)
    state = NoisyKVState(cache)
    session = NoisyKVSession(session_id="s1")

    _run(state, session, "A", chunks=3, chunk_tokens=4, history=history)

    retained_chunks = sorted({chunk for chunk, _step, _block in session.retained})
    assert retained_chunks == [1, 2], f"window kept {retained_chunks}, expected the last {history} chunks"


def test_offset_is_the_number_of_chunks_that_finished_across_requests():
    cache = _cache(history=2)
    state = NoisyKVState(cache)
    session = NoisyKVSession(session_id="s1")

    starts = []
    for chunks in (2, 2, 1):
        starts.append(session.committed_chunks)
        _run(state, session, f"r{chunks}-{len(starts)}", chunks=chunks, chunk_tokens=4, history=2)

    assert starts == [0, 2, 4]
    assert session.committed_chunks == 5


def test_a_failed_chunk_neither_advances_the_offset_nor_keeps_its_versions():
    cache = _cache(history=2)
    state = NoisyKVState(cache)
    session = NoisyKVSession(session_id="s1")

    with pytest.raises(RuntimeError, match="cell forward failed"):
        _run(state, session, "A", chunks=2, chunk_tokens=4, history=2, fail_chunk=1)

    assert session.committed_chunks == 1, "the failed chunk advanced the continuation offset"
    retained_chunks = sorted({chunk for chunk, _step, _block in session.retained})
    assert retained_chunks == [0], f"a failed chunk's versions were retained: {retained_chunks}"


def test_releasing_a_session_does_not_leave_it_claiming_released_versions():
    """After the slots are handed back, the session must not still claim them.

    Otherwise the next request reads a dictionary that points at versions the
    pool no longer holds, and the failure surfaces as a KV-layer error instead
    of a clean session state.
    """
    cache = _cache(history=1)
    state = NoisyKVState(cache)
    session = NoisyKVSession(session_id="s1")
    _run(state, session, "A", chunks=2, chunk_tokens=4)
    assert session.retained

    cache.release_session(session)

    assert session.retained == {}, "the session still claims versions that were released"


def test_a_continuation_after_release_restarts_from_a_clean_position():
    """Releasing the history also drops the offset it was committed at.

    Keeping the offset would let the next request continue from a position whose
    history no longer exists, which is exactly the silent cold start the design
    forbids.
    """
    cache = _cache(history=1)
    state = NoisyKVState(cache)
    session = NoisyKVSession(session_id="s1")
    _run(state, session, "A", chunks=2, chunk_tokens=4)
    cache.release_session(session)

    assert session.committed_chunks == 0
    recorder = _run(state, session, "B", chunks=1, chunk_tokens=4)

    assert recorder.views[0]["history_blocks"] == 0
    assert session.committed_chunks == 1


def test_lru_eviction_makes_the_evicted_session_start_cold():
    cache = _cache(history=1)
    state = NoisyKVState(cache)
    registry = NoisyKVSessionRegistry(capacity=1, on_release=cache.release_session)

    session, _generation = registry.admit("s1")
    _run(state, session, "A", chunks=2, chunk_tokens=4)
    assert session.committed_chunks == 2

    registry.admit("s2")

    resumed, _generation = registry.admit("s1")
    assert resumed.committed_chunks == 0, "an evicted session resumed from an offset it no longer holds"
    assert resumed.retained == {}


def test_reset_after_close_is_rejected():
    session = NoisyKVSession(session_id="s1")
    session.close()

    assert session.state is NoisySessionState.CLOSED
    with pytest.raises(NoisySessionError):
        session.reset()


def test_close_is_idempotent():
    session = NoisyKVSession(session_id="s1")
    session.close()
    session.close()

    assert session.state is NoisySessionState.CLOSED


def test_a_session_admits_nothing_after_it_was_invalidated():
    session = NoisyKVSession(session_id="s1")
    session.invalidate("boom")

    assert session.state is NoisySessionState.INVALID
    with pytest.raises(NoisySessionError):
        session.admit()


def test_a_stale_generation_cannot_commit_after_reset():
    session = NoisyKVSession(session_id="s1")
    generation = session.admit()
    session.reset()

    with pytest.raises(NoisySessionError):
        session.commit_chunks(1, generation=generation)


@pytest.mark.parametrize("versions", [0, -1])
def test_the_spec_rejects_a_non_positive_history_version_count(versions):
    with pytest.raises(ValueError):
        ARDiffusionNoisyKVSpec(
            num_layers=1,
            num_kv_heads=1,
            head_size=1,
            block_size=4,
            max_chunk_tokens=4,
            max_history_chunks=1,
            max_history_versions=versions,
        )


def test_a_chunk_size_that_is_not_a_block_multiple_is_rejected():
    with pytest.raises(ValueError):
        ARDiffusionNoisyKVSpec(
            num_layers=1,
            num_kv_heads=1,
            head_size=1,
            block_size=4,
            max_chunk_tokens=6,
            max_history_chunks=1,
        )


def test_the_same_configuration_reuses_one_cell_plan():
    """Roadmap 1.1/2.4: the plan is matched to the configuration and reused."""
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan import CausVidWanPipeline

    holder = SimpleNamespace()

    def schedule(chunks: int) -> CellSchedule:
        return CellSchedule(
            chunks=chunks,
            num_denoise_steps=1,
            step_replicas=1,
            layer_groups=1,
            ordering=Ordering.SELF_FORCING,
            kv_history_chunks=1,
        )

    first = CausVidWanPipeline._cell_plan(holder, schedule(2))
    same = CausVidWanPipeline._cell_plan(holder, schedule(2))
    other = CausVidWanPipeline._cell_plan(holder, schedule(3))

    assert first is same, "an identical configuration built a second plan"
    assert other is not first, "a different configuration reused the cached plan"


def test_conditioning_is_kept_with_the_session_and_dropped_on_reset():
    """Roadmap 1.1 item 4: conditioning is part of what a session keeps."""
    import torch

    from vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan import CausVidWanPipeline

    class _Encoder:
        def __init__(self) -> None:
            self.calls = 0

        def _encode_prompt(self, prompt, device, dtype):
            self.calls += 1
            return f"embeds:{prompt}"

    encoder = _Encoder()
    session = NoisyKVSession(session_id="s1")
    device, dtype = torch.device("cpu"), torch.float32

    first = CausVidWanPipeline._conditioning(encoder, "a world", device, dtype, session)
    again = CausVidWanPipeline._conditioning(encoder, "a world", device, dtype, session)
    assert again == first
    assert encoder.calls == 1, "the continuation re-encoded the same prompt"

    CausVidWanPipeline._conditioning(encoder, "another world", device, dtype, session)
    assert encoder.calls == 2

    session.reset()
    CausVidWanPipeline._conditioning(encoder, "a world", device, dtype, session)
    assert encoder.calls == 3, "a reset session reused the conditioning it dropped"
