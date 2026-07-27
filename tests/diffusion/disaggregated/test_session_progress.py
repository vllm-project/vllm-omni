# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch-free tests for the disaggregated session-progress control plane (RFC #4590 Part A).

These validate the ordering / idempotency / epoch logic that keeps the AR window
cursor coherent across the encode -> denoise -> decode process boundaries, WITHOUT
the model runtime: the coordinator (``vllm_omni.diffusion.session_progress``) is
torch-free and loaded by file path, mirroring how ``conftest.py`` loads the other
torch-free foundation modules. They run on any CPU host (no GPU / no checkpoint).

Each test maps to a required Part D case:

* five-chunk same-session progression (case 1)
* interleaved sessions (case 2)
* duplicate retry (case 3)
* stale sequence (case 4)
* failure before commit + retry-once (case 5)
* explicit reset / epoch fence (case 6)
* window boundary: csf resets, sequence stays monotonic, no epoch bump (case 7)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3] / "vllm_omni"


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Loaded by path so the test needs neither torch nor the vllm_omni package init.
_sp = _load("_rfc4590_session_progress", "diffusion/session_progress.py")

SessionProgressCoordinator = _sp.SessionProgressCoordinator
ProgressDecision = _sp.ProgressDecision
SessionProgressError = _sp.SessionProgressError

NFPB = 3  # a representative num_frame_per_block


def _run_chunk(enc, den, session_id, csf_before, *, first):
    """Issue on encode, authorize+commit on denoise for one healthy chunk.

    Returns the committed current_start_frame after this chunk, mirroring the
    pipeline's csf arithmetic (0->1 on the first chunk of a window, then += NFPB).
    """
    epoch, seq = enc.issue(session_id)
    decision = den.authorize(session_id, epoch=epoch, sequence_no=seq)
    assert decision is ProgressDecision.PROCEED, (session_id, seq, decision)
    csf_after = (1 if (first and csf_before == 0) else csf_before) + NFPB
    den.commit(session_id, epoch=epoch, sequence_no=seq, current_start_frame=csf_after)
    return epoch, seq, csf_after


# --- case 1: five-chunk same-session progression ---------------------------


def test_five_chunk_same_session_progression():
    enc, den = SessionProgressCoordinator(), SessionProgressCoordinator()
    csf = 0
    seqs, epochs = [], []
    for i in range(5):
        epoch, seq, csf = _run_chunk(enc, den, "A", csf, first=(i == 0))
        seqs.append(seq)
        epochs.append(epoch)
    assert seqs == [0, 1, 2, 3, 4]
    assert epochs == [0, 0, 0, 0, 0]  # epoch constant across the session
    rec = den.peek("A")
    assert rec.last_committed_sequence == 4
    # csf derived from NFPB, not hard-coded. Mirrors the pipeline: the first chunk
    # steps 0->1 (prefill) THEN += NFPB (denoise) => ends at 1 + NFPB; each later
    # chunk adds another NFPB. After 5 chunks: (1 + NFPB) + NFPB*4 = 1 + NFPB*5.
    assert rec.current_start_frame == 1 + NFPB * 5 == 16
    assert rec.epoch == 0


# --- case 2: interleaved sessions progress independently -------------------


def test_interleaved_sessions_independent():
    enc, den = SessionProgressCoordinator(), SessionProgressCoordinator()
    csf = {"A": 0, "B": 0}
    order = ["A", "B", "A", "B", "A"]
    seqs = {"A": [], "B": []}
    for idx, sid in enumerate(order):
        first = csf[sid] == 0
        _epoch, seq, csf[sid] = _run_chunk(enc, den, sid, csf[sid], first=first)
        seqs[sid].append(seq)
    assert seqs["A"] == [0, 1, 2]
    assert seqs["B"] == [0, 1]
    # No shared state: A's 3 commits and B's 2 commits are tracked separately.
    assert den.peek("A").last_committed_sequence == 2
    assert den.peek("B").last_committed_sequence == 1
    assert den.peek("A").current_start_frame != den.peek("B").current_start_frame


# --- case 3: duplicate retry -----------------------------------------------


def test_duplicate_retry_is_rejected_not_double_committed():
    enc, den = SessionProgressCoordinator(), SessionProgressCoordinator()
    csf = 0
    _e, _s, csf = _run_chunk(enc, den, "A", csf, first=True)  # commit seq 0
    committed_csf = den.peek("A").current_start_frame
    # Re-submit the just-committed sequence 0 with a different attempt id.
    decision = den.authorize("A", epoch=0, sequence_no=0, attempt_id="dup")
    assert decision is ProgressDecision.DUPLICATE
    assert not decision.ok
    # Committed progress is unchanged (no double advance / no second KV append).
    assert den.peek("A").last_committed_sequence == 0
    assert den.peek("A").current_start_frame == committed_csf


# --- case 4: stale sequence ------------------------------------------------


def test_stale_sequence_rejected_before_mutation():
    enc, den = SessionProgressCoordinator(), SessionProgressCoordinator()
    csf = 0
    for i in range(3):  # commit seq 0,1,2
        _e, _s, csf = _run_chunk(enc, den, "A", csf, first=(i == 0))
    assert den.peek("A").last_committed_sequence == 2
    # Submitting sequence 1 after 2 is committed: older than last_committed -> STALE.
    assert den.authorize("A", epoch=0, sequence_no=1) is ProgressDecision.STALE
    # The just-committed sequence 2 -> DUPLICATE (both reject, distinct reasons).
    assert den.authorize("A", epoch=0, sequence_no=2) is ProgressDecision.DUPLICATE
    # Progress untouched by the rejected authorizations.
    assert den.peek("A").last_committed_sequence == 2


# --- case 5: failure before commit + retry-once ----------------------------


def test_failure_before_commit_allows_single_retry():
    enc, den = SessionProgressCoordinator(), SessionProgressCoordinator()
    csf = 0
    _e, _s, csf = _run_chunk(enc, den, "A", csf, first=True)  # seq 0 committed
    # Chunk seq 1 arrives; denoise authorizes then FAILS before commit.
    epoch, seq = enc.issue("A")
    assert seq == 1
    assert den.authorize("A", epoch=epoch, sequence_no=seq) is ProgressDecision.PROCEED
    # (no commit — simulate a denoise exception)
    before = den.peek("A").last_committed_sequence
    assert before == 0  # committed progress did NOT advance on failure
    # Retry the SAME sequence (new attempt id) -> still the expected next -> PROCEED.
    assert den.authorize("A", epoch=epoch, sequence_no=seq, attempt_id="retry") is ProgressDecision.PROCEED
    den.commit("A", epoch=epoch, sequence_no=seq, current_start_frame=csf + NFPB)
    assert den.peek("A").last_committed_sequence == 1
    # A second commit of the same sequence is now a DUPLICATE (exactly-once).
    assert den.authorize("A", epoch=epoch, sequence_no=seq) is ProgressDecision.DUPLICATE


def test_commit_without_authorize_gate_raises():
    """commit() must be gated by authorize(): committing out of order is a bug."""
    den = SessionProgressCoordinator()
    with pytest.raises(SessionProgressError):
        den.commit("A", epoch=0, sequence_no=5, current_start_frame=10)


# --- case 6: explicit reset / epoch fence ----------------------------------


def test_explicit_reset_bumps_epoch_and_fences_old_attempts():
    enc, den = SessionProgressCoordinator(), SessionProgressCoordinator()
    csf = 0
    for i in range(2):  # a couple of chunks in epoch 0
        _e, _s, csf = _run_chunk(enc, den, "A", csf, first=(i == 0))
    assert den.peek("A").epoch == 0

    # Explicit reset on the encode (issuer) side -> new epoch, sequence restarts.
    new_epoch = enc.begin_epoch_reset("A")
    assert new_epoch == 1
    epoch, seq = enc.issue("A")
    assert (epoch, seq) == (1, 0)

    # The reset-carrying request reaches denoise in order; the authority adopts
    # the new epoch and the post-reset sequence 0 authorizes as first-chunk.
    assert den.authorize("A", epoch=1, sequence_no=0) is ProgressDecision.PROCEED
    den.commit("A", epoch=1, sequence_no=0, current_start_frame=1 + NFPB)
    assert den.peek("A").epoch == 1
    assert den.peek("A").last_committed_sequence == 0

    # A lingering pre-reset (epoch 0) in-flight attempt is now fenced out.
    assert den.authorize("A", epoch=0, sequence_no=2) is ProgressDecision.EPOCH_STALE
    # First post-reset chunk behaves like a first chunk (csf started from 0->1).
    assert den.peek("A").current_start_frame == 1 + NFPB


def test_adopt_epoch_cannot_go_backwards():
    den = SessionProgressCoordinator()
    den.adopt_epoch("A", 3)
    assert den.peek("A").epoch == 3
    den.adopt_epoch("A", 3)  # idempotent
    assert den.peek("A").epoch == 3
    with pytest.raises(SessionProgressError):
        den.adopt_epoch("A", 1)


# --- case 7: window boundary (csf reset, monotonic sequence, no epoch bump) --


def test_window_boundary_resets_csf_without_epoch_bump():
    enc, den = SessionProgressCoordinator(), SessionProgressCoordinator()
    csf = 0
    for i in range(4):
        epoch, seq = enc.issue("A")
        assert den.authorize("A", epoch=epoch, sequence_no=seq) is ProgressDecision.PROCEED
        # At chunk index 2 the AR window fills and slides back to 0 (an
        # "inference" reset), WITHOUT an epoch bump — this is normal progression.
        csf = 0 if i == 2 else ((1 if csf == 0 else csf) + NFPB)
        den.commit("A", epoch=epoch, sequence_no=seq, current_start_frame=csf)
    rec = den.peek("A")
    assert rec.epoch == 0  # window slide did NOT bump the epoch
    assert rec.last_committed_sequence == 3  # sequence stayed monotonic
    # A previously-committed sequence is still rejected (STALE for seq<last, or
    # DUPLICATE for seq==last), not confused by the csf reset: the ordering key is
    # the monotonic sequence, independent of the window cursor.
    assert den.authorize("A", epoch=0, sequence_no=2) is ProgressDecision.STALE
    assert den.authorize("A", epoch=0, sequence_no=3) is ProgressDecision.DUPLICATE


# --- LRU bound + misc guards ------------------------------------------------


def test_session_lru_bound():
    coord = SessionProgressCoordinator(max_sessions=2)
    coord.issue("A")
    coord.issue("B")
    coord.issue("C")  # evicts LRU "A"
    assert coord.peek("A") is None
    assert coord.peek("B") is not None
    assert coord.peek("C") is not None


def test_max_sessions_must_be_positive():
    with pytest.raises(ValueError):
        SessionProgressCoordinator(max_sessions=0)


def test_drop_forgets_session():
    coord = SessionProgressCoordinator()
    coord.issue("A")
    assert coord.peek("A") is not None
    coord.drop("A")
    assert coord.peek("A") is None
