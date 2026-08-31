# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU e2e for PersonaPlex elastic batching (bit-exact isolation + recycle).

Greedy decoding makes every cross-slot leak observable as a bit difference:

1. Isolation (bit-exact): recycling one slot mid-flight leaves every other
   slot's output bit-identical to a run that never recycled.
2. Replay exactness: a recycled slot's opening live ticks match a pristine boot
   at numerical-noise level (a broken KV mask / staging roll garbles frame 0).
3. Coherence (behavioral): a recycled slot fed a real spoken question produces
   a real reply in its inner-monologue text stream.

Run (moshi-free native stack; ~30 GB GPU + HF access to the gated repo; set
PPLEX_QUESTION_WAV to a spoken-question wav to enable the coherence test):
    pytest tests/e2e/features/fullduplex/test_personaplex_elastic.py -v -s

Bench (B sweep, ms/tick vs the 80 ms budget):
    python tests/e2e/features/fullduplex/test_personaplex_elastic.py --bench 2 4 8 16
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.helpers.mark import hardware_marks

torch = pytest.importorskip("torch")

# slow + H100 + omni + cards_1 routes this module into weekly
# "Omni · H100 · Single-GPU" (test-weekly.yml).
pytestmark = [
    pytest.mark.omni,
    pytest.mark.slow,
    *hardware_marks(res={"cuda": "H100"}),
]

if not torch.cuda.is_available():  # noqa: E402
    pytest.skip("needs CUDA", allow_module_level=True)

from vllm_omni.experimental.fullduplex.personaplex.config import PersonaPlexConfig  # noqa: E402
from vllm_omni.experimental.fullduplex.personaplex.runtime import (  # noqa: E402
    PersonaPlexEngine,
)

B = 4
LIVE_TICKS = 40  # live frames compared per claim
RECYCLE_AT = 10


def _make_engine() -> PersonaPlexEngine:
    cfg = PersonaPlexConfig(batch_size=B)  # native stepper is greedy-only
    try:
        return PersonaPlexEngine(cfg).load()
    except Exception as exc:  # gated-repo access is environment-dependent
        name = type(exc).__name__
        if "Gated" in name or "401" in str(exc) or "403" in str(exc):
            pytest.skip(f"no access to the gated nvidia/personaplex-7b-v1 repo: {name}")
        raise


@pytest.fixture(scope="module")
def engine() -> PersonaPlexEngine:
    return _make_engine()


def _slot_stream(slot: int, tick: int, frame: int) -> np.ndarray:
    """Deterministic per-slot input: distinct low-amplitude tones."""
    sr = 24000
    freq = 180.0 + 90.0 * slot
    t0 = tick * frame
    t = (np.arange(frame) + t0) / sr
    return (0.12 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _run(engine, total_ticks: int, recycle: bool):
    """Run the lockstep loop; optionally recycle slot 1 at RECYCLE_AT.

    Returns per-slot lists of (audio | None) per live tick index, where the
    recycled slot's list restarts its input stream at its post-prefill tick.
    """
    frame = engine.frame_size
    outs: list[list[np.ndarray | None]] = [[] for _ in range(B)]
    stream_tick = [0] * B  # per-slot conversation clock driving its input
    prefill = None
    tick = 0
    while tick < total_ticks:
        if recycle and tick == RECYCLE_AT and prefill is None:
            engine.recycle_slot(1)
            prefill = list(engine.prefill_steps(None))
        pcm = np.zeros((B, frame), dtype=np.float32)
        step_for_1 = None
        if prefill:
            while prefill and prefill[0].kind == "voice_end":
                engine.voice_end(1)
                prefill.pop(0)
            if prefill:
                step_for_1 = prefill.pop(0)
            else:
                engine.end_prefill(1)
                prefill = None
                stream_tick[1] = 0  # restart slot 1's input stream from its own t=0
        for b in range(B):
            if b == 1 and step_for_1 is not None:
                continue  # engine overrides the user channel for prefill rows
            pcm[b] = _slot_stream(b, stream_tick[b], frame)
            stream_tick[b] += 1
        frame_outs = engine.step_batch(pcm, {1: step_for_1} if step_for_1 is not None else None)
        for b in range(B):
            if b == 1 and step_for_1 is not None:
                continue  # prefill tick: nothing conversational to record
            frame_out = frame_outs[b]
            outs[b].append(None if frame_out is None else frame_out.audio)
        tick += 1
    return outs


def test_recycle_isolation_and_fresh_equivalence(engine):
    engine.open_batch()
    prefill_len = sum(1 for s in engine.prefill_steps(None) if s.kind != "voice_end")
    total = RECYCLE_AT + prefill_len + LIVE_TICKS

    base = _run(engine, total, recycle=False)

    engine.open_batch()  # deterministic re-boot (greedy, fixed voice/persona)
    rec = _run(engine, total, recycle=True)

    # Claim 1 -- isolation: untouched slots bit-identical to the no-recycle run.
    for b in (0, 2, 3):
        assert len(base[b]) == total and len(rec[b]) == total
        for i, (x, y) in enumerate(zip(base[b], rec[b])):
            assert (x is None) == (y is None), f"slot {b} tick {i} validity diverged"
            if x is not None:
                assert np.array_equal(x, y), f"slot {b} tick {i} audio diverged (recycle leaked)"

    # Claim 2 -- replay exactness: the recycled slot's opening live ticks match a
    # pristine boot at numerical-noise level. A broken KV mask / staging-ring
    # roll / Mimi reset garbles the audio from the very first frame (diffs of
    # 0.1-1.0); a correct replay differs only by decoder-state rounding (~1e-4).
    #
    # Full-length bitwise identity is NOT expected, for two structural reasons:
    # the replayed system prompt sits at a different ABSOLUTE RoPE offset (bf16
    # rounds differently there), and the recycled Mimi ENCODER cannot restore a
    # fresh stream's start padding (a batch-shared scalar), so from the first
    # non-silent user frame it emits different-but-equally-valid codes for the
    # same audio. Once any near-tie flips a token, the two equally valid
    # trajectories legitimately diverge; coherence of the recycled slot is
    # asserted behaviorally in test_recycled_slot_converses.
    fresh = [a for a in base[1] if a is not None][:LIVE_TICKS]
    reborn = [a for a in rec[1][RECYCLE_AT:] if a is not None][:LIVE_TICKS]
    assert len(reborn) >= LIVE_TICKS // 2, "recycled slot produced too few live frames"
    diffs = [float(np.max(np.abs(x - y))) for x, y in zip(fresh, reborn)]
    first_flip = next((i for i, d in enumerate(diffs) if d > 5e-2), len(diffs))
    print(
        f"[elastic] replay exactness: {first_flip}/{len(diffs)} opening ticks at noise level "
        f"(head diffs: {[f'{d:.1e}' for d in diffs[:6]]})"
    )
    assert first_flip >= 4, (
        f"recycled slot diverged from fresh behavior at tick {first_flip} "
        f"(diffs {diffs[: first_flip + 2]}); the system-prompt replay is broken"
    )
    assert max(diffs[:4]) < 2e-2, f"reborn audio not at noise level: {max(diffs[:4])}"


def test_recycled_slot_converses(engine):
    """Behavioral guarantee: a recycled slot holds a coherent conversation.

    Feed a real spoken question into a slot that was recycled mid-flight (while
    other slots keep running); the model's own inner-monologue text stream must
    produce a real reply. This is the end-user property the encoder-state
    caveats above cannot break.
    """
    import os

    import sphn

    wav_path = os.environ.get("PPLEX_QUESTION_WAV")
    if not wav_path:
        pytest.skip("set PPLEX_QUESTION_WAV to a spoken-question wav to run")
    wav, _ = sphn.read(wav_path, sample_rate=24000)
    wav = wav[0].astype(np.float32)
    frame = engine.frame_size
    pad = (-len(wav)) % frame
    wav = np.concatenate([wav, np.zeros(pad, dtype=np.float32)])
    q_frames = [wav[i : i + frame] for i in range(0, len(wav), frame)]

    engine.open_batch()
    pcm = np.zeros((B, frame), dtype=np.float32)
    for _ in range(5):  # everyone live briefly
        engine.step_batch(pcm)
    engine.recycle_slot(1)
    prefill = list(engine.prefill_steps(None))
    while prefill:
        while prefill and prefill[0].kind == "voice_end":
            engine.voice_end(1)
            prefill.pop(0)
        if not prefill:
            break
        engine.step_batch(pcm, {1: prefill.pop(0)})
    engine.end_prefill(1)

    text = ""
    total = len(q_frames) + 120  # question + ~10 s to answer
    for i in range(total):
        pcm = np.zeros((B, frame), dtype=np.float32)
        if i < len(q_frames):
            pcm[1] = q_frames[i]
        outs = engine.step_batch(pcm)
        if outs[1] is not None and outs[1].text:
            text += outs[1].text
    alpha = sum(c.isalpha() for c in text)
    print(f"[elastic] recycled slot inner monologue: {text!r}")
    assert alpha >= 15, f"recycled slot produced no coherent reply: {text!r}"


def _bench(batch_sizes: list[int]) -> None:
    import time

    for bs in batch_sizes:
        cfg = PersonaPlexConfig(batch_size=max(bs, 2))
        eng = PersonaPlexEngine(cfg).load()
        eng.open_batch()
        frame = eng.frame_size
        pcm = np.zeros((cfg.batch_size, frame), dtype=np.float32)
        for _ in range(10):  # warm
            eng.step_batch(pcm)
        torch.cuda.synchronize()
        t0 = time.monotonic()
        n = 100
        for _ in range(n):
            eng.step_batch(pcm)
        torch.cuda.synchronize()
        ms = (time.monotonic() - t0) / n * 1000
        mem = torch.cuda.max_memory_allocated() / 2**30
        print(f"B={cfg.batch_size:3d}  {ms:6.1f} ms/tick  RTF/stream {ms / 80:.3f}  {mem:.1f} GiB")
        del eng
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import sys

    sizes = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 and sys.argv[1] == "--bench" else [2, 4, 8]
    _bench(sizes)
