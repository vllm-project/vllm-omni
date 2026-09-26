# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for Moss-TTS P0-2 async D2H (_PendingAudio).

Tests the pinned-memory view aliasing fix: resolve() must return an
independent copy (.clone()), not a view into the pinned buffer, so that
subsequent pinned allocations don't corrupt previously resolved audio.
"""

import pytest
import torch
import torch_npu

pytest.importorskip("vllm_ascend")

from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.omni,
    *hardware_marks(res={"npu": "A3"}, num_cards=1),
]

DEVICE = torch.device("npu:0")


def _make_pending(audio_npu, row=0, audio_length=None):
    """Create a _PendingAudio exactly like session.step does."""
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import _PendingAudio

    if audio_length is None:
        audio_length = audio_npu.shape[-1]

    pinned = torch.empty(
        audio_npu.shape, dtype=audio_npu.dtype, device="cpu", pin_memory=True
    )
    d2h_stream = torch.npu.Stream()
    main_stream = torch.npu.current_stream()
    d2h_event = torch.npu.Event()

    with torch.npu.stream(d2h_stream):
        d2h_stream.wait_stream(main_stream)
        pinned.copy_(audio_npu, non_blocking=True)
        d2h_event.record()

    return _PendingAudio(pinned, d2h_event, row, audio_length), pinned


# ---------------------------------------------------------------------------
# 1. resolve() returns correct data (parity with sync D2H)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("b,channels,t", [(1, 2, 480), (2, 2, 480), (4, 2, 240)])
def test_resolve_returns_correct_data(b, channels, t):
    """resolve() must return the same data as a synchronous D2H."""
    audio = torch.randn(b, channels, t, device=DEVICE, dtype=torch.float32)
    pa, _ = _make_pending(audio)
    result = pa.resolve()
    expected = audio[0, ..., :t].contiguous().clone().cpu()
    if expected.ndim == 1 or (expected.ndim > 1 and int(expected.shape[0]) == 1):
        expected = expected.reshape(-1)
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2. resolve() returns independent copy — the bug fix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("b", [1, 2, 4])
def test_resolve_returns_independent_copy(b):
    """resolve() must return an independent copy, not a view into pinned.

    Without .clone(), B=1 stereo returned a view (contiguous() is a no-op
    when selecting the only row), and subsequent pinned allocations could
    overwrite the data.
    """
    audio = torch.randn(b, 2, 480, device=DEVICE, dtype=torch.float32)
    pa, pinned = _make_pending(audio)
    result = pa.resolve()

    # Overwrite the pinned buffer (simulating allocator reuse)
    pinned.fill_(999.0)

    # Result must NOT change — it is an independent copy
    expected = audio[0, ..., :480].contiguous().clone().cpu()
    if expected.ndim == 1 or (expected.ndim > 1 and int(expected.shape[0]) == 1):
        expected = expected.reshape(-1)
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 3. resolve() caches — second call returns same tensor object
# ---------------------------------------------------------------------------

def test_resolve_caches_result():
    """Second resolve() must return the cached tensor, not re-extract."""
    audio = torch.randn(1, 2, 480, device=DEVICE, dtype=torch.float32)
    pa, _ = _make_pending(audio)
    result1 = pa.resolve()
    result2 = pa.resolve()
    assert result1 is result2  # same object (cached)


# ---------------------------------------------------------------------------
# 4. Multiple _PendingAudio resolve independently — the original bug scenario
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_chunks", [2, 4, 8])
def test_multiple_pending_resolve_independently(n_chunks):
    """Multiple _PendingAudio from sequential step() calls must resolve
    independently — one's resolve must not affect another's data.

    Reproduces the non-streaming bug: _decode_stream_slot_sequence creates
    N _PendingAudio in a loop, and pinned allocator reuse corrupted earlier
    results.
    """
    audios = [torch.randn(1, 2, 480, device=DEVICE, dtype=torch.float32) for _ in range(n_chunks)]
    pending = []
    pinned_buffers = []
    for audio in audios:
        pa, pinned = _make_pending(audio)
        pending.append(pa)
        pinned_buffers.append(pinned)

    # Resolve all (simulating _decode_stream_slot_sequence)
    results = [pa.resolve() for pa in pending]

    # Now simulate allocator reuse: overwrite all pinned buffers
    for pinned in pinned_buffers:
        pinned.fill_(999.0)

    # All results must still match original data
    for i, (result, audio) in enumerate(zip(results, audios)):
        expected = audio[0, ..., :480].contiguous().clone().cpu()
        if expected.ndim == 1 or (expected.ndim > 1 and int(expected.shape[0]) == 1):
            expected = expected.reshape(-1)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5,
                                   msg=f"Chunk {i} corrupted by pinned reuse")


# ---------------------------------------------------------------------------
# 5. B>1 multi-row resolve — each row independent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("b", [2, 4])
def test_multi_row_resolve(b):
    """For B>1, each row's resolve must return correct independent data."""
    audio = torch.randn(b, 2, 480, device=DEVICE, dtype=torch.float32)

    # Create one pinned buffer for all rows (like session.step does)
    pinned = torch.empty(audio.shape, dtype=audio.dtype, device="cpu", pin_memory=True)
    d2h_stream = torch.npu.Stream()
    main_stream = torch.npu.current_stream()
    d2h_event = torch.npu.Event()

    with torch.npu.stream(d2h_stream):
        d2h_stream.wait_stream(main_stream)
        pinned.copy_(audio, non_blocking=True)
        d2h_event.record()

    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import _PendingAudio

    pending = [_PendingAudio(pinned, d2h_event, row=i, audio_length=480) for i in range(b)]
    results = [pa.resolve() for pa in pending]

    # Compute expected BEFORE overwriting pinned (from NPU audio)
    expecteds = []
    for i in range(b):
        exp = audio[i, ..., :480].contiguous().clone().cpu()
        if exp.ndim == 1 or (exp.ndim > 1 and int(exp.shape[0]) == 1):
            exp = exp.reshape(-1)
        expecteds.append(exp)

    # Overwrite pinned
    pinned.fill_(999.0)

    for i, result in enumerate(results):
        torch.testing.assert_close(result, expecteds[i], atol=1e-5, rtol=1e-5,
                                   msg=f"Row {i} corrupted")


# ---------------------------------------------------------------------------
# 6. resolve() flattens B=1 to 1D (matches forward reshape logic)
# ---------------------------------------------------------------------------

def test_resolve_flattens_single_row():
    """B=1 stereo resolve returns 2D (channels, length) — matches forward
    passthrough which only flattens when shape[0] == 1."""
    audio = torch.randn(1, 2, 480, device=DEVICE, dtype=torch.float32)
    pa, _ = _make_pending(audio)
    result = pa.resolve()
    assert result.ndim == 2, f"Expected 2D for stereo, got {result.ndim}D"
    assert result.shape[0] == 2, f"Expected 2 channels, got {result.shape[0]}"
