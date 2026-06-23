# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for ArState (delayed pattern state machine).

Tests the delayed codebook pattern [0, D, D] state tracking:
  - Stream activity follows delays
  - Token recording works correctly
  - EOS detection per stream
  - Completion signaling
  - Output code formatting
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.model_executor.models.songgeneration_v2.ar_state import (
    ArState,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_state(
    code_depth: int = 3,
    code_size: int = 17,  # = config.code_size + 1
    max_gen_len: int = 10,
    delays: list[int] | None = None,
) -> ArState:
    if delays is None:
        delays = [0, 3, 3]
    return ArState.create(
        code_depth=code_depth,
        code_size=code_size,
        max_gen_len=max_gen_len,
        delays=delays,
    )


def test_initial_state():
    state = _make_state()
    assert state.step == 0
    assert not state.finished
    assert state.active_streams == [0]
    assert state.stream0_frame == 0
    assert state.stream_frame(1) == -3
    assert state.stream_frame(2) == -3


def test_stream_activation_after_delay():
    state = _make_state(delays=[0, 3, 3])

    # Steps 0-2: only stream 0 active
    for s in range(3):
        assert state.active_streams == [0], f"step {s}"
        state.record_stream0_token(100)
        state.fill_inactive_streams()
        state.advance()

    # Step 3: all streams active
    assert state.step == 3
    assert state.active_streams == [0, 1, 2]
    assert state.stream_frame(1) == 0
    assert state.stream_frame(2) == 0


def test_record_stream0_token():
    state = _make_state()
    state.record_stream0_token(42)
    assert state.gen_codes[0, 0].item() == 42
    assert not state.is_end[0]


def test_eos_detection_stream0():
    state = _make_state(code_size=17)
    eos = state.eos_token_id  # = 16
    state.record_stream0_token(eos)
    assert state.is_end[0]


def test_record_stream12_tokens():
    state = _make_state(delays=[0, 2, 2])
    # Advance to step 2 so streams 1+2 become active
    state.record_stream0_token(1)
    state.advance()
    state.record_stream0_token(2)
    state.advance()

    assert state.step == 2
    assert state.active_streams == [0, 1, 2]
    assert state.stream_frame(1) == 0

    # Record tokens for streams 1+2 at their frame 0
    state.record_stream12_tokens(torch.tensor([55, 66]))
    assert state.gen_codes[1, 0].item() == 55
    assert state.gen_codes[2, 0].item() == 66


def test_finishes_on_all_eos():
    state = _make_state(delays=[0, 2, 2], max_gen_len=20)
    eos = state.eos_token_id

    # Steps 0-1: only stream 0
    state.record_stream0_token(1)
    state.advance()
    state.record_stream0_token(eos)
    state.advance()
    # Stream 0 done, streams 1+2 just activated
    assert state.is_end[0]
    assert not state.finished  # streams 1+2 not done yet

    # Step 2: stream 0 already done, record streams 1+2 EOS
    state.record_stream0_token(state.special_token_id)
    state.record_stream12_tokens(torch.tensor([eos, eos]))
    state.advance()

    assert state.finished


def test_finishes_at_max_gen_len():
    state = _make_state(delays=[0, 2, 2], max_gen_len=5)

    for _ in range(5):
        state.record_stream0_token(100)
        state.fill_inactive_streams()
        state.advance()

    assert state.finished


def test_get_output_codes_format():
    state = _make_state(delays=[0, 1, 1], max_gen_len=4)

    # Generate 3 steps:
    # step 0: stream 0 produces token 10 (streams 1+2 not yet active)
    # step 1: stream 0 → 20, streams 1+2 frame 0 → [11, 12]
    # step 2: stream 0 → 30, streams 1+2 frame 1 → [21, 22]
    state.record_stream0_token(10)
    state.advance()
    state.record_stream0_token(20)
    state.record_stream12_tokens(torch.tensor([11, 12]))
    state.advance()
    state.record_stream0_token(30)
    state.record_stream12_tokens(torch.tensor([21, 22]))
    state.advance()

    out = state.get_output_codes()
    # Output is [T_out, K] frame-major where T_out = step - max_delay = 3 - 1 = 2
    # Only 2 aligned frames exist (where all streams have produced data).
    assert out.shape == (2, 3)
    # Frame 0: stream 0 step-0 token, streams 1+2 frame-0 tokens
    assert out[0, 0].item() == 10
    assert out[0, 1].item() == 11
    assert out[0, 2].item() == 12
    # Frame 1: stream 0 step-1 token, streams 1+2 frame-1 tokens
    assert out[1, 0].item() == 20
    assert out[1, 1].item() == 21
    assert out[1, 2].item() == 22


def test_should_compute_stream12():
    state = _make_state(delays=[0, 3, 3])

    # Before delay: no stream12 computation needed
    assert not state.should_compute_stream12()

    # Advance past delay
    for _ in range(3):
        state.record_stream0_token(5)
        state.advance()

    # Now streams 1+2 are active
    assert state.should_compute_stream12()


def test_get_stream0_input_token():
    state = _make_state()
    # First step: returns special token (BOS)
    assert state.get_stream0_input_token() == state.special_token_id

    state.record_stream0_token(42)
    state.advance()
    # Second step: returns last generated token
    assert state.get_stream0_input_token() == 42


def test_single_stream_mode():
    """Test with code_depth=1 (no streams 1+2)."""
    state = _make_state(code_depth=1, delays=[0], max_gen_len=5)

    assert state.active_streams == [0]
    assert not state.should_compute_stream12()

    for i in range(5):
        state.record_stream0_token(i + 1)
        state.advance()

    assert state.finished
    out = state.get_output_codes()
    assert out.shape == (5, 1)
    assert out[:, 0].tolist() == [1, 2, 3, 4, 5]
