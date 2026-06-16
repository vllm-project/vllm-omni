# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU correctness tests for the CSM-1B Stage-0 -> Stage-1 input processor.

Locks the CSM-specific valid-frame mask and the codebook-major handoff:
  * ``backbone2mimi`` (sync): drops the all-zero EOS frame and any frame holding
    a reserved code (>= 2048), then flattens the survivors CODEBOOK-MAJOR.
  * ``_extract_last_frame``: all-zero frame -> None (nothing to decode), 1D / 2D
    handling, invalid shape -> ValueError.
  * ``backbone2mimi_async_chunk``: invalid chunk config -> ValueError; a finished
    short request flushes its windowed frames codebook-major; an empty finished
    request emits an empty finished payload.
"""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors import csm as proc

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeBackboneOutput:
    def __init__(self, audio_codes, finished=True):
        self.finished = finished
        self.outputs = [SimpleNamespace(multimodal_output={"codes": {"audio": audio_codes}})]


# --------------------------------------------------------------------------
# backbone2mimi (sync collect)
# --------------------------------------------------------------------------


def test_backbone2mimi_drops_eos_and_reserved_frames_then_flattens_codebook_major():
    frames = torch.zeros(3, 32, dtype=torch.long)
    frames[0, :] = 1  # valid frame (all ones)
    # frame 1 stays all-zero -> EOS -> dropped
    frames[2, :] = 1
    frames[2, 5] = 2048  # reserved code -> whole frame dropped (max >= 2048)

    out = proc.backbone2mimi([_FakeBackboneOutput(frames)])
    assert len(out) == 1
    # Only frame 0 survives; codebook-major flat of a single all-ones frame = 32 ones.
    assert out[0]["prompt_token_ids"] == [1] * 32


def test_backbone2mimi_codebook_major_order():
    # Two valid frames; verify the flatten is codebook-major (all of cb0, then cb1...).
    frames = torch.zeros(2, 4, dtype=torch.long)
    frames[0] = torch.tensor([10, 11, 12, 13])
    frames[1] = torch.tensor([20, 21, 22, 23])
    out = proc.backbone2mimi([_FakeBackboneOutput(frames)])
    # transpose(0,1) -> [[10,20],[11,21],[12,22],[13,23]] -> flat row-major.
    assert out[0]["prompt_token_ids"] == [10, 20, 11, 21, 12, 22, 13, 23]


def test_backbone2mimi_skips_unfinished_request():
    frames = torch.ones(2, 32, dtype=torch.long)
    out = proc.backbone2mimi([_FakeBackboneOutput(frames, finished=False)])
    assert out == []


def test_backbone2mimi_empty_audio_emits_empty_prompt():
    out = proc.backbone2mimi([_FakeBackboneOutput(torch.empty((0,), dtype=torch.long))])
    assert len(out) == 1
    assert out[0]["prompt_token_ids"] == []


def test_backbone2mimi_reshapes_1d_audio_to_frames():
    flat = torch.ones(64, dtype=torch.long)  # 2 frames of 32, flattened frame-major
    out = proc.backbone2mimi([_FakeBackboneOutput(flat)])
    assert len(out) == 1
    assert len(out[0]["prompt_token_ids"]) == 64


# --------------------------------------------------------------------------
# _extract_last_frame
# --------------------------------------------------------------------------


def test_extract_last_frame_returns_last_nonzero_2d_frame():
    codes = torch.zeros(3, 32, dtype=torch.long)
    codes[-1, :] = 7
    frame = proc._extract_last_frame({"codes": {"audio": codes}})
    assert frame is not None
    assert frame.tolist() == [7] * 32


def test_extract_last_frame_none_for_all_zero_eos_frame():
    codes = torch.zeros(2, 32, dtype=torch.long)
    assert proc._extract_last_frame({"codes": {"audio": codes}}) is None


def test_extract_last_frame_passthrough_1d():
    codes = torch.arange(32, dtype=torch.long)
    frame = proc._extract_last_frame({"codes": {"audio": codes}})
    assert frame.tolist() == list(range(32))


def test_extract_last_frame_none_for_missing_audio():
    assert proc._extract_last_frame({"codes": {}}) is None
    assert proc._extract_last_frame({"codes": {"audio": torch.empty((0,), dtype=torch.long)}}) is None


def test_extract_last_frame_raises_on_invalid_shape():
    bad = torch.zeros(2, 3, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="Invalid audio_codes shape"):
        proc._extract_last_frame({"codes": {"audio": bad}})


# --------------------------------------------------------------------------
# backbone2mimi_async_chunk
# --------------------------------------------------------------------------


def _tm(config):
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        connector=SimpleNamespace(config=config),
    )


def test_async_chunk_rejects_invalid_chunk_config():
    tm = _tm({"extra": {"codec_chunk_frames": 0}})
    req = SimpleNamespace(external_req_id="r", is_finished=lambda: False)
    with pytest.raises(ValueError, match="codec_chunk_frames"):
        proc.backbone2mimi_async_chunk(
            tm, {"codes": {"audio": torch.ones(1, 32, dtype=torch.long)}}, req, is_finished=False
        )


def test_async_chunk_finished_with_no_frames_emits_empty_finished_payload():
    tm = _tm({})
    req = SimpleNamespace(external_req_id="r", is_finished=lambda: True)
    out = proc.backbone2mimi_async_chunk(tm, None, req, is_finished=True)
    assert out is not None
    assert out.codes.audio.numel() == 0
    assert bool(out.meta.finished) is True


def test_async_chunk_flushes_codebook_major_window_on_finish():
    tm = _tm({"extra": {"codec_chunk_frames": 25, "codec_left_context_frames": 0}})
    req = SimpleNamespace(external_req_id="r", is_finished=lambda: False)

    # Frame 0 arrives unfinished -> buffered, nothing flushed yet.
    out0 = proc.backbone2mimi_async_chunk(tm, {"codes": {"audio": torch.tensor([[1, 2, 3]])}}, req, is_finished=False)
    assert out0 is None

    # Frame 1 arrives on the finishing step -> flush the 2-frame window.
    out1 = proc.backbone2mimi_async_chunk(tm, {"codes": {"audio": torch.tensor([[4, 5, 6]])}}, req, is_finished=True)
    assert out1 is not None
    # codebook-major: all of cb0 (1,4), then cb1 (2,5), then cb2 (3,6).
    assert out1.codes.audio.tolist() == [1, 4, 2, 5, 3, 6]
    assert int(out1.meta.left_context_size) == 0
    assert bool(out1.meta.finished) is True
