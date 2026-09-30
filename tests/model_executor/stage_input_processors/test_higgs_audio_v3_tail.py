# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""No checkpoint/GPU needed: preserve real frames, exclude EOC from codec input."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.higgs_audio_v3 import (
    talker2code2wav,
    talker2code2wav_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def delayed_rows(frames, tail):
    # Distinct values make missing/duplicated/shifted frames observable.
    real = torch.arange(8 * frames).reshape(8, frames) % 1024
    if tail == "mixed_eoc":
        # Captured failure: penultimate frame has EOC only in cb0; the
        # final frame has EOC in cb0..6 and -1 in cb7. Trimming one is wrong.
        suffix = torch.tensor([[1025, 1025]] + [[100 + q, 1025] for q in range(1, 7)] + [[107, -1]])
        codes = torch.cat([real, suffix], dim=1)
    elif tail == "all_eoc":
        codes = torch.cat([real, torch.full((8, 1), 1025)], dim=1)
    else:
        codes = real
    rows = torch.full((codes.shape[1] + 7, 8), 1024)
    for q in range(8):
        rows[q : q + codes.shape[1], q] = codes[q]
    return real, rows


@pytest.mark.parametrize("frames", [0, 1, 25, 144])
@pytest.mark.parametrize("tail", ["mixed_eoc", "all_eoc", "none"])
def test_sync_preserves_exact_real_frames(frames, tail):
    real, rows = delayed_rows(frames, tail)
    output = SimpleNamespace(
        finished=True, outputs=[SimpleNamespace(multimodal_output={"codes": {"audio": rows}})]
    )
    result = talker2code2wav([output])[0]["prompt_token_ids"]
    assert result == real.reshape(-1).tolist()


@pytest.mark.parametrize("frames", [0, 1, 25, 144])
@pytest.mark.parametrize("tail", ["mixed_eoc", "all_eoc", "none"])
@pytest.mark.parametrize("holdback", [0, 1, 4])
@pytest.mark.parametrize("left_context", [0, 3])
@pytest.mark.parametrize("chunk_size", [1, 25])
@pytest.mark.parametrize("separate_finish", [False, True])
def test_stream_matches_sync_without_invalid_context(frames, tail, holdback, left_context, chunk_size, separate_finish):
    real, rows = delayed_rows(frames, tail)
    tm = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": chunk_size,
                    "initial_codec_chunk_frames": 1,
                    "codec_left_context_frames": left_context,
                    "codec_right_holdback_frames": holdback,
                }
            }
        ),
    )
    request = SimpleNamespace(external_req_id="test", is_finished=lambda: False)
    emitted = []
    finished = []

    def consume(payload):
        if payload is None:
            return
        if payload.meta.finished is not None and bool(payload.meta.finished):
            finished.append(True)
        if payload.codes.audio.numel() == 0:
            return
        codes = payload.codes.audio.reshape(8, -1)
        assert bool(((codes >= 0) & (codes < 1024)).all())
        left = int(payload.meta.left_context_size)
        right = int(payload.meta.right_holdback_size)
        # Context also needs to be the exact real-code sequence: zeros
        # substituted for EOC are in range, but still corrupt the waveform.
        offset = sum(c.shape[1] for c in emitted) - left
        assert torch.equal(codes, real[:, offset : offset + codes.shape[1]])
        emitted.append(codes[:, left : codes.shape[1] - right])

    for i, row in enumerate(rows):
        consume(
            talker2code2wav_async_chunk(
                tm,
                {"codes": {"audio": row[None]}},
                request,
                is_finished=not separate_finish and i == len(rows) - 1,
            )
        )
    if separate_finish:
        consume(talker2code2wav_async_chunk(tm, None, request, is_finished=True))
    actual = torch.cat(emitted, dim=1) if emitted else torch.empty((8, 0), dtype=torch.long)
    assert torch.equal(actual, real)
    assert len(finished) == 1
    assert "test" not in tm.higgs_v3_emitted_frames
