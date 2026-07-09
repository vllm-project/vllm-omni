# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Audex TTA token space (phase ids, validation, masking)."""

import os

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate, MoveDirectionality

from vllm_omni.model_executor.models.audex.tta import (
    AUDEX_AUDIOCODEC_TOKEN_OFFSET,
    AUDEX_AUDIOCODEC_VOCAB_SIZE,
    AUDEX_AUDIOGEN_END_TOKEN_ID,
    AUDEX_AUDIOGEN_START_TOKEN_ID,
    XCODEC1_CODEBOOK_SIZE,
    XCODEC1_NUM_CODEBOOKS,
    TTARVQPhaseMaskLogitsProcessor,
    build_tta_phase_token_ids,
    validate_rvq_phase,
)


class _FakeTokenizer:
    """Contiguous <audiocodec_N> block plus the audiogen markers."""

    def __init__(self, offset: int = 1000, count: int = AUDEX_AUDIOCODEC_VOCAB_SIZE, drop: set[int] | None = None):
        self._vocab = {"<audiogen_start>": 7, "<audiogen_end>": 8}
        drop = drop or set()
        for n in range(count):
            if n not in drop:
                self._vocab[f"<audiocodec_{n}>"] = offset + n

    def get_vocab(self):
        return dict(self._vocab)


class TestPhaseTokenIds:
    def test_phases_group_by_codebook(self):
        phases, start_tid, end_tid = build_tta_phase_token_ids(_FakeTokenizer(offset=1000))
        assert (start_tid, end_tid) == (7, 8)
        assert len(phases) == XCODEC1_NUM_CODEBOOKS
        for p, ids in enumerate(phases):
            assert len(ids) == XCODEC1_CODEBOOK_SIZE
            assert ids[0] == 1000 + p * XCODEC1_CODEBOOK_SIZE
            assert ids[-1] == 1000 + (p + 1) * XCODEC1_CODEBOOK_SIZE - 1
        # Upper half of the codec vocab (ids 4096..8191) is never allowed.
        allowed = {i for ids in phases for i in ids}
        assert 1000 + 4 * XCODEC1_CODEBOOK_SIZE not in allowed

    def test_missing_marker_raises(self):
        tok = _FakeTokenizer()
        del tok._vocab["<audiogen_end>"]
        with pytest.raises(ValueError, match="marker"):
            build_tta_phase_token_ids(tok)

    def test_incomplete_codec_vocab_raises(self):
        with pytest.raises(ValueError, match="Incomplete codec vocab"):
            build_tta_phase_token_ids(_FakeTokenizer(drop={5}))


class TestValidateRvqPhase:
    def test_valid_interleaved_sequence(self):
        codes = [0, 1024, 2048, 3072, 1, 1025, 2049, 3073]
        result = validate_rvq_phase(codes)
        assert result["phase_valid"] and result["mismatch_count"] == 0
        assert result["codec_count"] == 8

    def test_mismatch_reported_with_position(self):
        codes = [0, 1024, 1, 3072]  # position 2 should be phase 2 (2048..3071)
        result = validate_rvq_phase(codes)
        assert not result["phase_valid"]
        assert result["mismatch_count"] == 1
        assert result["first_mismatch"] == [2, 1, 0, 2]


def _local_audiogen_dir() -> str | None:
    try:
        from huggingface_hub import snapshot_download

        root = snapshot_download("nvidia/Nemotron-Labs-Audex-2B", local_files_only=True)
        path = os.path.join(root, "checkpoint_folder_audiogen")
        return path if os.path.isdir(path) else None
    except Exception:
        return None


@pytest.mark.skipif(_local_audiogen_dir() is None, reason="Audex snapshot not in local HF cache")
def test_real_tokenizer_pins_audiocodec_layout():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_local_audiogen_dir())
    vocab = tok.get_vocab()
    assert vocab["<audiogen_start>"] == AUDEX_AUDIOGEN_START_TOKEN_ID
    assert vocab["<audiogen_end>"] == AUDEX_AUDIOGEN_END_TOKEN_ID
    assert vocab["<audiocodec_0>"] == AUDEX_AUDIOCODEC_TOKEN_OFFSET
    last = AUDEX_AUDIOCODEC_VOCAB_SIZE - 1
    assert vocab[f"<audiocodec_{last}>"] == AUDEX_AUDIOCODEC_TOKEN_OFFSET + last

    phases, start_tid, end_tid = build_tta_phase_token_ids(tok)
    assert (start_tid, end_tid) == (AUDEX_AUDIOGEN_START_TOKEN_ID, AUDEX_AUDIOGEN_END_TOKEN_ID)
    for p, ids in enumerate(phases):
        assert ids[0] == AUDEX_AUDIOCODEC_TOKEN_OFFSET + p * XCODEC1_CODEBOOK_SIZE
        assert len(ids) == XCODEC1_CODEBOOK_SIZE


# ---------------------------------------------------------------- mask processor

VOCAB = 32
# Tiny phase layout inside the test vocab: phase p allows ids [10+2p, 10+2p+1].
PHASES = [[10, 11], [12, 13], [14, 15], [16, 17]]
START_TID = 7
END_TID = 8


def _tta_params(codec_cap: int | None = None, start_in_prompt: bool = True) -> SamplingParams:
    return SamplingParams(
        extra_args={
            "tta_rvq": {
                "phase_token_ids": PHASES,
                "start_tid": START_TID,
                "end_tid": END_TID,
                "codec_cap": codec_cap,
                "start_in_prompt": start_in_prompt,
            }
        }
    )


def _processor() -> TTARVQPhaseMaskLogitsProcessor:
    return TTARVQPhaseMaskLogitsProcessor(vllm_config=None, device=torch.device("cpu"), is_pin_memory=False)


def _add(proc, added, removed=(), moved=(), batch_size: int = 8):
    proc.update_state(BatchUpdate(batch_size=batch_size, removed=list(removed), added=list(added), moved=list(moved)))


def _allowed(logits_row: torch.Tensor) -> set[int]:
    return set(torch.nonzero(logits_row > float("-inf")).reshape(-1).tolist())


class TestPhaseMask:
    def test_first_position_allows_only_phase0(self):
        proc = _processor()
        _add(proc, [(0, _tta_params(), None, [])])
        logits = torch.zeros(1, VOCAB)
        proc.apply(logits)
        assert _allowed(logits[0]) == set(PHASES[0])

    def test_interleave_walks_phases(self):
        proc = _processor()
        output_tokens: list[int] = []
        _add(proc, [(0, _tta_params(), None, output_tokens)])
        for step, expected_phase in [(1, 1), (2, 2), (3, 3)]:
            output_tokens.append(PHASES[(step - 1) % 4][0])
            logits = torch.zeros(1, VOCAB)
            proc.apply(logits)
            assert _allowed(logits[0]) == set(PHASES[expected_phase]), f"step {step}"

    def test_frame_boundary_allows_end_token(self):
        proc = _processor()
        output_tokens = [PHASES[i % 4][0] for i in range(4)]  # one full frame
        _add(proc, [(0, _tta_params(), None, output_tokens)])
        logits = torch.zeros(1, VOCAB)
        proc.apply(logits)
        assert _allowed(logits[0]) == set(PHASES[0]) | {END_TID}

    def test_codec_cap_forces_end(self):
        proc = _processor()
        output_tokens = [PHASES[i % 4][0] for i in range(4)]
        _add(proc, [(0, _tta_params(codec_cap=4), None, output_tokens)])
        logits = torch.zeros(1, VOCAB)
        proc.apply(logits)
        assert _allowed(logits[0]) == {END_TID}

    def test_start_not_in_prompt_waits_for_marker(self):
        proc = _processor()
        output_tokens: list[int] = [3, 4]  # no <audiogen_start> yet
        _add(proc, [(0, _tta_params(start_in_prompt=False), None, output_tokens)])
        logits = torch.zeros(1, VOCAB)
        proc.apply(logits)
        assert _allowed(logits[0]) == set(range(VOCAB))  # untouched

        output_tokens.append(START_TID)
        logits = torch.zeros(1, VOCAB)
        proc.apply(logits)
        assert _allowed(logits[0]) == set(PHASES[0])

    def test_non_tta_rows_untouched(self):
        proc = _processor()
        _add(proc, [(0, _tta_params(), None, []), (1, SamplingParams(), None, [])])
        logits = torch.zeros(2, VOCAB)
        proc.apply(logits)
        assert _allowed(logits[1]) == set(range(VOCAB))

    def test_moved_row_keeps_masking(self):
        proc = _processor()
        output_tokens: list[int] = []
        _add(proc, [(0, _tta_params(), None, output_tokens)])
        _add(proc, [], moved=[(0, 3, MoveDirectionality.UNIDIRECTIONAL)])
        logits = torch.zeros(4, VOCAB)
        proc.apply(logits)
        assert _allowed(logits[3]) == set(PHASES[0])
        assert _allowed(logits[0]) == set(range(VOCAB))

    def test_missing_tta_args_for_first_request_raises_on_apply(self):
        proc = _processor()
        proc._req[0] = {"start_tid": START_TID, "end_tid": END_TID, "codec_cap": None, "start_in_prompt": True}
        proc._output_tokens[0] = []
        with pytest.raises(RuntimeError, match="phase token ids missing"):
            proc.apply(torch.zeros(1, VOCAB))


class TestMaskSensitivity:
    """The phase-validity check must catch what unmasked sampling produces.

    This is the unit-level control for the e2e RVQ gate: the e2e cannot
    disable the mask in-engine (a different logits-processor config needs a
    second engine boot), so sensitivity is proven here against the official
    validator semantics.
    """

    def test_unmasked_sampling_violates_phase_validity(self):
        rng = torch.Generator().manual_seed(0)
        # Uniform draws over the full 4096-codec space, as an unmasked
        # sampler would produce: phase-invalid with overwhelming probability.
        codes = torch.randint(0, 4096, (64,), generator=rng).tolist()
        result = validate_rvq_phase(codes)
        assert not result["phase_valid"]
        assert result["mismatch_count"] > 0

    def test_masked_construction_passes_validity(self):
        # What the mask permits at each position: phase p codes only.
        codes = [(i % 4) * XCODEC1_CODEBOOK_SIZE + (i * 7) % XCODEC1_CODEBOOK_SIZE for i in range(64)]
        assert validate_rvq_phase(codes)["phase_valid"]


class TestPartialStepRobustness:
    """Rows beyond a step's logits must be skipped, not crash the engine."""

    def test_mask_skips_rows_beyond_step(self):
        proc = _processor()
        # Persistent-batch rows 2/3 tracked, but this step schedules 2 rows.
        _add(proc, [(2, _tta_params(), None, []), (3, _tta_params(), None, [])])
        logits = torch.zeros(2, VOCAB)
        proc.apply(logits)
        assert _allowed(logits[0]) == set(range(VOCAB))
        assert _allowed(logits[1]) == set(range(VOCAB))

    def test_mask_applies_once_row_back_in_range(self):
        proc = _processor()
        _add(proc, [(1, _tta_params(), None, [])])
        short = torch.zeros(1, VOCAB)
        proc.apply(short)  # row 1 out of range: untouched
        assert _allowed(short[0]) == set(range(VOCAB))

        full = torch.zeros(2, VOCAB)
        proc.apply(full)  # row 1 in range: masked to phase 0
        assert _allowed(full[1]) == set(PHASES[0])
