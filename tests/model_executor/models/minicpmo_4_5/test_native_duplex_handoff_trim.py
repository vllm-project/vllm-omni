# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Guard for the native-duplex Talker handoff when decision tokens outlive their rows.

A segment the session consumes without a Talker handoff (a listen decision, or
a ``speak`` whose turn ended with nothing to say) advances the request's
cumulative output, but the next streaming update folds those tokens into the
rewritten prompt and the forwarded hidden ledger starts over. The next handoff
then slices rows for tokens decoded before the rewrite -- rows that are gone --
and the ``missing own-token hidden states`` ValueError fails the request (the
serving crash logged at orchestrator.py:2513). These tests pin the trim that
skips the stale leading decision run, and that it never touches normal inputs.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import (
    _native_duplex_trim_unbacked_decisions,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

#: listen / speak / turn_eos / chunk_eos of MiniCPM-o 4.5 (decoded from the
#: crash log: unit=[151706, 151717, 151705, 151706, 576, 883, 151718]).
LISTEN_ID = 151705
SPEAK_ID = 151706
TURN_EOS_ID = 151717
CHUNK_EOS_ID = 151718
TTS_BOS_ID = 151703

SPECIAL_TOKEN_IDS = {
    "listen_token_id": LISTEN_ID,
    "speak_token_id": SPEAK_ID,
    "turn_eos_token_id": TURN_EOS_ID,
    "chunk_eos_token_id": CHUNK_EOS_ID,
    "chunk_tts_eos_token_id": 151719,
    "tts_bos_token_id": TTS_BOS_ID,
}


def _ledger_mm_output(ids: list[int], positions: list[int]) -> dict[str, object]:
    """A handoff ``multimodal_output`` carrying only the row ledger."""
    return {
        "latent_input_ids": torch.tensor(ids, dtype=torch.long).reshape(-1, 1),
        "latent_positions": torch.tensor(positions, dtype=torch.long).reshape(-1, 1),
    }


def test_crash_shape_trims_stale_leading_decisions() -> None:
    """The exact unit from the serving crash, against its actual ledger tail.

    The ledger only backs the trailing ``[speak, 576, 883]`` decode rows; the
    leading ``[speak, turn_eos, listen]`` were folded into the rewritten
    prompt. The trim drops exactly those control decisions, leaving a handoff
    the ledger can satisfy.
    """
    ledger_ids = [128244] * 18 + [151670] + [151698] * 10 + [SPEAK_ID, 576, 883]
    ledger_positions = list(range(len(ledger_ids)))
    mm_output = _ledger_mm_output(ledger_ids, ledger_positions)
    unit = [SPEAK_ID, TURN_EOS_ID, LISTEN_ID, SPEAK_ID, 576, 883, CHUNK_EOS_ID]

    trimmed = _native_duplex_trim_unbacked_decisions(unit, mm_output, SPECIAL_TOKEN_IDS, request_id="req-crash")

    assert trimmed == [SPEAK_ID, 576, 883, CHUNK_EOS_ID]


def test_normal_unit_is_untouched() -> None:
    """A unit whose rows are present is returned unchanged (no trim, no warning)."""
    ledger_ids = [128244] * 18 + [SPEAK_ID, 576, 883]
    mm_output = _ledger_mm_output(ledger_ids, list(range(len(ledger_ids))))
    unit = [SPEAK_ID, 576, 883, CHUNK_EOS_ID]

    trimmed = _native_duplex_trim_unbacked_decisions(unit, mm_output, SPECIAL_TOKEN_IDS, request_id="req-ok")

    assert trimmed == unit


def test_listen_run_without_rows_is_skipped_when_remainder_backed() -> None:
    """Accumulated listen decisions ahead of a speak are skippable too."""
    ledger_ids = [151698] * 10 + [SPEAK_ID, 42]
    mm_output = _ledger_mm_output(ledger_ids, list(range(len(ledger_ids))))
    unit = [LISTEN_ID, LISTEN_ID, SPEAK_ID, 42, CHUNK_EOS_ID]

    trimmed = _native_duplex_trim_unbacked_decisions(unit, mm_output, SPECIAL_TOKEN_IDS, request_id="req-listen")

    assert trimmed == [SPEAK_ID, 42, CHUNK_EOS_ID]


def test_text_tokens_are_never_trimmed() -> None:
    """The trim stops at the first non-control token: stale text is an error, not a trim."""
    ledger_ids = [SPEAK_ID, 883]
    mm_output = _ledger_mm_output(ledger_ids, list(range(len(ledger_ids))))
    unit = [SPEAK_ID, TURN_EOS_ID, 576, 883, CHUNK_EOS_ID]

    trimmed = _native_duplex_trim_unbacked_decisions(unit, mm_output, SPECIAL_TOKEN_IDS, request_id="req-text")

    assert trimmed == unit


def test_missing_ledger_returns_input_unchanged() -> None:
    unit = [SPEAK_ID, 576, 883]
    trimmed = _native_duplex_trim_unbacked_decisions(unit, {}, SPECIAL_TOKEN_IDS, request_id="req-nol")
    assert trimmed == unit


def test_unbackable_input_returns_input_unchanged() -> None:
    """When no suffix is backed either, the input is left for the explicit error."""
    ledger_ids = [151698] * 4
    mm_output = _ledger_mm_output(ledger_ids, list(range(len(ledger_ids))))
    unit = [SPEAK_ID, 576, 883, CHUNK_EOS_ID]

    trimmed = _native_duplex_trim_unbacked_decisions(unit, mm_output, SPECIAL_TOKEN_IDS, request_id="req-bad")

    assert trimmed == unit
