"""Regression test: t2i constraint metadata must follow batch condensation.

The model caches the per-request metadata list during forward, but a replayed
FULL decode graph skips the Python forward. The runner therefore refreshes the
cache via set_runtime_additional_information() every step, outside any
capture. This pins the failure sequence from review: prefill [A, B] with
different grid widths, finish A, decode B in slot 0 — the constraint logic
must read B's metadata at index 0, not A's.
"""

import pytest
import torch

from vllm_omni.model_executor.models.mammoth_moda2.mammoth_moda2 import (
    MammothModa2ARForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

VOCAB = 32
EOL = 20
VIS_START, VIS_END = 8, 19


def _info(ar_width: int, generated_len: int) -> dict:
    return {
        "meta": {
            "omni_task": ["t2i"],
            "ar_width": [ar_width],
            "eol_token_id": [EOL],
            "visual_token_start_id": [VIS_START],
            "visual_token_end_id": [VIS_END],
        },
        "generated_len": generated_len,
    }


class _FakeLM:
    base_vocab_size = VOCAB


class _FakeModel:
    language_model = _FakeLM()

    def __init__(self):
        self._last_runtime_additional_information = None

    set_runtime_additional_information = MammothModa2ARForConditionalGeneration.set_runtime_additional_information

    def apply(self, logits):
        return MammothModa2ARForConditionalGeneration._apply_t2i_token_constraints(self, logits)


def test_constraints_follow_batch_condensation():
    model = _FakeModel()
    # Step with batch [A (ar_width=4), B (ar_width=2)], both at generated_len=2:
    # A is mid-row (2 % 5 != 4) -> visual tokens only; B is at a row boundary
    # (2 % 3 == 2) -> EOL forced.
    a, b = _info(4, 2), _info(2, 2)
    model.set_runtime_additional_information([a, b])

    # A finishes; batch condenses to [B] in slot 0. A replayed FULL decode
    # graph skips forward, so only the runner-side refresh updates the cache.
    model.set_runtime_additional_information([b])

    logits = model.apply(torch.zeros(1, VOCAB))
    row = logits[0]
    # B's row-boundary constraint: EOL is the only allowed token.
    assert row[EOL] == 0.0
    assert torch.isinf(row[:EOL]).all() and (row[:EOL] < 0).all()
    assert torch.isinf(row[EOL + 1 :]).all()


def test_stale_metadata_would_apply_wrong_constraint():
    # Sanity for the test itself: with the stale list (A still at index 0),
    # the same logits get A's mid-row constraint instead - the exact wrong
    # behavior the runner-side refresh prevents.
    model = _FakeModel()
    a, b = _info(4, 2), _info(2, 2)
    model.set_runtime_additional_information([a, b])

    logits = model.apply(torch.zeros(1, VOCAB))
    row = logits[0]
    # A mid-row: visual tokens allowed, EOL forbidden.
    assert torch.isinf(row[EOL]) and row[EOL] < 0
    assert (row[VIS_START : VIS_END + 1] == 0.0).all()


def test_setter_rejects_non_list():
    model = _FakeModel()
    model.set_runtime_additional_information([_info(4, 0)])
    model.set_runtime_additional_information("not-a-list")
    assert model._last_runtime_additional_information is None
