# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""#8710 phase-1 equivalence: the multi-chunk merge plumbing keeps per-unit semantics.

Three contracts:
1. The K-terminator queue is a strict superset of the legacy single token
   (the property shim: single writes replace, reads are FIFO, K writes queue).
2. A single-unit append (offsets of length 1) never enters the multi-boundary
   path: no boundary rows, no outcomes, the legacy per-unit path is unchanged.
3. All-listen acceptance queues one terminator per accepted unit, so the next
   append re-injects exactly what per-unit execution would have.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.mrv2_sampling import (
    MiniCPMO45DuplexSampler,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import (
    _MiniCPMO45Stage0SessionState,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration,
)
from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CHUNK_EOS = 42


def _state() -> _MiniCPMO45Stage0SessionState:
    return _MiniCPMO45Stage0SessionState(session_id="s0")


def _sampler(
    boundary: dict[int, torch.Tensor],
    offsets_by_req: dict[str, list[int]],
    seeds_by_req: dict[str, int] | None = None,
) -> Any:
    seeds_by_req = seeds_by_req or {}
    infos = {
        req_id: {
            "duplex": {"duplex_boundary_offsets": offsets},
            "sampling_params": SimpleNamespace(seed=seeds_by_req.get(req_id)),
        }
        for req_id, offsets in offsets_by_req.items()
    }
    model = SimpleNamespace(
        _mrv2_sampling_infos=infos,
        _minicpmo45_duplex_token_ids={"chunk_eos_token_id": CHUNK_EOS},
    )

    def probs(logits: torch.Tensor, eos_id: int) -> torch.Tensor:
        # The mock policy: one Bernoulli probability per boundary row.
        return torch.sigmoid(logits[:, 0])

    model._duplex_boundary_chunk_eos_probs = probs
    sampler = object.__new__(MiniCPMO45DuplexSampler)
    sampler.model = model
    sampler.generators = {}
    input_batch = SimpleNamespace(
        _duplex_boundary_logits=boundary,
        req_ids=list(offsets_by_req),
    )
    return sampler, input_batch


# --------------------------------------------------------------------- 1. shim


def test_single_token_shim_replaces_the_queue() -> None:
    state = _state()
    assert state.pending_terminator_token is None
    assert state.pending_terminator_tokens == []

    state.pending_terminator_token = 7
    assert state.pending_terminator_tokens == [7]

    state.pending_terminator_tokens = [7, 8, 9]
    assert state.pending_terminator_token == 7  # FIFO head

    state.pending_terminator_token = None
    assert state.pending_terminator_tokens == []
    assert state.pending_terminator_token is None


def test_queue_write_coerces_to_ints() -> None:
    state = _state()
    state.pending_terminator_tokens = [True, 3.0, CHUNK_EOS]
    assert state.pending_terminator_tokens == [1, 3, CHUNK_EOS]
    assert all(isinstance(t, int) for t in state.pending_terminator_tokens)


# ------------------------------------------------- 2. single-unit stays legacy


def test_multi_boundary_outcomes_empty_without_boundary_rows() -> None:
    sampler, batch = _sampler({}, {"r0": [5]})
    assert sampler._duplex_multi_boundary_outcomes(batch) == {}
    batch = SimpleNamespace(req_ids=["r0"], generators={})  # attribute absent
    assert sampler._duplex_multi_boundary_outcomes(batch) == {}


def test_multi_boundary_outcomes_skip_single_unit_and_mismatched_rows() -> None:
    # offsets of length 1 is a normal single-unit append: skipped.
    sampler, batch = _sampler({0: torch.zeros(1, 8)}, {"r0": [5]})
    assert sampler._duplex_multi_boundary_outcomes(batch) == {}
    # K-1 logits must match K offsets; a mismatched shape is skipped.
    sampler, batch = _sampler({0: torch.zeros(3, 8)}, {"r0": [3, 4, 5]})
    assert sampler._duplex_multi_boundary_outcomes(batch) == {}


def test_multi_boundary_outcomes_report_all_listen_and_first_speak() -> None:
    # probabilities of 1.0 always draw chunk_eos -> all-listen.
    sampler, batch = _sampler(
        {0: torch.full((2, 8), 20.0)},
        {"r0": [3, 4, 5]},
    )
    outcomes = sampler._duplex_multi_boundary_outcomes(batch)
    assert outcomes == {
        0: {"all_listen": True, "first_speak": None, "offsets": [3, 4, 5]},
    }

    # A probability of 0.0 never draws chunk_eos -> the first boundary speaks.
    sampler, batch = _sampler(
        {0: torch.full((2, 8), -20.0)},
        {"r0": [3, 4, 5]},
    )
    outcomes = sampler._duplex_multi_boundary_outcomes(batch)
    assert outcomes == {
        0: {"all_listen": False, "first_speak": 0, "offsets": [3, 4, 5]},
    }


def test_multi_boundary_outcomes_draw_from_the_request_seeded_generator() -> None:
    """Intermediate draws consume the per-request seeded stream in order.

    A seeded request's K-1 intermediate draws must come from the same
    get-or-create generator the regular boundary draw uses, so the final
    boundary sees exactly the draws separately scheduled appends would have
    left it (Dong1017's review point).
    """
    sampler, batch = _sampler({0: torch.zeros(2, 8)}, {"r0": [3, 4, 5]}, seeds_by_req={"r0": 1234})
    outcomes = sampler._duplex_multi_boundary_outcomes(batch)
    assert 0 in outcomes

    # The call created and cached the request's seeded generator...
    row_gen = sampler.generators["r0"]
    assert isinstance(row_gen, torch.Generator)

    # ...and consumed exactly K-1 = 2 draws from it: the next draw from the
    # cached generator matches the reference stream seeded the same way.
    ref = torch.Generator().manual_seed(1234)
    _ = torch.rand((2,), generator=ref)  # the two draws the call consumed
    assert torch.equal(torch.rand((1,), generator=row_gen), torch.rand((1,), generator=ref))

    # A second call continues the same stream (draws 3 and 4).
    sampler, batch = _sampler({0: torch.zeros(2, 8)}, {"r0": [3, 4, 5]}, seeds_by_req={"r0": 1234})
    sampler.generators["r0"] = row_gen
    outcomes = sampler._duplex_multi_boundary_outcomes(batch)
    _ = torch.rand((2,), generator=ref)  # the second call's two draws
    assert torch.equal(torch.rand((1,), generator=row_gen), torch.rand((1,), generator=ref))


# ------------------------------------------------ 3. all-listen acceptance hook


def _hook_model(state: _MiniCPMO45Stage0SessionState, multi: Any) -> Any:
    model = MagicMock()
    model._minicpmo45_duplex_state_for_row.return_value = state
    model._minicpmo45_duplex_payload_for_row.return_value = None
    model._minicpmo45_duplex_multi_boundaries = multi
    model._minicpmo45_chunk_terminator_token_ids = (
        MiniCPMO45OmniForConditionalGeneration._minicpmo45_chunk_terminator_token_ids
    )
    return model


def test_all_listen_acceptance_queues_one_terminator_per_unit() -> None:
    state = _state()
    model = _hook_model(state, {0: {"all_listen": True, "offsets": [3, 4, 5]}})
    token_ids = {"chunk_eos_token_id": CHUNK_EOS, "listen_token_id": 7, "tts_bos_token_id": 9}
    MiniCPMO45OmniForConditionalGeneration._record_minicpmo45_duplex_terminator(model, 0, CHUNK_EOS, token_ids)
    assert state.pending_terminator_tokens == [CHUNK_EOS, CHUNK_EOS, CHUNK_EOS]
    assert state.pending_terminator_token == CHUNK_EOS  # FIFO view stays honest


def test_legacy_acceptance_stays_a_single_terminator() -> None:
    # No multi outcome for the row: the legacy single-terminator semantics.
    state = _state()
    model = _hook_model(state, {})
    token_ids = {"chunk_eos_token_id": CHUNK_EOS, "listen_token_id": 7, "tts_bos_token_id": 9}
    MiniCPMO45OmniForConditionalGeneration._record_minicpmo45_duplex_terminator(model, 0, CHUNK_EOS, token_ids)
    assert state.pending_terminator_tokens == [CHUNK_EOS]

    # A speak outcome (all_listen False) is not an all-listen span either.
    state = _state()
    model = _hook_model(state, {0: {"all_listen": False, "first_speak": 1, "offsets": [3, 4, 5]}})
    MiniCPMO45OmniForConditionalGeneration._record_minicpmo45_duplex_terminator(model, 0, CHUNK_EOS, token_ids)
    assert state.pending_terminator_tokens == [CHUNK_EOS]


# ------------------------------------------------ 4. runner boundary positions


def _runner(offsets: list[int], *, prefilling: bool) -> Any:
    runner = SimpleNamespace(
        model_intermediate_buffer={"r0": {"duplex": {"duplex_boundary_offsets": offsets}}},
        model_state=SimpleNamespace(compute_sample_logits=lambda hidden: hidden * 10),
    )
    hidden = torch.arange(12, dtype=torch.float32).reshape(12, 1)
    batch = SimpleNamespace(
        num_reqs=1,
        req_ids=["r0"],
        is_prefilling_np=np.array([prefilling]),
        num_computed_prefill_tokens_np=np.array([0]),
        num_scheduled_tokens=np.array([12]),
        prefill_len_np=np.array([12]),
        query_start_loc_np=np.array([2]),
    )
    return runner, hidden, batch


def test_boundary_logits_cover_the_intermediate_unit_ends() -> None:
    runner, hidden, batch = _runner([3, 4, 5], prefilling=False)
    rows = OmniARModelRunner._duplex_boundary_logits(runner, hidden, batch)
    # Offsets [3, 4, 5] are cumulative unit ends; the K-1 intermediate ends sit
    # at query_start + offset - 1 -> rows 2+3-1=4 and 2+4-1=5.
    assert sorted(rows) == [0]
    assert torch.equal(rows[0], torch.stack([hidden[4] * 10, hidden[5] * 10]))


def test_partial_prefills_are_skipped_until_the_append_rearrives() -> None:
    runner, hidden, batch = _runner([3, 4, 5], prefilling=True)
    # 0 computed + 12 scheduled == 12 prefill len: this prefill is complete.
    assert OmniARModelRunner._duplex_boundary_logits(runner, hidden, batch) is not None
    batch.num_scheduled_tokens = np.array([5])  # partial: 0 + 5 < 12
    assert OmniARModelRunner._duplex_boundary_logits(runner, hidden, batch) is None


def test_single_unit_append_has_no_boundary_rows() -> None:
    runner, hidden, batch = _runner([5], prefilling=False)
    assert OmniARModelRunner._duplex_boundary_logits(runner, hidden, batch) is None
