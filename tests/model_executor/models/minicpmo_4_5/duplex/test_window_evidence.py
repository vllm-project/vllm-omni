# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_evidence import (
    EvidenceScorer,
    select_evidence_units,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_plan import prepare_stage0_window
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_retrieval import plan_retrieval

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_evidence_scorer_sees_silent_visual_and_audio_only_events():
    scorer = EvidenceScorer()
    assert scorer.score(b"\x00", 0.0, 0.0)[1] == 0
    assert scorer.score(b"\x01", 0.0, 0.0) == ("vision", 1)
    assert scorer.score(b"\x01", 1.0, 1.0) == ("audio", 1)
    assert scorer.score(b"\x02", 2.0, 1.0)[0] == "av"


def test_keep_evidence_and_recent_units_in_source_order():
    units = [{"length": 8, "evidence_score": score} for score in (0.8, 0.1, 0.9, 0.0, 0.0)]
    selected = select_evidence_units(units, 26, 10)
    assert [index for index, _ in selected] == [2, 3, 4]
    assert [unit.get("historical", False) for _, unit in selected] == [True, False, False]
    assert all("historical" not in unit for unit in units)


def test_oversized_unit_does_not_consume_history_capacity():
    units = [
        {"length": 20, "evidence_score": 1.0},
        {"length": 5, "evidence_score": 0.5},
        {"length": 8, "evidence_score": 0.0},
    ]
    assert [index for index, _ in select_evidence_units(units, 18, 10)] == [1, 2]


def test_history_projection_is_charged_once():
    units = [
        {"length": 79, "evidence_score": 1.0, "remove_media": (1, 66)},
        {"length": 79, "evidence_score": 0.0},
    ]
    selected = select_evidence_units(units, 99, 20)
    assert [unit["length"] for _, unit in selected] == [13, 79]
    assert "remove_media" not in selected[0][1]
    assert select_evidence_units([unit for _, unit in selected], 99, 20) == selected


def test_evidence_plan_tracks_noncontiguous_units_and_projected_length():
    session = SimpleNamespace(num_prompt_tokens=43, _minicpmo45_window_open_start=43)
    session._minicpmo45_window_units = [
        {"length": 10, "evidence_score": 0.9, "remove_media": (1, 4)},
        {"length": 10, "evidence_score": 0.1},
        {"length": 10, "evidence_score": 0.0},
        {"length": 10, "evidence_score": 0.0},
    ]
    duplex = {
        "seq": 6,
        "data_plane": True,
        "runtime_config": {
            "duplex_first_append_context_tokens": 3,
            "duplex_window_config": {
                "sliding_window_mode": "evidence",
                "basic_window_high_tokens": 45,
                "basic_window_low_tokens": 32,
                "evidence_history_tokens": 6,
            },
        },
    }
    update = SimpleNamespace(prompt_token_ids=[0] * 7, model_intermediate_buffer={"duplex": duplex})
    assert prepare_stage0_window(session, update, segment_output_ids=[])
    plan = duplex["stage0_window"]
    assert isinstance(plan, dict)
    assert plan["evidence_keep"] == [(0, 1, 4), (3, 0, 0), (4, 0, 0)]
    assert plan["replacement_prompt_len"] == 26
    assert len(update.prompt_token_ids) == 26


def test_query_restores_archived_opening_without_expanding_window():
    def unit(seq):
        return {
            "append_seq": seq,
            "length": 79,
            "modality_costs": (66, 10),
            "evidence_score": 0.8,
            "salience": (0.8, 0.0),
        }

    active = [unit(seq) for seq in range(30, 34)]
    archive = {seq: unit(seq) for seq in (1, 2)}
    kept, cold, plan = plan_retrieval(
        active,
        archive,
        projected_len=500,
        high=8000,
        low=6000,
        history_budget=800,
        relevance=[(1, 0.9, None), (2, 0.8, None)],
        restore_opening=True,
    )
    assert [item["append_seq"] for item in kept] == [1, 2, 33]
    assert plan is not None
    assert plan["retrieval_keep"] == [(-1, 0, 0, 79), (-2, 0, 0, 79), (3, 0, 0, 79)]
    assert set(cold) == {30, 31, 32}
    assert sum(item["length"] for item in kept) <= sum(item["length"] for item in active)
