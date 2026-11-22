# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare the causal planner against imaginaire4-generated replay-visibility fixtures.

The fixtures encode the reference teacher-forcing replay predicate evaluated per
token (see ``fixtures/reference_visibility/README.md``). Comparing against them,
rather than against this package's own predicate, catches mis-ported rules such as
the observed-condition role.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.causal_plan import build_causal_maskless_plan, dense_token_mask
from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.geometry import SensorGrid, build_chunk_schedule
from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.layout import (
    ROLE_UND,
    CaptionDocument,
    RingSlot,
    TokenMetadata,
    build_chunk_items,
    caption_metadata,
    chunk_token_metadata,
    committed_metadata,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

FIXTURES = sorted((Path(__file__).parent / "fixtures" / "reference_visibility").glob("*.json"))
UNIT = 64


def _identities(metadata: TokenMetadata, counts: dict) -> list[tuple[int, int, int, int]]:
    result = []
    for view, frame, role in zip(metadata.view_id.tolist(), metadata.frame_id.tolist(), metadata.role.tolist()):
        key = (view, frame, role)
        result.append((view, frame, role, counts[key]))
        counts[key] += 1
    return result


def _build(scenario: dict):
    joint = scenario["joint"]
    grid = SensorGrid(1, scenario["tokens_per_frame"], 1, 1)
    schedule = build_chunk_schedule(
        rgb_latents=1 + scenario["frames_per_chunk"] * 6,
        lidar_sweeps=1 + scenario["sweeps_per_chunk"] * 6 if joint else None,
        frames_per_chunk=scenario["frames_per_chunk"],
        rgb_seconds_per_frame=scenario["rgb_seconds_per_frame"],
        lidar_seconds_per_frame=scenario["lidar_seconds_per_frame"] if joint else None,
    )

    def items(step: int):
        return build_chunk_items(
            schedule[step],
            num_views=scenario["num_views"],
            rgb_grid=grid,
            rgb_seconds_per_frame=scenario["rgb_seconds_per_frame"],
            lidar_grid=grid if joint else None,
            lidar_seconds_per_frame=scenario["lidar_seconds_per_frame"] if joint else None,
            rgb_condition_frames=(0,) if scenario["rgb_condition"] else (),
            lidar_condition_sweeps=(0,) if scenario["lidar_condition"] else (),
        )

    step = scenario["step"]
    slots = []
    if step >= 1:
        slots.append(RingSlot(0, 0, committed_metadata(chunk_token_metadata(items(0), step=0, pass_kind="clean")[0])))
    if step >= 2:
        slots.append(
            RingSlot(
                1, UNIT, committed_metadata(chunk_token_metadata(items(step - 1), step=step - 1, pass_kind="clean")[0])
            )
        )
    current, _ = chunk_token_metadata(items(step), step=step, pass_kind=scenario["pass_kind"])
    documents = tuple(
        CaptionDocument(view, scenario["caption_tokens"], start_seconds=0.0, end_seconds=float("inf"))
        for view in range(scenario["num_views"])
    )
    captions = caption_metadata(documents)
    plan = build_causal_maskless_plan(
        current=current, history=slots, captions=captions, window_seconds=scenario["window_seconds"], device="cpu"
    )
    counts: dict = defaultdict(int)
    queries = _identities(current, counts)
    key_counts: dict = defaultdict(int)
    keys = _identities(current, key_counts)
    for slot in slots:
        keys.extend([None] * (len(queries) + slot.base - len(keys)))  # virtual padding before this slot
        keys.extend([(*identity[:3], identity[3]) for identity in _identities(slot.metadata, key_counts)])
    for view in range(scenario["num_views"]):
        for _ in range(scenario["caption_tokens"]):
            key = (view, -1, ROLE_UND)
            keys.append((view, -1, ROLE_UND, key_counts[key]))
            key_counts[key] += 1
    mask = dense_token_mask(plan)
    assert mask.shape == (len(queries), len(keys))
    return queries, keys, mask


@pytest.mark.parametrize(
    "fixture_path",
    FIXTURES
    or [
        pytest.param(
            None,
            marks=pytest.mark.skip(
                reason="reference visibility fixtures not generated; see fixtures/reference_visibility/README.md"
            ),
        )
    ],
    ids=lambda path: "missing" if path is None else path.stem,
)
def test_planner_matches_reference_fixtures(fixture_path: Path) -> None:
    for fixture in json.loads(fixture_path.read_text()):
        scenario = fixture["scenario"]
        queries, keys, mask = _build(scenario)
        query_index = {tuple(identity): row for row, identity in enumerate(queries)}
        key_index = {tuple(identity): column for column, identity in enumerate(keys) if identity is not None}
        reference_queries = [tuple(identity) for identity in fixture["queries"]]
        reference_keys = [tuple(identity[:4]) for identity in fixture["keys"]]
        missing_q = [identity for identity in reference_queries if identity not in query_index]
        missing_k = [identity for identity in reference_keys if identity not in key_index]
        assert not missing_q and not missing_k, (scenario, missing_q[:5], missing_k[:5])
        assert len(reference_queries) == len(queries) and len(reference_keys) == len(key_index), scenario
        expected = torch.tensor(fixture["allowed"], dtype=torch.bool)
        rows = torch.tensor([query_index[identity] for identity in reference_queries])
        columns = torch.tensor([key_index[identity] for identity in reference_keys])
        actual = mask[rows][:, columns]
        mismatch = (actual ^ expected).nonzero().tolist()
        assert not mismatch, (scenario, [(reference_queries[r], reference_keys[c]) for r, c in mismatch[:8]])
