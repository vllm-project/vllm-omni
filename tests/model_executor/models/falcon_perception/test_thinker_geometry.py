# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Falcon Perception coordinate normalization and request-scoped deduplication."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.falcon_perception.falcon_perception_thinker import (
    FalconPerceptionThinker,
    _select_coordinate_from_logits,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_coordinate_bins_cover_the_closed_unit_interval():
    bins = 4
    logits = torch.full((1, 2, bins), -10.0)
    logits[0, 0, -1] = 10.0
    logits[0, 1, -1] = 10.0
    stub = SimpleNamespace(coord_decoder=lambda hidden: logits.reshape(1, -1).expand(hidden.shape[0], -1))

    decoded = FalconPerceptionThinker._decode_coords(stub, torch.zeros(1, 3))

    assert torch.equal(decoded, torch.ones(1, 2))


def test_coordinate_dedup_is_scoped_to_the_supplied_request_history():
    # Both axes prefer bin 1, then bin 2. With four bins those centres are
    # 1/3 and 2/3 respectively.
    logits = torch.tensor(
        [
            [0.0, 5.0, 4.0, 1.0],
            [0.0, 5.0, 4.0, 1.0],
        ]
    )
    repeated_for_request_a = torch.tensor([[1.0 / 3.0, 1.0 / 3.0]])

    selected_a = _select_coordinate_from_logits(logits, repeated_for_request_a)
    selected_b = _select_coordinate_from_logits(logits, None)

    assert torch.allclose(selected_a, torch.tensor([[2.0 / 3.0, 2.0 / 3.0]]))
    assert torch.allclose(selected_b, torch.tensor([[1.0 / 3.0, 1.0 / 3.0]]))


def test_make_omni_output_emits_selected_coordinates_as_per_request_deltas():
    stub = SimpleNamespace(
        config=SimpleNamespace(perception_heads=True),
        _decode_coords=lambda hidden: torch.zeros(hidden.shape[0], 2),
        _decode_sizes=lambda hidden: torch.ones(hidden.shape[0], 2),
    )
    runtime_infos = [
        {
            "falcon_perception_geometry": {
                "selected_xy": torch.tensor([[0.25, 0.75]]),
                "selected_xy_active": torch.tensor([True]),
            }
        },
        {
            "falcon_perception_geometry": {
                "selected_xy": torch.tensor([[0.5, 0.5]]),
                "selected_xy_active": torch.tensor([False]),
            }
        },
    ]

    output = FalconPerceptionThinker.make_omni_output(
        stub,
        torch.zeros(2, 3),
        runtime_additional_information=runtime_infos,
    )
    selected = output.multimodal_outputs["hidden_states"]["selected_box_xy"]

    assert torch.equal(selected[0], torch.tensor([[0.25, 0.75]]))
    assert selected[1].shape == (0, 2)
