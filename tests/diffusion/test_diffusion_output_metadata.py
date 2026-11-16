# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm_omni.outputs import output_metadata
from vllm_omni.outputs.output_metadata import (
    strip_internal_metadata,
    validate_diffusion_metadata,
    validate_public_diffusion_metadata,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("value", "expected_len", "expected"),
    [
        ([7], 1, True),
        ((7,), 1, True),
        ([7, 9], 2, True),
        ((7, 9), 2, True),
        ([], 0, True),
        (None, 1, False),
        ([], 1, False),
        ([7, 9], 1, False),
        ([7], 2, False),
        ([0], 1, False),
        ([-1], 1, False),
        ([True], 1, False),
        ([False], 1, False),
        ([7, True], 2, False),
        ([7.0], 1, False),
        (["7"], 1, False),
        ([np.int64(7)], 1, False),
        (7, 1, False),
        ("7", 1, False),
        (b"7", 1, False),
        ({"count": 7}, 1, False),
        (range(1, 2), 1, False),
    ],
)
def test_valid_frame_counts_requires_aligned_positive_builtin_ints(value, expected_len, expected) -> None:
    assert output_metadata.valid_frame_counts(value, expected_len) is expected


def test_validate_accepts_canonical_metadata_fields() -> None:
    validate_diffusion_metadata(
        {
            "video": {
                "fps": 24.0,
                "video_fps_multiplier": 2,
                "frame_counts": [16, 17],
                "shape": [1, 3, 16, 64, 64],
            },
            "audio": {
                "sample_rate": 48000,
            },
            "actions": {
                "raw_action_dim": 7,
                "action_mode": "policy",
                "domain_id": 3,
            },
            "transfer": {
                "controls": {"edge": torch.zeros(1, 3, 4, 8, 8)},
                "hints": ["edge"],
            },
            "text": {
                "text_output": "caption",
                "think_text": "reasoning",
            },
            "common": {
                "action_only_output": True,
            },
        }
    )


@pytest.mark.parametrize(
    ("metadata", "error_type"),
    [
        ({"video": {"fps": 0}}, ValueError),
        ({"video": {"fps": "24"}}, TypeError),
        ({"video": {"video_fps_multiplier": 0}}, ValueError),
        ({"video": {"shape": [1, 0, 64]}}, ValueError),
        ({"video": {"frame_counts": [0]}}, ValueError),
        ({"video": {"frame_counts": [True]}}, ValueError),
        ({"video": {"frame_counts": "16"}}, TypeError),
        ({"audio": {"sample_rate": 0}}, ValueError),
        ({"actions": {"raw_action_dim": 0}}, ValueError),
        ({"actions": {"action_mode": ""}}, ValueError),
        ({"actions": {"domain_id": "7"}}, TypeError),
        ({"transfer": {"controls": {"edge": "not-a-tensor"}}}, TypeError),
        ({"transfer": {"hints": ["edge", 1]}}, TypeError),
        ({"text": {"text_output": 1}}, TypeError),
        ({"text": {"think_text": 1}}, TypeError),
        ({"common": {"action_only_output": "true"}}, TypeError),
    ],
)
def test_validate_rejects_invalid_canonical_metadata(metadata: dict[str, object], error_type: type[Exception]) -> None:
    with pytest.raises(error_type):
        validate_diffusion_metadata(metadata)


def test_validate_allows_unknown_model_metadata() -> None:
    validate_diffusion_metadata(
        {
            "model_specific": {"opaque": object()},
            "video": {"model_key": object()},
        }
    )


def test_public_metadata_rejects_internal_group() -> None:
    with pytest.raises(ValueError, match="internal"):
        validate_public_diffusion_metadata({"internal": {"secret": object()}})


def test_strip_internal_metadata_removes_private_group() -> None:
    metadata = {
        "actions": {"action_mode": "policy"},
        "internal": {"robolab_action_postprocess": object()},
    }

    public_metadata = strip_internal_metadata(metadata)

    assert public_metadata == {"actions": {"action_mode": "policy"}}
    assert "internal" not in public_metadata
