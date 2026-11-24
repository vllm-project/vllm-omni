# SPDX-License-Identifier: Apache-2.0
"""Bimanual action-schema and model-input validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.action_inputs import (
    prepare_action_values,
    validate_action_values,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.action_schema import (
    Cosmos3NanoSimBimanualActionSchema,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.config import Cosmos3NanoSimBimanualManifest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
FIXTURES = Path(__file__).with_name("fixtures")


def schema_payload() -> dict:
    return json.loads((FIXTURES / "unified_action_schema.json").read_text())


def manifest() -> Cosmos3NanoSimBimanualManifest:
    return Cosmos3NanoSimBimanualManifest(
        checkpoint_id="cpu-fixture",
        checkpoint_iteration=1,
        checkpoint_hash="a" * 64,
        chunk_size=2,
        window_frames=None,
        conditioning=Cosmos3NanoSimBimanualActionSchema.model_validate({"mode": "action", **schema_payload()}),
    )


def test_action_schema_loads() -> None:
    schema = manifest().require_action_schema()
    assert schema.schema_version == 5
    assert schema.input_action_dim == 59
    assert schema.model_action_dim == 64
    assert schema.embodiment_to_domain["agibotworld"] == 15
    assert schema.embodiment_to_domain["camera_pose"] == 2


@pytest.mark.parametrize(
    "field",
    ["contract_sha256", "action_dim", "runtime_normalization", "embodiment_dim", "layout", "source_sha256"],
)
def test_action_schema_rejects_tampering(field: str) -> None:
    payload = copy.deepcopy(schema_payload())
    if field == "contract_sha256":
        payload[field] = "0" * 64
    elif field == "action_dim":
        payload["input_contract"]["action_dim"] = 29
    elif field == "runtime_normalization":
        payload["input_contract"]["runtime_normalization"] = True
    elif field == "embodiment_dim":
        payload["embodiments"]["agibotworld"]["input_action_dim"] = 29
    elif field == "layout":
        payload["layout"]["fields"][0]["size"] = 8
    else:
        payload["normalizer"]["source"]["sha256"] = "invalid"
    with pytest.raises(ValidationError):
        Cosmos3NanoSimBimanualActionSchema.model_validate({"mode": "action", **payload})


def test_unified_actions_are_padded_without_modification() -> None:
    values = torch.arange(3 * 59, dtype=torch.float32).reshape(3, 59)
    prepared = prepare_action_values(values, width=59, model_width=64)
    torch.testing.assert_close(prepared[:, :59], values, rtol=0, atol=0)
    assert prepared.shape == (3, 64)
    assert not torch.count_nonzero(prepared[:, 59:])


@pytest.mark.parametrize(
    "value",
    [
        [[True] * 59],
        [["1"] * 59],
        [[float("nan")] * 59],
        [[float("inf")] * 59],
        [],
        [[1.0] * 58],
        torch.ones(2, 59, dtype=torch.bool),
    ],
)
def test_invalid_action_values(value: object) -> None:
    with pytest.raises(ValueError):
        validate_action_values(value, width=59)
