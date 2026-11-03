# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the Cosmos3 reasoner -> generator stage bridge."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3_pipeline_config import COSMOS3_UND_SCHEMA
from vllm_omni.model_executor.stage_input_processors.cosmos3 import (
    KV_KEY,
    META_KEY,
    _find_und_payload,
    reasoner2generator,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _kv_table(num_branches: int = 1, num_layers: int = 2) -> dict[str, list]:
    """A stand-in for the reasoner's fingerprint-keyed per-layer K/V table."""
    return {
        f"fingerprint{branch}": [(torch.zeros(1, 3, 8, 128), torch.ones(1, 3, 8, 128)) for _ in range(num_layers)]
        for branch in range(num_branches)
    }


def _meta(**overrides: Any) -> dict[str, Any]:
    meta = {
        "height": 1024,
        "width": 1024,
        "max_sequence_length": 512,
        "use_system_prompt": False,
        "num_branches": 1,
        "payload_mib": 12.5,
        "num_layers": 2,
        "num_kv_heads": 8,
        "head_dim": 128,
        "reasoner_tp_size": 1,
        "schema": COSMOS3_UND_SCHEMA,
    }
    meta.update(overrides)
    return meta


def _reasoner_output(
    table: dict[str, list] | None = None,
    meta: dict[str, Any] | None = None,
    *,
    on_completion: bool = False,
):
    """A reasoner output with the payload on the request or on its completion."""
    payload = {KV_KEY: table if table is not None else _kv_table(), META_KEY: meta if meta is not None else _meta()}
    mm = {"trajectory": payload}
    if on_completion:
        return SimpleNamespace(multimodal_output=None, outputs=[SimpleNamespace(multimodal_output=mm)])
    return SimpleNamespace(multimodal_output=mm, outputs=[SimpleNamespace(multimodal_output=None)])


class TestFindUndPayload:
    @pytest.mark.parametrize("on_completion", [True, False])
    def test_locates_payload_in_every_envelope(self, on_completion: bool):
        output = _reasoner_output(on_completion=on_completion)

        payload = _find_und_payload([output])

        assert payload is not None
        assert KV_KEY in payload


class TestReasoner2Generator:
    def test_builds_generator_prompt(self):
        table = _kv_table(num_branches=2)
        result = reasoner2generator([_reasoner_output(table)], prompt={"prompt": "a red car"})

        assert result["prompt"] == "a red car"
        # The generator stage re-runs _is_t2i_request, which keys purely off this.
        assert result["modalities"] == ["image"]
        assert result["extra"][KV_KEY] is table
        assert result["extra"][META_KEY]["payload_mib"] == 12.5  # meta passes through intact

    def test_forwards_exactly_prompt_modalities_and_kv(self):
        """The bridge's whole job, stated as a closed set.

        Anything else it added would be a second source of truth for a value the
        generator resolves from its own sampling params.
        """
        result = reasoner2generator([_reasoner_output()], prompt={"prompt": "a red car"})

        assert set(result) == {"prompt", "modalities", "extra"}
        assert set(result["extra"]) == {KV_KEY, META_KEY}

    def test_forwards_negative_prompt(self):
        """With guidance active the reasoner encoded an unconditional branch from
        it, and the generator re-tokenizes it to look that branch up."""
        result = reasoner2generator(
            [_reasoner_output(_kv_table(num_branches=2))],
            prompt={"prompt": "a red car", "negative_prompt": "blurry"},
        )

        assert result["negative_prompt"] == "blurry"

    def test_reasoner_output_without_kv_drops_the_request(self):
        """Better a dropped request than a generator forward with no K/V to replay."""
        output = SimpleNamespace(multimodal_output={"trajectory": {}}, outputs=[])

        assert reasoner2generator([output], prompt={"prompt": "x"}) is None
