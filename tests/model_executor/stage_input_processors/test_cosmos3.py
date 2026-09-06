# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the Cosmos3 reasoner -> generator stage bridge."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.cosmos3 import (
    KV_KEY,
    META_KEY,
    _as_dict,
    _find_und_payload,
    reasoner2generator,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# =============================================================================
# Helpers
# =============================================================================


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
        "num_kv_heads_local": 8,
        "head_dim": 128,
        "tp_size": 1,
    }
    meta.update(overrides)
    return meta


def _reasoner_output(
    table: dict[str, list] | None = None,
    meta: dict[str, Any] | None = None,
    *,
    nested: bool = True,
    on_completion: bool = False,
):
    """Build a reasoner ``RequestOutput`` in one of the envelopes seen in practice.

    ``nested`` mirrors what the output formatter actually produces (the payload
    under the ``trajectory`` key); ``on_completion`` mirrors the external-transport
    path, where the payload is promoted onto the inner completion output.
    """
    payload = {KV_KEY: table if table is not None else _kv_table(), META_KEY: meta if meta is not None else _meta()}
    mm = {"trajectory": payload} if nested else dict(payload)
    if on_completion:
        return SimpleNamespace(multimodal_output=None, outputs=[SimpleNamespace(multimodal_output=mm)])
    return SimpleNamespace(multimodal_output=mm, outputs=[SimpleNamespace(multimodal_output=None)])


# =============================================================================
# Tests for _as_dict
# =============================================================================


class TestAsDict:
    def test_none(self):
        assert _as_dict(None) == {}

    def test_dict_passthrough(self):
        prompt = {"prompt": "a cat"}
        assert _as_dict(prompt) is prompt

    def test_list_takes_first(self):
        assert _as_dict([{"prompt": "first"}, {"prompt": "second"}]) == {"prompt": "first"}

    def test_empty_list(self):
        assert _as_dict([]) == {}

    def test_object_with_dunder_dict(self):
        assert _as_dict(SimpleNamespace(prompt="a cat")) == {"prompt": "a cat"}

    def test_unsupported_type(self):
        assert _as_dict("a bare string") == {}


# =============================================================================
# Tests for _find_und_payload
# =============================================================================


class TestFindUndPayload:
    @pytest.mark.parametrize("nested", [True, False])
    @pytest.mark.parametrize("on_completion", [True, False])
    def test_locates_payload_in_every_envelope(self, nested: bool, on_completion: bool):
        output = _reasoner_output(nested=nested, on_completion=on_completion)

        payload = _find_und_payload([output])

        assert payload is not None
        assert KV_KEY in payload

    def test_no_multimodal_output(self):
        assert _find_und_payload([SimpleNamespace(multimodal_output=None, outputs=[])]) is None

    def test_multimodal_output_without_kv(self):
        output = SimpleNamespace(multimodal_output={"trajectory": {"latents": []}}, outputs=[])
        assert _find_und_payload([output]) is None

    def test_empty_source_list(self):
        assert _find_und_payload([]) is None


# =============================================================================
# Tests for reasoner2generator
# =============================================================================


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

    def test_does_not_forward_geometry_or_tokenization_settings(self):
        """Both stages resolve these from the same sampling params, through the same
        ``_resolve_t2i_geometry`` / ``_resolve_text_encode_params`` helpers, and the
        generator does not read them from its prompt dict at all. They ride in
        ``META_KEY`` purely as diagnostics."""
        output = _reasoner_output(meta=_meta(height=512, width=768, use_system_prompt=True, max_sequence_length=256))

        result = reasoner2generator([output], prompt={"prompt": "x", "height": 1024, "width": 1024})

        assert not {"height", "width", "use_system_prompt", "max_sequence_length"} & set(result)
        assert result["extra"][META_KEY]["height"] == 512
        assert result["extra"][META_KEY]["max_sequence_length"] == 256

    def test_forwards_the_kv_layout_for_the_generators_consistency_check(self):
        """The generator compares these against its own config to report a
        stage-configuration mismatch in those terms rather than as a bare shape
        error from inside cross-attention."""
        output = _reasoner_output(meta=_meta(num_layers=64, num_kv_heads_local=4, head_dim=128, tp_size=2))

        meta = reasoner2generator([output], prompt={"prompt": "x"})["extra"][META_KEY]

        assert (meta["num_layers"], meta["num_kv_heads_local"], meta["head_dim"], meta["tp_size"]) == (64, 4, 128, 2)

    def test_forwards_negative_prompt(self):
        """With guidance active the reasoner encoded an unconditional branch from
        it, and the generator re-tokenizes it to look that branch up."""
        result = reasoner2generator(
            [_reasoner_output(_kv_table(num_branches=2))],
            prompt={"prompt": "a red car", "negative_prompt": "blurry"},
        )

        assert result["negative_prompt"] == "blurry"

    def test_omits_absent_negative_prompt(self):
        result = reasoner2generator([_reasoner_output()], prompt={"prompt": "a red car"})

        assert "negative_prompt" not in result

    def test_does_not_forward_generation_knobs(self):
        """Sampling params ride to every stage untouched; re-copying them into the
        prompt would create a second, competing source of truth."""
        prompt = {
            "prompt": "a red car",
            "seed": 7,
            "num_inference_steps": 12,
            "guidance_scale": 4.5,
            "flow_shift": 2.0,
        }

        result = reasoner2generator([_reasoner_output()], prompt=prompt)

        assert not {"seed", "num_inference_steps", "guidance_scale", "flow_shift"} & set(result)

    def test_accepts_a_prompt_list(self):
        result = reasoner2generator([_reasoner_output()], prompt=[{"prompt": "first"}])

        assert result["prompt"] == "first"

    def test_tolerates_a_missing_prompt(self):
        result = reasoner2generator([_reasoner_output()], prompt=None)

        assert result["prompt"] == ""
        assert result["extra"][KV_KEY]

    def test_requires_multimodal_data_is_ignored_for_t2i(self):
        result = reasoner2generator([_reasoner_output()], prompt={"prompt": "x"}, requires_multimodal_data=True)

        assert "pil_image" not in result

    def test_empty_source_outputs_drops_the_request(self):
        assert reasoner2generator([], prompt={"prompt": "x"}) is None

    def test_reasoner_output_without_kv_drops_the_request(self):
        """Better a dropped request than a generator forward with no K/V to replay."""
        output = SimpleNamespace(multimodal_output={"trajectory": {}}, outputs=[])

        assert reasoner2generator([output], prompt={"prompt": "x"}) is None
