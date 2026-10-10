# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for Pi-family checkpoint-name translation."""

import pytest

from vllm_omni.diffusion.models.pi.common import checkpoint

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

VISION_PREFIX = "paligemma_with_expert.paligemma.model.vision_tower."


@pytest.mark.parametrize("submodule", ["vision_tower", "multi_modal_projector", "language_model"])
def test_resolver_strips_wrapper_and_nests_paligemma_submodules(submodule):
    name = f"model.paligemma_with_expert.paligemma.{submodule}.layer.weight"
    expected = f"paligemma_with_expert.paligemma.model.{submodule}.layer.weight"

    assert checkpoint.resolve_parameter_name(name, {expected}) == expected


def test_resolver_redirects_tied_lm_head_to_token_embeddings():
    expected = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"

    actual = checkpoint.resolve_parameter_name(
        "model.paligemma_with_expert.paligemma.lm_head.weight",
        {expected},
    )

    assert actual == expected


@pytest.mark.parametrize(
    "name,aliases,expected",
    [
        ("model.time_mlp_in.weight", (("time_mlp_in.", "action_time_mlp_in."),), "action_time_mlp_in.weight"),
        (
            "model.action_time_mlp_out.bias",
            (("action_time_mlp_out.", "time_mlp_out."),),
            "time_mlp_out.bias",
        ),
    ],
)
def test_resolver_applies_variant_owned_prefix_aliases(name, aliases, expected):
    assert checkpoint.resolve_parameter_name(name, {expected}, prefix_aliases=aliases) == expected


@pytest.mark.parametrize(
    "checkpoint_suffix,model_suffix",
    [
        ("vision_model.embeddings.patch_embedding.weight", "embeddings.patch_embedding.weight"),
        ("embeddings.patch_embedding.weight", "vision_model.embeddings.patch_embedding.weight"),
    ],
)
def test_resolver_reconciles_transformers_vision_tower_layout(checkpoint_suffix, model_suffix):
    expected = VISION_PREFIX + model_suffix

    assert checkpoint.resolve_parameter_name(VISION_PREFIX + checkpoint_suffix, {expected}) == expected


def test_resolver_preserves_unknown_name_when_no_candidate_exists():
    name = VISION_PREFIX + "vision_model.unknown.weight"

    assert checkpoint.resolve_parameter_name(name, set()) == name
