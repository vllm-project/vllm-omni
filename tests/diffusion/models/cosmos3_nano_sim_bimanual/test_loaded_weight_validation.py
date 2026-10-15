# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Weight validation must accept the names every loader hands it.

Pipeline loading reports ``transformer.``-prefixed names. The pre-sharded HSDP
and layerwise-offload loaders strip that prefix before calling the transformer.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.transformer_cosmos3_nano_sim_bimanual import (
    Cosmos3NanoSimBimanualTransformer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PARAMETERS = ("language_model.embed_tokens.weight", "gen_layers.0.mlp.up_proj.weight")


def validate(loaded: set[str]) -> None:
    model = SimpleNamespace(
        named_parameters=lambda: ((name, torch.nn.Parameter(torch.zeros(1))) for name in PARAMETERS)
    )
    Cosmos3NanoSimBimanualTransformer.validate_loaded_weights(model, loaded)


@pytest.mark.parametrize("prefix", ["transformer.", ""])
def test_complete_weights_pass_with_or_without_pipeline_prefix(prefix):
    validate({prefix + name for name in PARAMETERS})


def test_other_components_in_the_pipeline_listing_are_ignored():
    validate({"transformer." + name for name in PARAMETERS} | {"vae.decoder.conv.weight"})


@pytest.mark.parametrize("prefix", ["transformer.", ""])
def test_missing_weight_is_reported_with_or_without_pipeline_prefix(prefix):
    loaded = {prefix + PARAMETERS[0]}
    with pytest.raises(ValueError, match=r"missing required transformer weights: transformer\.gen_layers\.0\.mlp"):
        validate(loaded)
