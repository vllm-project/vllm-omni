# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch
from torch import nn
from vllm.model_executor.models.utils import WeightsMapper

from vllm_omni.model_executor.models.weight_loader import AutoWeightsLoader


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kept = nn.Linear(1, 1, bias=False)
        self.skipped = nn.Linear(1, 1, bias=False)
        self.rotary_embed = nn.Module()
        self.rotary_embed.register_parameter("inv_freq", nn.Parameter(torch.zeros(1)))


def test_auto_weights_loader_preserves_removed_skip_arguments() -> None:
    model = _TinyModel()
    model.skipped.weight.data.zero_()

    loaded = AutoWeightsLoader(
        model,
        skip_prefixes=["skipped."],
        skip_substrs=["rotary_embed.inv_freq"],
    ).load_weights(
        [
            ("checkpoint.kept.weight", torch.ones(1, 1)),
            ("skipped.weight", torch.full((1, 1), 2.0)),
            ("rotary_embed.inv_freq", torch.ones(1)),
        ],
        mapper=WeightsMapper(orig_to_new_prefix={"checkpoint.": ""}),
    )

    assert loaded == {"kept.weight"}
    torch.testing.assert_close(model.kept.weight, torch.ones(1, 1))
    torch.testing.assert_close(model.skipped.weight, torch.zeros(1, 1))
    torch.testing.assert_close(model.rotary_embed.inv_freq, torch.zeros(1))
