# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _DummyPipelineModel(nn.Module):
    def __init__(self, *, source_prefix: str):
        super().__init__()
        self.transformer = nn.Linear(2, 2, bias=False)
        self.vae = nn.Linear(2, 2, bias=False)
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path="dummy",
                subfolder="transformer",
                revision=None,
                prefix=source_prefix,
                fall_back_to_pt=True,
            )
        ]

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            if name not in params:
                continue
            params[name].data.copy_(tensor.to(dtype=params[name].dtype))
            loaded.add(name)
        return loaded


def _make_loader_with_weights(weight_names: list[str]) -> DiffusersPipelineLoader:
    loader = object.__new__(DiffusersPipelineLoader)
    loader.counter_before_loading_weights = 0.0
    loader.counter_after_loading_weights = 0.0

    def _iter_weights(_model):
        for name in weight_names:
            yield name, torch.zeros((2, 2))

    loader.get_all_weights = _iter_weights  # type: ignore[assignment]
    return loader


def test_strict_check_only_validates_source_prefix_parameters():
    model = _DummyPipelineModel(source_prefix="transformer.")
    loader = _make_loader_with_weights(["transformer.weight"])

    # Should not require VAE parameters because they are outside weights_sources.
    loader.load_weights(model)


def test_strict_check_raises_when_source_parameters_are_missing():
    model = _DummyPipelineModel(source_prefix="transformer.")
    loader = _make_loader_with_weights([])

    with pytest.raises(ValueError, match="transformer.weight"):
        loader.load_weights(model)


def test_empty_source_prefix_keeps_full_model_strict_check():
    model = _DummyPipelineModel(source_prefix="")
    loader = _make_loader_with_weights(["transformer.weight"])

    with pytest.raises(ValueError, match="vae.weight"):
        loader.load_weights(model)
