# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise conditioning cache ownership with a small transformer shell."""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.mossimage import mossimage_transformer as model

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class Text(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, rope, mask):
        self.calls += 1
        return x * 2


def shell():
    transformer = model.MOSSImageTransformer.__new__(model.MOSSImageTransformer)
    nn.Module.__init__(transformer)
    transformer.latent_channels = 128
    transformer.video_embedder = nn.Linear(128, 128)
    transformer.rope = lambda h, w, s, device, text_lengths=None: torch.zeros(1, h * w + s, 128, device=device)
    transformer.text_embedder = Text()
    transformer._time_modulation = lambda t, b, d: (None, None)
    transformer.denoise_body = lambda x, rope, mod, mask: x + x.mean(dim=1, keepdim=True)
    transformer.final_out = lambda x, head: x
    return transformer


def test_cache_reuses_conditioning_and_isolates_branches(monkeypatch):
    monkeypatch.setattr(model, "get_sequence_parallel_world_size", lambda: 1)
    transformer = shell()
    latent = torch.randn(1, 128, 2, 2)
    positive = torch.randn(1, 3, 128)
    negative = torch.randn(1, 3, 128)
    cache: dict = {}
    with torch.no_grad():

        def run(context, cache=None):
            return model.MOSSImageTransformer.forward(
                transformer, latent, 1.0, context, return_dict=False, _conditioning_cache=cache
            )[0]

        expected = run(positive)
        first = run(positive, cache)
        second = run(positive, cache)
        assert transformer.text_embedder.calls == 2
        torch.testing.assert_close(first, expected, atol=0, rtol=0)
        torch.testing.assert_close(second, expected, atol=0, rtol=0)
        negative_cache: dict = {}
        different = run(negative, negative_cache)
        assert transformer.text_embedder.calls == 3
        assert not torch.equal(different, first)
        assert cache["conditioning"][1] is not negative_cache["conditioning"][1]
        run(positive, {})  # A new request does not reuse an earlier cache.
        assert transformer.text_embedder.calls == 4


def test_sp_and_autograd_do_not_populate_cache(monkeypatch):
    transformer = shell()
    latent = torch.randn(1, 128, 2, 2)
    text = torch.randn(1, 4, 128)
    for sp, grad in [(2, False), (1, True)]:
        monkeypatch.setattr(model, "get_sequence_parallel_world_size", lambda: sp)
        cache: dict = {}
        with torch.set_grad_enabled(grad):
            model.MOSSImageTransformer.forward(transformer, latent, 1.0, text, _conditioning_cache=cache)
        assert not cache
