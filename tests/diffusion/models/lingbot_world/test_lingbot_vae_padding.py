# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
ROOT = Path(__file__).parents[4] / "vllm_omni/platforms/npu/models/lingbot_world"


def load(name):
    spec = importlib.util.spec_from_file_location("_lingbot_" + name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WanLikeConv(torch.nn.Conv3d):
    def __init__(self):
        super().__init__(2, 3, 3, padding=0)
        self._padding = (1, 1, 1, 1, 2, 0)
        self._lingbot_padding_verdicts = {}
        self._lingbot_original_forward = self.forward

    def forward(self, x, cache_x=None):
        p = list(self._padding)
        if cache_x is not None:
            x = torch.cat((cache_x, x), 2)
            p[4] -= cache_x.shape[2]
        return super().forward(torch.nn.functional.pad(x, p))


@pytest.mark.parametrize("history", [0, 1, 2, 3])
def test_vae_padding_preserves_temporal_cache_and_crop(history):
    module = load("vae")
    conv = WanLikeConv()
    x = torch.randn(1, 2, 3, 7, 9)
    cache = torch.randn(1, 2, history, 7, 9) if history else None
    expected = conv(x, cache)
    torch.testing.assert_close(module.spatial_padding_candidate(conv, x, cache), expected)
    torch.testing.assert_close(module.verified_spatial_forward(conv, x, cache), expected, atol=0, rtol=0)


def test_vae_padding_mismatch_returns_reference_and_remembers_fallback(monkeypatch):
    module = load("vae")
    conv = WanLikeConv()
    x = torch.randn(1, 2, 3, 7, 9)
    calls = []

    def wrong(*args):
        calls.append(1)
        return conv(x) + 1

    monkeypatch.setattr(module, "spatial_padding_candidate", wrong)
    for _ in range(2):
        torch.testing.assert_close(module.verified_spatial_forward(conv, x), conv(x), atol=0, rtol=0)
    assert len(calls) == 1
