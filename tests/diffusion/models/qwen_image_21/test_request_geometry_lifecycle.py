# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import contextlib
import gc
import weakref
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.qwen_image_21.cfg_parallel import QwenImage21CFGParallelMixin
from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
    QwenImage21KVCache,
    QwenImage21Transformer2DModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _Pipeline(QwenImage21CFGParallelMixin):
    def __init__(self, mode):
        self.mode = mode
        self.interrupt = False
        self.scheduler = Mock()
        self.transformer = QwenImage21Transformer2DModel.__new__(QwenImage21Transformer2DModel)
        nn.Module.__init__(self.transformer)
        self.transformer.causal_condition = True
        self.transformer.transformer_blocks = [nn.Identity()]
        self.transformer.sequence_prepare = nn.Identity()
        self.transformer._decode_graph_manager = Mock()
        self.prefix_refs = []

    def progress_bar(self, total):
        return contextlib.nullcontext(Mock())

    def predict_noise_maybe_with_cfg(self, do_true_cfg, true_cfg_scale, positive_kwargs, negative_kwargs, **kwargs):
        cache = positive_kwargs["kv_cache"]
        prefix = torch.ones(1, 4, 8)
        self.prefix_refs.append(weakref.ref(prefix))
        cache[0]["cond"] = {"key": prefix, "value": prefix}
        geometry = {"positions": torch.arange(4)}
        self.transformer._request_geometry_owner = (cache, positive_kwargs["img_mask"], (), geometry)
        self.transformer.sequence_prepare._request_geometry = geometry
        if self.mode == "predict_error":
            raise RuntimeError("prediction failed")
        if self.mode == "interrupt":
            self.interrupt = True
        return torch.zeros_like(positive_kwargs["hidden_states"])

    def scheduler_step_maybe_with_cfg(self, noise_pred, timestep, latents, do_true_cfg):
        if self.mode == "scheduler_error":
            raise RuntimeError("scheduler failed")
        return latents


@pytest.mark.parametrize("mode", ["success", "interrupt", "predict_error", "scheduler_error"])
def test_denoising_releases_prefix_kv_and_geometry_without_clearing_graphs(mode):
    pipeline = _Pipeline(mode)
    inputs = dict(
        latents=torch.zeros(1, 4, 8),
        timesteps=torch.tensor([1000.0, 500.0]),
        prompt_embeds=torch.zeros(1, 2, 8),
        prompt_embeds_mask=torch.ones(1, 2),
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        negative_img_mask=None,
        image_latents=None,
        img_mask=torch.tensor([[False, False, True]]),
        img_shapes=[[(1, 2, 2)]],
        do_true_cfg=False,
        true_cfg_scale=1.0,
    )
    if mode.endswith("error"):
        with pytest.raises(RuntimeError, match="failed"):
            pipeline.diffuse(**inputs)
    else:
        pipeline.diffuse(**inputs)
    gc.collect()
    assert pipeline.prefix_refs
    assert all(ref() is None for ref in pipeline.prefix_refs)
    assert pipeline.transformer._request_geometry_owner is None
    assert pipeline.transformer.sequence_prepare._request_geometry is None
    pipeline.transformer._decode_graph_manager.clear.assert_not_called()


def test_dropped_step_request_releases_geometry_and_prefix_without_explicit_finalization():
    pipeline = _Pipeline("success")
    model = pipeline.transformer
    prefix = torch.ones(1, 4, 8)
    prefix_ref = weakref.ref(prefix)
    cache = QwenImage21KVCache([{"cond": {"key": prefix, "value": prefix}}])
    mask = torch.ones(1, 4, dtype=torch.bool)
    geometry = model._geometry_for_request(cache, mask, ())
    model.sequence_prepare._request_geometry = geometry
    assert geometry is model._geometry_for_request(cache, mask, ())
    del prefix, cache
    assert prefix_ref() is None
    assert model._request_geometry_owner is None
    assert model.sequence_prepare._request_geometry is None
    model._decode_graph_manager.clear.assert_not_called()


def test_caller_owned_plain_list_uses_uncached_geometry():
    model = _Pipeline("success").transformer
    assert model._geometry_for_request([{}], torch.ones(1, 4), ()) is None
