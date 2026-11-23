# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from cache_dit import ForwardPattern
from torch import nn

from vllm_omni.diffusion.cache.cachedit import CacheDiTAdapterConfig, CacheDiTBackend
from vllm_omni.diffusion.models.hidream_o1_image.hidream_o1_image_transformer import (
    HiDreamO1TextModel,
)
from vllm_omni.diffusion.models.hidream_o1_image.pipeline_hidream_o1_image import (
    HiDreamO1ImagePipeline,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _empty_text_model() -> HiDreamO1TextModel:
    model = HiDreamO1TextModel.__new__(HiDreamO1TextModel)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([nn.Identity()])
    return model


def test_cache_dit_targets_the_decoder_module_that_owns_layers():
    text_model = _empty_text_model()
    pipeline = SimpleNamespace(
        _dit_modules=HiDreamO1ImagePipeline._dit_modules,
        model=SimpleNamespace(
            model=SimpleNamespace(language_model=text_model),
        ),
    )

    with (
        patch("vllm_omni.diffusion.cache.cachedit.backend.BlockAdapter") as block_adapter_cls,
        patch("vllm_omni.diffusion.cache.cachedit.backend.cache_dit") as cache_dit,
    ):
        backend = CacheDiTBackend()
        backend.enable(pipeline)
        backend.refresh(pipeline, num_inference_steps=50)

    adapter_config = HiDreamO1TextModel._cache_dit_adapter_config
    assert isinstance(adapter_config, CacheDiTAdapterConfig)
    assert adapter_config.block_forward_patterns == {"layers": ForwardPattern.Pattern_3}
    assert adapter_config.has_separate_cfg is True

    block_adapter_cls.assert_called_once_with(
        transformer=text_model,
        blocks=[text_model.layers],
        forward_pattern=[ForwardPattern.Pattern_3],
        has_separate_cfg=True,
        check_forward_pattern=True,
    )
    cache_dit.enable_cache.assert_called_once()
    enable_call = cache_dit.enable_cache.call_args
    assert enable_call.args == (block_adapter_cls.return_value,)
    assert enable_call.kwargs["cache_config"].Fn_compute_blocks == 1
    assert enable_call.kwargs["calibrator_config"] is None
    cache_dit.refresh_context.assert_called_once_with(
        text_model,
        num_inference_steps=50,
        verbose=True,
    )
