# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_tokenizer import TokenizerEncodeOutput
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import HunyuanImage3Text2ImagePipeline

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@dataclass
class _Config:
    cfg_distilled: bool = False


@dataclass
class _Layer:
    layer_idx: int


@dataclass
class _Decoder:
    layers: list[_Layer]


@dataclass
class _Model:
    config: _Config
    model: _Decoder
    inject_ar_kv_into_layers: Mock


def _pipeline_and_inputs(*, guidance_scale=1.0, negative_boundary=None, image_slices=None):
    pipe = object.__new__(HunyuanImage3Text2ImagePipeline)
    pipe._guidance_scale = guidance_scale
    pipe.model = _Model(
        config=_Config(cfg_distilled=False),
        model=_Decoder(layers=[_Layer(layer_idx=0), _Layer(layer_idx=1)]),
        inject_ar_kv_into_layers=Mock(),
    )
    input_ids = torch.arange(12).reshape(1, 12)
    tokenizer_output = TokenizerEncodeOutput(
        think_recaption_end_pos=[[8]],
        uncond_cfg_start_pos=[[None], [negative_boundary]] if negative_boundary is not None else [[None]],
        joint_image_slices=[image_slices or []],
    )
    kwargs = dict(
        tokenizer_output=tokenizer_output,
        query_lens=[12],
        attention_mask=torch.ones(1, 1, 12, 12),
        position_ids=input_ids.clone(),
        image_mask=torch.zeros_like(input_ids, dtype=torch.bool),
        gen_timestep_scatter_index=torch.tensor([[9]]),
    )
    return pipe, input_ids, kwargs


def _cache(*lengths):
    return {
        index: {"key": torch.zeros(key_len, 1, 2), "value": torch.zeros(value_len, 1, 2)}
        for index, (key_len, value_len) in enumerate(lengths)
    }


@pytest.mark.parametrize(
    ("lengths", "expected"),
    [([(8, 8), (8, 8)], 8), ([(6, 6), (4, 5)], 4), ([(6, 6), (5, 4)], 4), ([(9, 9), (9, 9)], 8)],
)
def test_dense_ar_reuse_respects_every_layers_actual_key_and_value_coverage(lengths, expected):
    pipe, input_ids, kwargs = _pipeline_and_inputs()
    cache = _cache(*lengths)
    kwargs["ar_kv_data"] = cache

    output, reuse_len = pipe._maybe_handle_ar_kv_reuse(input_ids, kwargs, 1, False, None, torch.device("cpu"))

    assert reuse_len == expected
    torch.testing.assert_close(output, input_ids[:, expected:])
    assert kwargs["query_lens"] == [12 - expected]
    torch.testing.assert_close(kwargs["position_ids"], input_ids[:, expected:])
    torch.testing.assert_close(kwargs["gen_timestep_scatter_index"], torch.tensor([[9 - expected]]))
    pipe.model.inject_ar_kv_into_layers.assert_called_once_with(cache, expected)


@pytest.mark.parametrize("missing", ["layer", "key", "value"])
def test_dense_ar_reuse_rejects_missing_required_layer_cache_before_injection(missing):
    pipe, input_ids, kwargs = _pipeline_and_inputs()
    cache = _cache((8, 8), (8, 8))
    if missing == "layer":
        del cache[1]
    else:
        del cache[1][missing]
    kwargs["ar_kv_data"] = cache

    with pytest.raises(ValueError, match="missing key/value tensors for layer 1"):
        pipe._maybe_handle_ar_kv_reuse(input_ids, kwargs, 1, False, None, torch.device("cpu"))

    assert kwargs["query_lens"] == [12]
    pipe.model.inject_ar_kv_into_layers.assert_not_called()


@pytest.mark.parametrize(
    ("pipeline_kwargs", "lengths", "message"),
    [
        ({}, [(0, 0), (8, 8)], "computed prefix"),
        ({"image_slices": [slice(2, 6)]}, [(8, 8), (5, 5)], "conditional images"),
        ({"guidance_scale": 5.0, "negative_boundary": 5}, [(8, 8), (5, 5)], "negative CFG"),
    ],
)
def test_dense_ar_reuse_rejects_coverage_that_cannot_rebuild_the_remaining_prefix(pipeline_kwargs, lengths, message):
    pipe, input_ids, kwargs = _pipeline_and_inputs(**pipeline_kwargs)
    kwargs["ar_kv_data"] = _cache(*lengths)

    with pytest.raises(ValueError, match=message):
        pipe._maybe_handle_ar_kv_reuse(input_ids, kwargs, 1, False, None, torch.device("cpu"))

    assert kwargs["query_lens"] == [12]
    pipe.model.inject_ar_kv_into_layers.assert_not_called()
