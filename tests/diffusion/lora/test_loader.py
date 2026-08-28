from collections import defaultdict

import pytest
import torch
import torch.nn as nn
from diffusers.loaders.lora_conversion_utils import (
    _convert_non_diffusers_qwen_lora_to_diffusers,
)
from pytest_mock import MockerFixture
from torch.testing import assert_close

from vllm_omni.diffusion.lora.loader import (
    LoraLoaderMixin,
    QwenImageLoraLoaderMixin,
    WanLoraLoaderMixin,
    _prepare_lora_delta,
    _remap_state_dict_keys,
)

HEAD_DIM = 24
RANK = 2
NUM_LAYERS = 10

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class DummyTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        bias: bool = False,
        stacked_params: bool = False,
    ):
        super().__init__()
        self.attn = nn.Module()
        if stacked_params:
            self.attn.to_qkv = nn.Linear(d_model * 3, d_model, bias=bias)
        else:
            self.attn.to_q = nn.Linear(d_model, d_model, bias=bias)
            self.attn.to_k = nn.Linear(d_model, d_model, bias=bias)
            self.attn.to_v = nn.Linear(d_model, d_model, bias=bias)
        self.attn.to_out = nn.Linear(d_model, d_model, bias=bias)


class DummyTransformer(nn.Module):
    def __init__(self, num_layers: int, d_model: int, bias: bool = False, stacked_params: bool = False):
        super().__init__()

        self.blocks = nn.ModuleList(
            [DummyTransformerBlock(d_model, bias=bias, stacked_params=stacked_params) for _ in range(num_layers)]
        )
        if stacked_params:
            self.stacked_params_mapping = [
                (".attn.to_qkv", ".attn.to_q", "q"),
                (".attn.to_qkv", ".attn.to_k", "k"),
                (".attn.to_qkv", ".attn.to_v", "v"),
            ]


class DummyPipeline(nn.Module, LoraLoaderMixin):
    def __init__(self, num_layers: int, d_model: int, bias: bool = False, stacked_params: bool = False):
        super().__init__()
        self.transformer = DummyTransformer(num_layers, d_model, bias=bias, stacked_params=stacked_params)


# ======================================================
# helper functions for test
# ======================================================
def _filter_parameters(named_params, mode: str = "all"):
    if mode == "all":
        return named_params

    filtered = []
    for name, param in named_params:
        if name.endswith(mode):
            filtered.append((name, param))

    return filtered


def assert_params_equal(module, original_params, mode: str = "all"):
    for name, param in _filter_parameters(module.named_parameters(), mode):
        assert_close(param, original_params[name])


def assert_params_not_equal(module, original_params, mode: str = "all"):
    for name, param in _filter_parameters(module.named_parameters(), mode):
        assert not torch.allclose(param, original_params[name])


def make_dummy_lora_state_dict(
    base_key,
    weight_a: torch.Tensor | None = None,
    weight_b: torch.Tensor | None = None,
    weight_bias: torch.Tensor | None = None,
    bias: bool = False,
    lora_a_suffix: str = "lora_A.weight",
    lora_b_suffix: str = "lora_B.weight",
    lora_bias_suffix: str = "bias",
):
    """
    Create a dummy LoRA state dict.

    Returns:
        tuple: (state_dict, expected_delta)
    """
    key_a = f"{base_key}.{lora_a_suffix}"
    key_b = f"{base_key}.{lora_b_suffix}"
    key_bias = f"{base_key}.{lora_bias_suffix}" if bias else None

    if weight_a is None:
        weight_a = torch.randn(RANK, HEAD_DIM)
    if weight_b is None:
        weight_b = torch.randn(HEAD_DIM, RANK)

    state_dict = {}
    expected_delta = None

    if bias:
        state_dict[key_a] = weight_a
        state_dict[key_b] = weight_b
        weight_bias = torch.randn(HEAD_DIM)
        state_dict[key_bias] = weight_bias
        expected_delta = weight_bias
    else:
        state_dict[key_a] = weight_a
        state_dict[key_b] = weight_b
        expected_delta = weight_b @ weight_a

    return state_dict, expected_delta


def make_lora_state_dict_for_module(
    module,
    lora_rank=RANK,
    bias=False,
    lora_a_suffix: str = "lora_A.weight",
    lora_b_suffix: str = "lora_B.weight",
    lora_bias_suffix: str = "bias",
    prefix: str = "transformer",
    remap_proj_out: bool = True,
):
    param_to_weight_names = defaultdict(list)
    if hasattr(module, "stacked_params_mapping"):
        for param_name, weight_name, _ in module.stacked_params_mapping:
            param_to_weight_names[param_name].append(weight_name)

    state_dict = {}
    for name, param in module.named_parameters(prefix=prefix):
        if name.endswith(".weight"):
            base_key = name[: -len(".weight")]
            if remap_proj_out and base_key.endswith(".to_out"):
                base_key = base_key.replace(".to_out", ".to_out.0")

            d_in, d_out = param.shape
            is_stacked_param = False
            for param_name, weight_names in param_to_weight_names.items():
                if param_name not in name:
                    continue
                is_stacked_param = True
                target_in = d_in // len(weight_names)
                for weight_name in weight_names:
                    lora_key = base_key.replace(param_name, weight_name)
                    weight_a = torch.randn(lora_rank, d_out)
                    weight_b = torch.randn(target_in, lora_rank)
                    state_dict[f"{lora_key}.{lora_a_suffix}"] = weight_a
                    state_dict[f"{lora_key}.{lora_b_suffix}"] = weight_b

            if is_stacked_param:
                continue

            weight_a = torch.randn(lora_rank, d_out)
            weight_b = torch.randn(d_in, lora_rank)
            key_a = f"{base_key}.{lora_a_suffix}"
            key_b = f"{base_key}.{lora_b_suffix}"
            state_dict[key_a] = weight_a
            state_dict[key_b] = weight_b
        elif name.endswith(".bias") and bias:
            base_key = name[: -len(".bias")]
            if remap_proj_out and base_key.endswith(".to_out"):
                base_key = base_key.replace(".to_out", ".to_out.0")

            d_bias = param.shape[0]
            is_stacked_param = False
            for param_name, weight_names in param_to_weight_names.items():
                if param_name not in name:
                    continue
                is_stacked_param = True
                target_bias = d_bias // len(weight_names)
                for weight_name in weight_names:
                    lora_key = base_key.replace(param_name, weight_name)
                    weight_bias = torch.randn(target_bias)
                    state_dict[f"{lora_key}.{lora_bias_suffix}"] = weight_bias

            if is_stacked_param:
                continue

            key_bias = f"{base_key}.{lora_bias_suffix}"
            state_dict[key_bias] = torch.randn_like(param)

    return state_dict


# ======================================================
# Test loader helper functions
# ======================================================
def test_prepare_lora_delta_with_stacked_params():
    base_key = "transformer.attn.to_qkv"

    sd0, delta0 = make_dummy_lora_state_dict("transformer.attn.to_q")
    sd1, delta1 = make_dummy_lora_state_dict("transformer.attn.to_k")
    sd2, delta2 = make_dummy_lora_state_dict("transformer.attn.to_v")

    lora_state_dict = {}
    lora_state_dict.update(sd0)
    lora_state_dict.update(sd1)
    lora_state_dict.update(sd2)
    expected_delta = torch.concat([delta0, delta1, delta2])

    stacked_mapping = {".attn.to_qkv": [".attn.to_q", ".attn.to_k", ".attn.to_v"]}

    actual_delta, used_keys = _prepare_lora_delta(
        lora_state_dict,
        base_key,
        stacked_mapping,
    )

    # validate the delta
    assert actual_delta is not None
    assert_close(actual_delta, expected_delta)

    # validate the used keys
    key_q_a = "transformer.attn.to_q.lora_A.weight"
    key_q_b = "transformer.attn.to_q.lora_B.weight"
    key_k_a = "transformer.attn.to_k.lora_A.weight"
    key_k_b = "transformer.attn.to_k.lora_B.weight"
    key_v_a = "transformer.attn.to_v.lora_A.weight"
    key_v_b = "transformer.attn.to_v.lora_B.weight"

    for key in [key_q_a, key_q_b, key_k_a, key_k_b, key_v_a, key_v_b]:
        assert key in used_keys


def test_remap_state_dict_keys_matched_two():
    lora_state_dict, _ = make_dummy_lora_state_dict(
        "transformer.attn.to_out.0", bias=True, lora_bias_suffix="lora_B.bias"
    )

    lora_state_dict = _remap_state_dict_keys(
        lora_state_dict,
        [(".to_out.0.", ".to_out."), (".lora_B.bias", ".bias")],
    )

    assert len(lora_state_dict) == 3
    assert "transformer.attn.to_out.bias" in lora_state_dict


# ======================================================
# Test LoraLoaderMixin
# ======================================================
class TestLoraLoaderMixin:
    @pytest.mark.parametrize(
        "bias, stacked_params",
        [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ],
        ids=[
            "basic",
            "bias",
            "stacked_params",
            "bias_stacked_params",
        ],
    )
    def test_load_lora_into_module(self, bias: bool, stacked_params: bool):
        pipeline = DummyPipeline(NUM_LAYERS, HEAD_DIM, bias=bias, stacked_params=stacked_params)
        original_params = {name: param.clone() for name, param in pipeline.transformer.named_parameters()}

        lora_state_dict = make_lora_state_dict_for_module(pipeline.transformer, bias=bias)
        assert len(lora_state_dict) > 0

        # remap the keys to vllm-omni format
        lora_state_dict = _remap_state_dict_keys(lora_state_dict, [(".to_out.0.", ".to_out.")])

        used_keys = pipeline.load_lora_into_module(lora_state_dict, pipeline.transformer)
        # validate the parameters are updated
        assert_params_not_equal(pipeline.transformer, original_params)

        # validate all keys are used
        missing_keys = set(lora_state_dict.keys()) - used_keys
        assert len(missing_keys) == 0

    def test_unload_module_lora(self):
        pipeline = DummyPipeline(NUM_LAYERS, HEAD_DIM)
        original_params = {name: param.clone() for name, param in pipeline.transformer.named_parameters()}

        lora_state_dict = make_lora_state_dict_for_module(pipeline.transformer)
        # validate the state dict is valid
        assert len(lora_state_dict) > 0

        # remap the keys to vllm-omni format
        lora_state_dict = _remap_state_dict_keys(lora_state_dict, [(".to_out.0.", ".to_out.")])

        used_keys = pipeline.load_lora_into_module(lora_state_dict, pipeline.transformer)

        # validate the weights are updated
        assert_params_not_equal(pipeline.transformer, original_params)

        # validate all keys are used
        missing_keys = set(lora_state_dict.keys()) - used_keys
        assert len(missing_keys) == 0

        unload_used_keys = pipeline.unload_module_lora(lora_state_dict, pipeline.transformer)
        missing_keys = set(lora_state_dict.keys()) - unload_used_keys
        assert len(missing_keys) == 0
        assert len(unload_used_keys - used_keys) == 0

        # validate the weights are restored
        assert_params_equal(pipeline.transformer, original_params)


# ======================================================
# Test QwenImageLoraLoaderMixin
# ======================================================
class DummyQwenImagePipeline(nn.Module, QwenImageLoraLoaderMixin):
    def __init__(self, num_layers: int, d_model: int, bias: bool = False):
        super().__init__()
        self.transformer = DummyTransformer(num_layers, d_model, bias=bias, stacked_params=True)


class TestQwenImageLoraLoaderMixin:
    def test_load_lora_weights(self, mocker: MockerFixture):
        pipeline = DummyQwenImagePipeline(NUM_LAYERS, HEAD_DIM)
        original_params = {name: param.clone() for name, param in pipeline.transformer.named_parameters()}

        lora_state_dict = make_lora_state_dict_for_module(pipeline.transformer)
        assert len(lora_state_dict) > 0

        mocker.patch(
            "vllm_omni.diffusion.lora.loader.get_converter_by_pipeline",
            return_value=_convert_non_diffusers_qwen_lora_to_diffusers,
        )
        pipeline.load_lora_weights(lora_state_dict, "adapter0")

        # validate the weights are updated after lora loaded
        assert_params_not_equal(pipeline.transformer, original_params)

        # validate lora_loaded map is updated after lora loaded
        assert "adapter0" in pipeline.lora_loaded

    def test_unload_lora_weights(self, mocker: MockerFixture):
        pipeline = DummyQwenImagePipeline(NUM_LAYERS, HEAD_DIM)
        original_params = {name: param.clone() for name, param in pipeline.transformer.named_parameters()}

        lora_state_dict = make_lora_state_dict_for_module(pipeline.transformer)
        assert len(lora_state_dict) > 0

        mocker.patch(
            "vllm_omni.diffusion.lora.loader.get_converter_by_pipeline",
            return_value=_convert_non_diffusers_qwen_lora_to_diffusers,
        )
        pipeline.load_lora_weights(lora_state_dict, "adapter0")
        # validate the weights are updated after lora loaded
        assert_params_not_equal(pipeline.transformer, original_params)

        # validate lora_loaded map is updated after lora loaded
        assert "adapter0" in pipeline.lora_loaded

        pipeline.unload_lora_weights("adapter0")
        # validate the weights are restored after lora unloaded
        assert_params_equal(pipeline.transformer, original_params)

        # validate lora_loaded map is updated
        assert "adapter0" not in pipeline.lora_loaded

    def test_load_diffusers_lora_applies_non_unit_alpha(self, mocker: MockerFixture):
        pipeline = DummyQwenImagePipeline(1, HEAD_DIM)
        to_out = pipeline.transformer.blocks[0].attn.to_out
        original_weight = to_out.weight.detach().clone()
        base_key = "blocks.0.attn.to_out.0"
        lora_a = torch.randn(RANK, HEAD_DIM)
        lora_b = torch.randn(HEAD_DIM, RANK)
        alpha = torch.tensor(6.0)
        state_dict = {
            f"{base_key}.lora_A.weight": lora_a,
            f"{base_key}.lora_B.weight": lora_b,
            f"{base_key}.alpha": alpha,
        }
        mocker.patch(
            "vllm_omni.diffusion.lora.loader.get_converter_by_pipeline",
            return_value=_convert_non_diffusers_qwen_lora_to_diffusers,
        )

        pipeline.load_lora_weights(state_dict, "adapter0")

        expected_weight = original_weight + (lora_b @ lora_a) * (alpha.item() / RANK)
        assert_close(to_out.weight, expected_weight)

        pipeline.unload_lora_weights("adapter0")
        assert_close(to_out.weight, original_weight)


# ======================================================
# Test WanLoraLoaderMixin
# ======================================================
class DummyWanPipeline(nn.Module, WanLoraLoaderMixin):
    def __init__(self, num_layers: int, d_model: int):
        super().__init__()
        self.has_transformer_2 = True
        self.transformer = DummyTransformer(num_layers, d_model, stacked_params=True)
        self.transformer_2 = DummyTransformer(num_layers, d_model, stacked_params=True)


def test_wan_load_lora_weights_targets_both_transformers(mocker: MockerFixture):
    pipeline = DummyWanPipeline(NUM_LAYERS, HEAD_DIM)
    original_high = {name: param.clone() for name, param in pipeline.transformer.named_parameters()}
    original_low = {name: param.clone() for name, param in pipeline.transformer_2.named_parameters()}
    high_state_dict = make_lora_state_dict_for_module(pipeline.transformer)
    low_state_dict = make_lora_state_dict_for_module(pipeline.transformer_2)

    mocker.patch(
        "vllm_omni.diffusion.lora.loader._load_lora_state_dict",
        side_effect=[high_state_dict, low_state_dict],
    )

    pipeline.load_lora_weights(["high.safetensors", "low.safetensors"], "adapter0")

    assert_params_not_equal(pipeline.transformer, original_high)
    assert_params_not_equal(pipeline.transformer_2, original_low)
    assert set(pipeline.lora_loaded["adapter0"]) == {"transformer", "transformer_2"}

    pipeline.unload_lora_weights("adapter0")

    assert_params_equal(pipeline.transformer, original_high)
    assert_params_equal(pipeline.transformer_2, original_low)
    assert "adapter0" not in pipeline.lora_loaded
