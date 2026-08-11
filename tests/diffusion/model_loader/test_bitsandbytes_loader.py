# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm.config.load import LoadConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.bitsandbytes import BitsAndBytesConfig
from vllm.model_executor.model_loader.bitsandbytes_loader import (
    BitsAndBytesModelLoader,
)

from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import (
    QwenImagePipeline,
)
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenImageTransformer2DModel,
)
from vllm_omni.quantization import build_quant_config


class _DummyPipelineModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Linear(2, 2, bias=False)
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path="dummy",
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
            )
        ]

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            params[name].data.copy_(tensor)
            loaded.add(name)
        return loaded


def test_prequant_bitsandbytes_uses_vllm_state_loader(mocker):
    quant_config = BitsAndBytesConfig.from_config(
        {
            "quant_method": "bitsandbytes",
            "load_in_4bit": True,
            "load_in_8bit": False,
            "bnb_4bit_quant_type": "nf4",
        }
    )
    od_config = SimpleNamespace(
        dtype=torch.float32,
        model="dummy",
        revision=None,
        parallel_config=SimpleNamespace(
            use_hsdp=False,
            tensor_parallel_size=1,
        ),
        quantization_config=quant_config,
    )
    loader = DiffusersPipelineLoader(LoadConfig(), od_config)
    model = _DummyPipelineModel()
    quant_state = object()
    weights = iter([("weight", torch.zeros((2, 2)))])

    bnb_loader = mocker.patch(
        "vllm_omni.diffusion.model_loader.diffusers_loader.BitsAndBytesModelLoader",
        autospec=True,
    ).return_value
    bnb_loader.weight_mapper = lambda name: name

    def fake_quantized_generator(weight_files, use_safetensors, quant_states):
        assert weight_files == ["model.safetensors"]
        assert use_safetensors is True
        quant_states[bnb_loader.weight_mapper("weight")] = quant_state
        return weights

    bnb_loader._quantized_4bit_generator.side_effect = fake_quantized_generator
    bnb_loader._fuse_moe_quant_states.return_value = {}
    bnb_loader._stack_quantization_states.return_value = {
        "transformer.weight": {0: quant_state},
    }
    mocker.patch.object(
        loader,
        "_prepare_weights",
        return_value=("dummy/transformer", ["model.safetensors"], True),
    )

    loader.load_weights(model)

    bnb_loader._initialize_loader_state.assert_called_once_with(
        model,
        model_config=None,
    )
    bnb_loader._quantized_4bit_generator.assert_called_once_with(
        ["model.safetensors"],
        True,
        {"transformer.weight": quant_state},
    )
    assert bnb_loader.weight_mapper("weight") == "weight"
    bnb_loader._bind_quant_states_to_params.assert_called_once_with(
        model,
        {"transformer.weight": {0: quant_state}},
    )


def test_prequant_bitsandbytes_rejects_tensor_parallelism():
    quant_config = BitsAndBytesConfig.from_config(
        {
            "quant_method": "bitsandbytes",
            "load_in_4bit": True,
        }
    )
    od_config = SimpleNamespace(
        dtype=torch.float32,
        model="dummy",
        revision=None,
        parallel_config=SimpleNamespace(
            use_hsdp=False,
            tensor_parallel_size=2,
        ),
        quantization_config=quant_config,
    )
    loader = DiffusersPipelineLoader(LoadConfig(), od_config)

    with pytest.raises(
        ValueError,
        match="do not support tensor parallelism",
    ):
        loader.load_weights(_DummyPipelineModel())


def test_prequant_bitsandbytes_initializes_packed_linear_state(
    mocker,
    monkeypatch,
):
    bnb = types.ModuleType("bitsandbytes")
    bnb.__version__ = "0.48.1"
    bnb_nn = types.ModuleType("bitsandbytes.nn")
    bnb_nn.Int8Params = nn.Parameter
    bnb.nn = bnb_nn
    monkeypatch.setitem(sys.modules, "bitsandbytes", bnb)
    monkeypatch.setitem(sys.modules, "bitsandbytes.nn", bnb_nn)
    mocker.patch(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    mocker.patch(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
        return_value=0,
    )
    mocker.patch(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        return_value=0,
    )

    quant_config = BitsAndBytesConfig.from_config(
        {
            "quant_method": "bitsandbytes",
            "load_in_4bit": True,
        }
    )

    class _PackedTransformer(nn.Module):
        packed_modules_mapping = {"to_qkv": ["to_q", "to_k", "to_v"]}

        def __init__(self):
            super().__init__()
            self.to_qkv = ReplicatedLinear(
                input_size=4,
                output_size=12,
                bias=False,
                quant_config=quant_config,
                prefix="to_qkv",
                return_bias=False,
            )

    class _PackedPipeline(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = _PackedTransformer()

    bnb_loader = BitsAndBytesModelLoader(LoadConfig())
    bnb_loader.pre_quant = True

    bnb_loader._initialize_loader_state(_PackedPipeline(), model_config=None)

    assert "transformer.to_q" in bnb_loader.target_modules
    assert "transformer.to_k" in bnb_loader.target_modules
    assert "transformer.to_v" in bnb_loader.target_modules
    assert bnb_loader.modules_mapping.inverse_packed_mapping["to_q"] == (
        "to_qkv",
        0,
    )


def test_qwen_image_checkpoint_bnb_packs_unskipped_precision_sensitive_linears(
    mocker,
    monkeypatch,
):
    bnb = types.ModuleType("bitsandbytes")
    bnb.__version__ = "0.48.1"
    bnb_nn = types.ModuleType("bitsandbytes.nn")
    bnb_nn.Int8Params = nn.Parameter
    bnb.nn = bnb_nn
    monkeypatch.setitem(sys.modules, "bitsandbytes", bnb)
    monkeypatch.setitem(sys.modules, "bitsandbytes.nn", bnb_nn)
    mocker.patch(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    mocker.patch(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
        return_value=0,
    )
    mocker.patch(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        return_value=0,
    )
    mocker.patch(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        return_value=1,
    )

    quant_config = build_quant_config(
        {
            "quant_method": "bitsandbytes",
            "_load_in_4bit": True,
            "load_in_4bit": True,
            "llm_int8_skip_modules": [
                "transformer_blocks.0.img_mod.1",
                "norm_out.linear",
                "proj_out",
            ],
        }
    )
    od_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            sequence_parallel_size=1,
            tensor_parallel_size=1,
        )
    )

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        transformer = QwenImageTransformer2DModel(
            od_config=od_config,
            in_channels=4,
            out_channels=4,
            num_layers=2,
            attention_head_dim=4,
            num_attention_heads=2,
            joint_attention_dim=6,
            axes_dims_rope=(2, 2, 4),
            quant_config=quant_config,
        )
    finally:
        torch.set_default_dtype(previous_dtype)

    assert transformer.img_in.weight.use_bitsandbytes_4bit is True
    assert transformer.img_in.weight.pack_factor == 2
    assert transformer.img_in.weight.shape == (16, 1)
    assert transformer.txt_in.weight.use_bitsandbytes_4bit is True
    assert transformer.txt_in.weight.shape == (24, 1)
    assert not hasattr(
        transformer.transformer_blocks[0].img_mod[1].weight,
        "use_bitsandbytes_4bit",
    )
    assert transformer.transformer_blocks[1].img_mod[1].weight.use_bitsandbytes_4bit is True
    assert transformer.transformer_blocks[1].txt_mod[1].weight.use_bitsandbytes_4bit is True


def test_qwen_image_checkpoint_bnb_normalizes_quant_state_parameter_names():
    assert (
        QwenImagePipeline.hf_to_vllm_mapper._map_name("transformer_blocks.1.attn.to_out.0.weight")
        == "transformer_blocks.1.attn.to_out.weight"
    )
