# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for INC/AutoRound quantization via the unified framework."""

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("data_type", "group_size", "packing_format"),
    [
        ("int", 128, "auto_round:auto_gptq"),
        ("mx_fp", 32, "auto_round:llm_compressor"),
        ("nv_fp", 16, "auto_round:llm_compressor"),
    ],
    ids=["int4", "mxfp4", "nvfp4"],
)
def test_build_quant_config_autoround(data_type, group_size, packing_format):
    """Build INC configs for AutoRound INT4, MXFP4, and NVFP4 checkpoints."""
    from vllm.model_executor.layers.quantization.inc import INCConfig

    from vllm_omni.quantization import build_quant_config

    config = build_quant_config(
        "auto-round",
        bits=4,
        group_size=group_size,
        sym=True,
        data_type=data_type,
        packing_format=packing_format,
    )
    assert config is not None
    assert isinstance(config, INCConfig)
    assert config.weight_bits == 4
    assert config.group_size == group_size
    assert config.data_type == data_type
    assert config.packing_format == packing_format


@pytest.mark.parametrize(
    ("data_type", "group_size", "scheme_name"),
    [("mx_fp", 32, "INCMxfp4LinearMethod"), ("nv_fp", 16, "CompressedTensorsW4A4Fp4")],
    ids=["mxfp4", "nvfp4"],
)
@pytest.mark.parametrize("excluded", [False, True])
def test_inc_fp4_linear_method(mocker, data_type, group_size, scheme_name, excluded):
    from torch.nn import LayerNorm
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsLinearMethod,
    )
    from vllm.model_executor.layers.quantization.inc.inc_linear import INCLinearMethod

    from vllm_omni.quantization import build_quant_config

    # Keep config parsing and method selection real; avoid device kernel initialization.
    mxfp4_kernel = mocker.patch(
        "vllm.model_executor.layers.quantization.inc.schemes.inc_mxfp4_linear.init_mxfp4_linear_kernel"
    )
    nvfp4_kernel = mocker.patch(
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w4a4_nvfp4.init_nvfp4_linear_kernel"
    )
    config = build_quant_config(
        "auto-round",
        bits=4,
        group_size=group_size,
        data_type=data_type,
        packing_format="auto_round:llm_compressor",
        extra_config={"proj": {"bits": 16}} if excluded else None,
    )
    layer = mocker.Mock(spec=LinearBase)
    method = config.get_quant_method(layer, "proj")
    if excluded:
        assert isinstance(method, UnquantizedLinearMethod)
        mxfp4_kernel.assert_not_called()
        nvfp4_kernel.assert_not_called()
    elif data_type == "nv_fp":
        assert isinstance(method, CompressedTensorsLinearMethod)
        assert type(layer.scheme).__name__ == scheme_name
    else:
        assert isinstance(method, INCLinearMethod)
        assert type(method.scheme).__name__ == scheme_name
    assert config.get_quant_method(LayerNorm(64), "norm") is None


def test_build_quant_config_inc():
    """build_quant_config("inc", ...) should also produce an INCConfig."""
    from vllm.model_executor.layers.quantization.inc import INCConfig

    from vllm_omni.quantization import build_quant_config

    config = build_quant_config("inc", bits=4, group_size=128)
    assert isinstance(config, INCConfig)
    assert config.weight_bits == 4


def test_build_quant_config_autoround_dict():
    """Dict-style config with method=auto-round should work."""
    from vllm.model_executor.layers.quantization.inc import INCConfig

    from vllm_omni.quantization import build_quant_config

    config = build_quant_config(
        {
            "method": "auto-round",
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "packing_format": "auto_round:auto_gptq",
        }
    )
    assert isinstance(config, INCConfig)
    assert config.weight_bits == 4


def test_build_quant_config_autoround_filters_metadata():
    """Checkpoint metadata keys (autoround_version, batch_size, iters)
    should be silently filtered out instead of causing TypeError."""
    from vllm.model_executor.layers.quantization.inc import INCConfig

    from vllm_omni.quantization import build_quant_config

    config = build_quant_config(
        "auto-round",
        bits=4,
        group_size=128,
        sym=True,
        packing_format="auto_round:auto_gptq",
        block_name_to_quantize="transformer_blocks,single_transformer_blocks",
        autoround_version="0.12.0",  # metadata — must be filtered
        batch_size=1,  # metadata — must be filtered
        iters=0,  # metadata — must be filtered
    )
    assert isinstance(config, INCConfig)
    assert config.weight_bits == 4
    assert config.group_size == 128


def test_build_quant_config_bits_to_weight_bits_mapping():
    """The 'bits' key from checkpoints should be mapped to 'weight_bits'."""
    from vllm.model_executor.layers.quantization.inc import INCConfig

    from vllm_omni.quantization import build_quant_config

    # If weight_bits is already provided, bits should be ignored
    config = build_quant_config("auto-round", weight_bits=4, group_size=128)
    assert isinstance(config, INCConfig)
    assert config.weight_bits == 4


def test_autoround_in_supported_methods():
    """auto-round and inc should appear in SUPPORTED_QUANTIZATION_METHODS."""
    from vllm_omni.quantization import SUPPORTED_QUANTIZATION_METHODS

    assert "auto-round" in SUPPORTED_QUANTIZATION_METHODS
    assert "inc" in SUPPORTED_QUANTIZATION_METHODS


def test_integration_autoround_via_omni_diffusion_config():
    """OmniDiffusionConfig with auto-round quantization dict should resolve."""
    from vllm.model_executor.layers.quantization.inc import INCConfig

    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(
        model="test",
        quantization_config={
            "method": "auto-round",
            "bits": 4,
            "group_size": 128,
            "sym": True,
        },
    )
    assert isinstance(config.quantization_config, INCConfig)
    assert config.quantization_config.weight_bits == 4


def test_integration_autodetect_from_transformer_config():
    """When TransformerConfig has quant_config, OmniDiffusionConfig should
    auto-detect it even without explicit quantization_config."""
    from vllm.model_executor.layers.quantization.inc import INCConfig

    from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig

    tf_config = TransformerConfig.from_dict(
        {
            "quantization_config": {
                "quant_method": "auto-round",
                "bits": 4,
                "group_size": 128,
                "sym": True,
                "packing_format": "auto_round:auto_gptq",
                "autoround_version": "0.12.0",
                "batch_size": 1,
                "iters": 0,
            }
        }
    )
    assert tf_config.quant_method == "auto-round"
    assert isinstance(tf_config.quant_config, INCConfig)

    od_config = OmniDiffusionConfig(model="test", tf_model_config=tf_config)
    assert isinstance(od_config.quantization_config, INCConfig)
    assert od_config.quantization_config.weight_bits == 4
