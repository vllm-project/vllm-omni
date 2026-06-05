# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for SenseNova-U1 gen-only FP8 quantization.

Validates that the ``_GenOnlyQuantConfig`` wrapper correctly routes FP8
quantization to only the gen-path (``*_mot_gen``) layers while leaving
understanding-path layers in BF16.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.distributed import parallel_state
from vllm.model_executor.layers.linear import LinearBase

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _fake_tp_group():
    """Provide a fake TP group so vLLM parallel layers can be instantiated.

    Note: a shared ``init_fake_tp_group`` fixture exists in
    ``tests/helpers/fixtures/env.py`` but requires pytest-mock.
    """
    mock_tp = MagicMock()
    mock_tp.world_size = 1
    mock_tp.rank_in_group = 0
    old = parallel_state._TP
    parallel_state._TP = mock_tp
    yield
    parallel_state._TP = old


def _make_minimal_llm_config(num_layers: int = 2) -> SimpleNamespace:
    """Create a minimal SenseNova-U1 LLM config for unit testing."""
    return SimpleNamespace(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=64,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        vocab_size=1024,
        rope_theta=1000000.0,
        rope_theta_hw=10000.0,
        max_position_embeddings=4096,
        max_position_embeddings_hw=1024,
        attention_bias=True,
        layer_types=["full_attention"] * num_layers,
        tie_word_embeddings=False,
    )


def _build_model_with_gen_only_quant(quant_method: str | None = "fp8"):
    """Construct SenseNovaU1ForCausalLM with gen-only FP8 wrapping."""
    from unittest.mock import MagicMock, patch

    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
        _GenOnlyQuantConfig,
    )
    from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import (
        SenseNovaU1ForCausalLM,
    )
    from vllm_omni.quantization import build_quant_config

    inner_config = build_quant_config(quant_method)
    if inner_config is not None:
        quant_config = _GenOnlyQuantConfig(inner_config)
    else:
        quant_config = None

    llm_cfg = _make_minimal_llm_config(num_layers=2)

    mock_vllm_config = MagicMock()
    mock_vllm_config.model_config.dtype = torch.bfloat16
    mock_vllm_config.model_config.quantization = quant_method

    with (
        torch.device("meta"),
        patch("vllm.model_executor.layers.quantization.fp8.get_current_vllm_config", return_value=mock_vllm_config),
    ):
        model = SenseNovaU1ForCausalLM(
            llm_cfg,
            quant_config=quant_config,
            prefix="language_model",
        )
    return model


def test_mot_gen_layers_have_fp8_quant_method():
    """Gen-path (mot_gen) Linear layers should have a real FP8 quant_method."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    model = _build_model_with_gen_only_quant("fp8")

    mot_gen_linears = [
        (name, mod) for name, mod in model.named_modules() if isinstance(mod, LinearBase) and "mot_gen" in name
    ]

    assert len(mot_gen_linears) > 0, "No mot_gen LinearBase modules found"

    for name, mod in mot_gen_linears:
        assert mod.quant_method is not None, f"mot_gen layer {name!r} should have quant_method with FP8"
        assert not isinstance(mod.quant_method, UnquantizedLinearMethod), (
            f"mot_gen layer {name!r} should NOT use UnquantizedLinearMethod"
        )


def test_und_layers_have_no_quant_method():
    """Understanding-path Linear layers should use UnquantizedLinearMethod (stay BF16)."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    model = _build_model_with_gen_only_quant("fp8")

    und_linears = [
        (name, mod) for name, mod in model.named_modules() if isinstance(mod, LinearBase) and "mot_gen" not in name
    ]

    assert len(und_linears) > 0, "No und LinearBase modules found"

    for name, mod in und_linears:
        assert isinstance(mod.quant_method, UnquantizedLinearMethod), (
            f"und layer {name!r} should use UnquantizedLinearMethod, got {type(mod.quant_method).__name__}"
        )


def test_no_quant_method_without_quantization():
    """Without quantization, all layers should use UnquantizedLinearMethod."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    model = _build_model_with_gen_only_quant(None)

    for name, mod in model.named_modules():
        if isinstance(mod, LinearBase):
            assert isinstance(mod.quant_method, UnquantizedLinearMethod), (
                f"LinearBase module {name!r} has quant_method="
                f"{type(mod.quant_method).__name__} "
                "but no quantization was configured"
            )


def test_und_and_gen_layers_both_present():
    """Sanity check: the model has both und and gen layers."""
    model = _build_model_with_gen_only_quant(None)

    layer_names = [name for name, _ in model.named_modules()]

    und_qkv = [n for n in layer_names if "qkv_proj" in n and "mot_gen" not in n]
    gen_qkv = [n for n in layer_names if "qkv_proj_mot_gen" in n]
    und_mlp = [n for n in layer_names if ".mlp." in n and "mot_gen" not in n]
    gen_mlp = [n for n in layer_names if "mlp_mot_gen" in n]

    assert len(und_qkv) > 0, "No und qkv_proj layers found"
    assert len(gen_qkv) > 0, "No gen qkv_proj_mot_gen layers found"
    assert len(und_mlp) > 0, "No und mlp layers found"
    assert len(gen_mlp) > 0, "No gen mlp_mot_gen layers found"


def test_gen_only_wrapper_delegates_attributes():
    """_GenOnlyQuantConfig should delegate attribute access to the inner config."""
    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
        _GenOnlyQuantConfig,
    )
    from vllm_omni.quantization import build_quant_config

    inner = build_quant_config("fp8")
    wrapper = _GenOnlyQuantConfig(inner)

    assert wrapper.get_name() == "fp8"
    assert wrapper.activation_scheme == inner.activation_scheme
