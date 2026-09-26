# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU regression tests for MammothModa2 DiT quantization configuration wiring."""

import pytest
import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.models.mammoth_moda2 import pipeline_mammothmoda2_dit as pipeline_module
from vllm_omni.diffusion.models.mammoth_moda2.mammothmoda2_dit_model import Transformer2DModel
from vllm_omni.transformers_utils.configs.mammoth_moda2 import Mammothmoda2Config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion, pytest.mark.usefixtures("mock_tp1")]


class RecordingQuantConfig(QuantizationConfig):
    def __init__(self):
        super().__init__()
        self.prefixes: list[str] = []

    @classmethod
    def get_name(cls):
        return "recording"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.float32]

    @classmethod
    def get_min_capability(cls):
        return 0

    @classmethod
    def get_config_filenames(cls):
        return []

    @classmethod
    def from_config(cls, config):
        return cls()

    def get_quant_method(self, layer, prefix):
        self.prefixes.append(prefix)
        # Exercise real layer construction without requiring an FP8-capable GPU.
        return UnquantizedLinearMethod()


@pytest.fixture
def tiny_dit_config():
    return {
        "hidden_size": 48,
        "num_attention_heads": 6,
        "num_kv_heads": 2,
        "num_layers": 2,
        "num_refiner_layers": 2,
        "in_channels": 4,
        "multiple_of": 8,
        "ffn_dim_multiplier": 1.0,
        "axes_dim_rope": (2, 2, 4),
        "axes_lens": (8, 8, 8),
        "text_feat_dim": 16,
    }


@pytest.fixture(autouse=True)
def sdpa_config():
    with set_current_diffusion_config(
        OmniDiffusionConfig(diffusion_attention_config={"default": {"backend": "TORCH_SDPA"}})
    ):
        yield


def _assert_quantized_projections(model, quant_config, prefix):
    # Explicit targets catch a projection replaced by nn.Linear, a missing
    # refiner branch, and incorrect prefixes on blocks after index zero.
    expected_names = {
        f"{branch}.{index}.{projection}"
        for branch in ("noise_refiner", "ref_image_refiner", "context_refiner", "layers")
        for index in range(2)
        for projection in (
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "feed_forward.gate_up_proj",
            "feed_forward.linear_2",
        )
    }
    linears = {name: module for name, module in model.named_modules() if isinstance(module, LinearBase)}
    assert set(linears) == expected_names
    expected_prefixes = {f"{prefix}.{name}" if prefix else name for name in expected_names}
    assert set(quant_config.prefixes) == expected_prefixes
    assert len(quant_config.prefixes) == len(expected_prefixes)
    for name, module in linears.items():
        assert module.quant_config is quant_config, name
        assert module.prefix == (f"{prefix}.{name}" if prefix else name), name


@pytest.mark.parametrize("prefix", ["", "gen_transformer"])
def test_transformer_propagates_quant_config_and_prefix(tiny_dit_config, prefix):
    quant_config = RecordingQuantConfig()
    model = Transformer2DModel.from_config(tiny_dit_config, quant_config=quant_config, prefix=prefix)
    _assert_quantized_projections(model, quant_config, prefix)


@pytest.mark.parametrize("prefix", ["", "stage1"])
def test_pipeline_propagates_quant_config_to_all_dit_projections(monkeypatch, tiny_dit_config, prefix):
    quant_config = RecordingQuantConfig()
    hf_config = Mammothmoda2Config(
        llm_config={
            "model_type": "mammothmoda2_qwen2_5_vl",
            "text_config": {"hidden_size": 16},
        },
        gen_vae_config={},
        gen_dit_config=tiny_dit_config,
        gen_axes_dim_rope=[2, 2, 4],
        gen_axes_lens=[8, 8, 8],
    )
    od_config = OmniDiffusionConfig(
        model="/models/MammothModa2-Preview",
        model_class_name="MammothModa2DiTPipeline",
        tf_model_config=TransformerConfig.from_dict(hf_config.to_dict()),
        quantization_config=quant_config,
    )
    # Only the unrelated VAE is stubbed; keep the complete pipeline ->
    # transformer -> block -> linear constructor chain real.
    monkeypatch.setattr(pipeline_module.AutoencoderKL, "from_config", lambda config: torch.nn.Identity())
    pipeline = pipeline_module.MammothModa2DiTPipeline(od_config=od_config, prefix=prefix)
    transformer_prefix = f"{prefix}.gen_transformer" if prefix else "gen_transformer"
    _assert_quantized_projections(pipeline.gen_transformer, quant_config, transformer_prefix)
