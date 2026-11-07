# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineLinearMethod

from vllm_omni.diffusion.models.stable_audio import pipeline_stable_audio as pipeline
from vllm_omni.diffusion.models.stable_audio import stable_audio_transformer as transformer
from vllm_omni.quantization import build_quant_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _RecordingLinear(nn.Linear):
    """Record model wiring without allocating GPU quantization kernels."""

    def __init__(self, input_size, output_size, bias=False, quant_config=None, prefix=""):
        super().__init__(input_size, output_size, bias=bias, dtype=torch.bfloat16)
        self.quant_config = quant_config
        self.prefix = prefix


@pytest.mark.parametrize("quantized", [False, True])
def test_attention_only_quantization(monkeypatch, quantized):
    monkeypatch.setattr(transformer, "ReplicatedLinear", _RecordingLinear)
    monkeypatch.setattr(transformer, "Attention", lambda **kwargs: nn.Identity())
    ignored = "transformer_blocks.1.attn2.to_out.0"
    quant_config = build_quant_config({"method": "fp8", "ignored_layers": [ignored]}) if quantized else None
    model = transformer.StableAudioDiTModel(
        num_layers=2,
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        num_key_value_attention_heads=1,
        attention_head_dim=4,
        cross_attention_dim=8,
        cross_attention_input_dim=8,
        global_states_input_dim=8,
        time_proj_dim=8,
        quant_config=quant_config,
    )
    linears = {name: module for name, module in model.named_modules() if isinstance(module, _RecordingLinear)}
    expected = {
        f"transformer_blocks.{i}.{attention}.{projection}"
        for i in range(2)
        for attention in ("attn1", "attn2")
        for projection in ("to_q", "to_k", "to_v", "to_out.0")
    }
    assert linears.keys() == expected
    for name, module in linears.items():
        assert module.prefix == name
        assert module.quant_config is quant_config
        if quantized:
            layer = Mock(spec=LinearBase)
            if name == ignored:
                assert isinstance(quant_config.get_quant_method(layer, name), UnquantizedLinearMethod)
            else:
                resolved = quant_config.resolve_quant_method_cls(layer, name)
                assert resolved is not None
                assert resolved[-1] is Fp8PerTensorOnlineLinearMethod

    for block in model.transformer_blocks:
        assert type(block.ff.net[0].proj) is nn.Linear
        assert type(block.ff.net[2]) is nn.Linear
    for component in (model.timestep_proj, model.global_proj, model.cross_attention_proj):
        assert type(component[0]) is nn.Linear
        assert type(component[2]) is nn.Linear
    assert type(model.proj_in) is nn.Linear
    assert type(model.proj_out) is nn.Linear


@pytest.mark.parametrize("scope", ["global", "transformer", "text_encoder", "excluded", "none"])
def test_pipeline_resolves_transformer_quantization(monkeypatch, scope):
    quantization = {
        "global": "fp8",
        "transformer": {"transformer": {"method": "fp8"}},
        "text_encoder": {"text_encoder": {"method": "fp8"}},
        "excluded": {"transformer": None, "default": "fp8"},
        "none": None,
    }[scope]
    quant_config = build_quant_config(quantization)
    expected = quant_config
    if scope not in ("global", "none"):
        expected = quant_config.resolve("transformer")
    model = Mock(return_value=nn.Identity())
    model.return_value.config = SimpleNamespace(attention_head_dim=4)
    monkeypatch.setattr(pipeline, "StableAudioDiTModel", model)
    monkeypatch.setattr(pipeline, "get_transformer_config_kwargs", lambda *args: {})
    monkeypatch.setattr(pipeline, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(pipeline, "prefetch_subfolders", Mock())
    monkeypatch.setattr(pipeline.T5TokenizerFast, "from_pretrained", Mock())
    monkeypatch.setattr(pipeline, "from_pretrained_with_prefetch", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(pipeline.CosineDPMSolverMultistepScheduler, "from_pretrained", Mock())
    monkeypatch.setattr(pipeline, "StableAudioSchedulerWrapper", Mock())
    monkeypatch.setattr(pipeline.StableAudioPipeline, "setup_diffusion_pipeline_profiler", Mock())
    config = SimpleNamespace(
        model="unused-local-model",
        dtype=torch.bfloat16,
        quantization_config=quant_config,
        tf_model_config={},
        enable_diffusion_pipeline_profiler=False,
    )

    pipeline.StableAudioPipeline(od_config=config)

    assert model.call_args.kwargs["quant_config"] is expected
    if scope in ("text_encoder", "excluded", "none"):
        assert expected is None
