# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.mimo_audio import mimo_audio_llm
from vllm_omni.model_executor.models.mimo_audio.config_mimo_audio import MiMoAudioConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("module_name", ["local_transformer", "input_local_transformer"])
def test_constructor_disables_dropout_before_capture(monkeypatch, module_name):
    config = MiMoAudioConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        local_dim=16,
        local_layers=1,
        local_attn_heads=2,
        local_ffn_dim=32,
        local_attn_dropout=0.1,
        input_local_layers=1,
        speech_vocab_size=8,
        speech_zeroemb_idx=7,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        quant_config=None,
        lora_config=None,
    )
    cuda_parameter = SimpleNamespace(device=torch.device("cuda"))
    captured_training = {}

    def report_cuda(module):
        monkeypatch.setattr(module, "parameters", lambda: iter((cuda_parameter,)))
        return module

    def capture(model, buffer, batch_size, *, name):
        captured_training[name] = any(module.training for module in getattr(model, name).modules())

    transformer_cls = mimo_audio_llm.MiMoAudioQwen2Model
    monkeypatch.setattr(mimo_audio_llm, "MiMoAudioQwen2Model", lambda config: report_cuda(transformer_cls(config)))
    monkeypatch.setattr(
        mimo_audio_llm,
        "init_vllm_registered_model",
        lambda **kwargs: SimpleNamespace(make_empty_intermediate_tensors=None),
    )
    monkeypatch.setattr(
        mimo_audio_llm,
        "ColumnParallelLinear",
        lambda input_size, output_size, **kwargs: report_cuda(nn.Linear(input_size, output_size, bias=kwargs["bias"])),
    )
    monkeypatch.setattr(mimo_audio_llm.current_omni_platform, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(mimo_audio_llm, "MIMO_CUDAGRAPH_BATCH_SIZES", (1,))
    for buffer_name in ("MiMoLocalDecodeBuffer", "MiMoInputLocalTransformerBuffer"):
        monkeypatch.setattr(mimo_audio_llm, buffer_name, lambda *args, **kwargs: None)
    for name, graph_cls in (
        ("local_transformer", mimo_audio_llm.MiMoLocalDecodeCudaGraph),
        ("input_local_transformer", mimo_audio_llm.MiMoInputLocalTransformerCudaGraph),
    ):
        monkeypatch.setattr(
            graph_cls,
            "capture",
            lambda model, buffer, batch_size, name=name: capture(model, buffer, batch_size, name=name),
        )

    model = mimo_audio_llm.MiMoAudioLLMForConditionalGeneration(vllm_config=vllm_config)
    transformer = getattr(model, module_name)

    assert captured_training[module_name] is False
    assert all(not module.training for module in transformer.modules())
