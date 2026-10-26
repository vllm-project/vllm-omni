# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the Drama 3 (fish_omni) HF config conversion."""

import copy

import pytest
from transformers import AutoConfig, Qwen3_5MoeTextConfig

import vllm_omni.transformers_utils.configs  # noqa: F401  (registers fish_omni)
from vllm_omni.model_executor.models.fish_omni.configuration_fish_omni import (
    FishOmniAudioDecoderConfig,
    FishOmniConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Field values from the public fishaudio/drama-3 config.json.
_LAYER_TYPES = (["linear_attention"] * 3 + ["full_attention"]) * 10
TEXT_CONFIG = {
    "model_type": "fish_omni_text",
    "attention_bias": False,
    "attn_output_gate": True,
    "eos_token_id": 248046,
    "head_dim": 256,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "id2label": {"0": "LABEL_0", "1": "LABEL_1"},
    "label2id": {"LABEL_0": 0, "LABEL_1": 1},
    "architectures": None,
    "_name_or_path": "",
    "dtype": None,
    "intermediate_size": 512,
    "layer_types": _LAYER_TYPES,
    "layernorm_zero_centered_gamma": True,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 262144,
    "mlp_layer_types": ["moe"] * 40,
    "moe_intermediate_size": 512,
    "num_attention_heads": 16,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 40,
    "num_key_value_heads": 2,
    "pad_token_id": 248044,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000.0, "rope_type": "default"},
    "rotary_interleaved": False,
    "shared_expert_gate": True,
    "shared_expert_intermediate_size": 512,
    "tie_word_embeddings": False,
    "vocab_size": 252288,
    "use_qk_norm": True,
}
AUDIO_DECODER_CONFIG = {
    "model_type": "fish_qwen3_audio_decoder",
    "dim": 2048,
    "head_dim": 256,
    "intermediate_size": 6144,
    "n_head": 16,
    "n_local_heads": 2,
    "norm_eps": 1e-06,
    "num_codebooks": 10,
    "rope_base": 10000000.0,
    "rotary_percent": 0.25,
    "shared_enrichment_layers": 4,
    "vocab_size": 4096,
    "vq_backbone_codebooks": 1,
}
TOP_LEVEL = {
    "model_type": "fish_omni",
    "audio_pad_token_id": 248076,
    "eos_token_id": 248046,
    "pad_token_id": 248044,
    "semantic_start_token_id": 248099,
    "semantic_end_token_id": 252194,
    "speech_template_version": "reference-bracket-v1",
}


def _config(**text_overrides) -> FishOmniConfig:
    text = copy.deepcopy(TEXT_CONFIG)
    text.update(text_overrides)
    return FishOmniConfig(text_config=text, audio_decoder_config=dict(AUDIO_DECODER_CONFIG), **TOP_LEVEL)


def test_text_config_is_qwen3_5_moe():
    config = _config()
    text = config.get_text_config()
    assert isinstance(text, Qwen3_5MoeTextConfig)
    assert text.hidden_size == 2048
    assert text.num_hidden_layers == 40
    assert text.layer_types == _LAYER_TYPES
    assert text.num_experts == 256
    assert text.num_experts_per_tok == 8
    assert text.moe_intermediate_size == 512
    assert text.shared_expert_intermediate_size == 512
    assert text.num_attention_heads == 16
    assert text.num_key_value_heads == 2
    assert text.head_dim == 256
    assert text.linear_num_value_heads == 32
    assert text.vocab_size == 252288
    assert text.rope_parameters["partial_rotary_factor"] == 0.25
    assert text.rope_parameters["rope_theta"] == 1e7
    assert text.mamba_ssm_dtype == "float32"
    for dropped in ("mlp_layer_types", "rotary_interleaved", "use_qk_norm", "shared_expert_gate"):
        assert not hasattr(text, dropped)


def test_top_level_fields():
    config = _config()
    assert config.semantic_start_token_id == 248099
    assert config.semantic_end_token_id == 252194
    assert config.audio_pad_token_id == 248076
    assert config.eos_token_id == 248046
    assert config.tie_word_embeddings is False
    decoder = config.audio_decoder_config
    assert isinstance(decoder, FishOmniAudioDecoderConfig)
    assert decoder.num_codebooks == 10
    assert decoder.rotary_dim == 64


def test_auto_config_registration():
    assert AutoConfig.for_model("fish_omni").__class__ is FishOmniConfig


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("rotary_interleaved", True),
        ("use_qk_norm", False),
        ("layernorm_zero_centered_gamma", False),
        ("shared_expert_gate", False),
    ],
)
def test_rejects_unsupported_text_values(key, value):
    with pytest.raises(ValueError, match=f"text_config.{key}"):
        _config(**{key: value})


def test_rejects_dense_backbone_layer():
    with pytest.raises(ValueError, match="every backbone layer must be MoE"):
        _config(mlp_layer_types=["moe"] * 39 + ["dense"])


def test_npu_fused_qk_rope_is_rejected_after_construction():
    import vllm_omni.platforms.npu.models.fish_omni as fish_model

    class _Eager:
        use_fused_qk_norm_rope_gate = False

    class _Fused:
        use_fused_qk_norm_rope_gate = True

    class _Root:
        def __init__(self, children: list[object]) -> None:
            self._children = children

        def modules(self) -> list[object]:
            return self._children

    fish_model._require_npu_eager_qk_rope(_Root([_Eager()]))
    with pytest.raises(RuntimeError, match="fused QK-norm RoPE"):
        fish_model._require_npu_eager_qk_rope(_Root([_Eager(), _Fused()]))
    for children in ([], [object()]):
        with pytest.raises(RuntimeError, match="no attention modules expose use_fused_qk_norm_rope_gate"):
            fish_model._require_npu_eager_qk_rope(_Root(children))


def test_tensor_parallel_is_allowed_on_npu_only(monkeypatch):
    from types import SimpleNamespace

    import vllm_omni.model_executor.models.fish_omni.fish_omni_model as fish_model

    parallel = SimpleNamespace(pipeline_parallel_size=1, prefill_context_parallel_size=1, tensor_parallel_size=2)
    monkeypatch.setattr(fish_model, "current_platform", SimpleNamespace(device_type="npu"))
    fish_model._reject_unsupported_fish_parallel(parallel)
    monkeypatch.setattr(fish_model, "current_platform", SimpleNamespace(device_type="cuda"))
    with pytest.raises(NotImplementedError, match="tensor parallel"):
        fish_model._reject_unsupported_fish_parallel(parallel)
    with pytest.raises(NotImplementedError, match="one pipeline rank"):
        fish_model._reject_unsupported_fish_parallel(
            SimpleNamespace(pipeline_parallel_size=2, prefill_context_parallel_size=1)
        )
    with pytest.raises(NotImplementedError, match="prefill context"):
        fish_model._reject_unsupported_fish_parallel(
            SimpleNamespace(pipeline_parallel_size=1, prefill_context_parallel_size=2)
        )
