import torch

from vllm_omni.model_executor.models.breeze_tts_2.configuration_breeze_tts_2 import (
    BreezeTTS2DepthDecoderConfig,
    BreezeTTS2TextEncoderConfig,
)
from vllm_omni.model_executor.models.breeze_tts_2.modeling_breeze_tts_2_depth import (
    BreezeDepthDecoderForCausalLM,
)
from vllm_omni.model_executor.models.breeze_tts_2.modeling_breeze_tts_2_text import (
    BreezeTTS2TextEncoder,
)


def _depth_config() -> BreezeTTS2DepthDecoderConfig:
    return BreezeTTS2DepthDecoderConfig(
        hidden_size=16,
        backbone_hidden_size=16,
        audio_embed_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=8,
        num_codebooks=4,
        vocab_size=8,
        rope_scaling={
            "rope_type": "llama3",
            "rope_theta": 100.0,
            "factor": 1.0,
            "low_freq_factor": 0.5,
            "high_freq_factor": 1.0,
            "original_max_position_embeddings": 4,
        },
    )


def test_depth_decoder_position_specific_logits_and_offsets():
    config = _depth_config()
    decoder = BreezeDepthDecoderForCausalLM(config)

    output = decoder(
        input_ids=torch.tensor([[0, 2]]),
        backbone_last_hidden_state=torch.randn(1, 16),
        use_cache=False,
    )

    assert output.logits.shape == (1, 1, 8)
    with torch.no_grad():
        embedding = decoder.model.embed_tokens.weight
        expected_first = embedding[2] @ decoder.model.inputs_embeds_projector.weight.T
        # The first hidden row is replaced by the backbone state, so compare the
        # second input row (codebook 0, id 2 + 0 * vocab_size).
        actual_input = decoder.model.embed_tokens(torch.tensor([[2]]))[0, 0]
        actual_projected = actual_input @ decoder.model.inputs_embeds_projector.weight.T
    assert torch.allclose(expected_first, actual_projected)


def _text_config() -> BreezeTTS2TextEncoderConfig:
    return BreezeTTS2TextEncoderConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=32,
        layer_types=["sliding_attention", "full_attention"],
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 100.0},
            "full_attention": {"rope_type": "default", "rope_theta": 100.0},
        },
    )


def test_text_encoder_masks_right_padding_independently(monkeypatch):
    import vllm_omni.model_executor.models.breeze_tts_2.modeling_breeze_tts_2_text as text_module

    class _ParallelEmbedding(torch.nn.Embedding):
        def __init__(self, num_embeddings: int, embedding_dim: int):
            super().__init__(num_embeddings, embedding_dim)

    class _QKVLinear(torch.nn.Module):
        def __init__(
            self,
            hidden_size: int,
            head_size: int,
            total_num_heads: int,
            total_num_kv_heads: int,
            **_: object,
        ):
            super().__init__()
            output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
            self.linear = torch.nn.Linear(hidden_size, output_size, bias=False)

        def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
            return self.linear(hidden_states), None

    class _RowLinear(torch.nn.Module):
        def __init__(self, input_size: int, output_size: int, **_: object):
            super().__init__()
            self.linear = torch.nn.Linear(input_size, output_size, bias=False)

        def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
            return self.linear(hidden_states), None

    class _MergedLinear(torch.nn.Module):
        def __init__(self, input_size: int, output_sizes: list[int], **_: object):
            super().__init__()
            self.linear = torch.nn.Linear(input_size, sum(output_sizes), bias=False)

        def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
            return self.linear(hidden_states), None

    monkeypatch.setattr(text_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(text_module, "VocabParallelEmbedding", _ParallelEmbedding)
    monkeypatch.setattr(text_module, "QKVParallelLinear", _QKVLinear)
    monkeypatch.setattr(text_module, "RowParallelLinear", _RowLinear)
    monkeypatch.setattr(text_module, "MergedColumnParallelLinear", _MergedLinear)

    config = _text_config()
    encoder = BreezeTTS2TextEncoder(config).eval()
    ids = torch.tensor([[3, 4]])
    padded_ids = torch.tensor([[3, 4, 0, 0]])
    mask = torch.tensor([[True, True, False, False]])

    with torch.inference_mode():
        short = encoder(ids)
        padded = encoder(padded_ids, attention_mask=mask)

    assert short.shape == (1, 2, 16)
    assert padded.shape == (1, 4, 16)
    assert torch.allclose(short[0, :2], padded[0, :2], atol=1e-5)


def test_text_encoder_weight_loader_reports_fused_target_names():
    encoder = object.__new__(BreezeTTS2TextEncoder)
    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    parameter.weight_loader = lambda param, tensor, shard_id: param.data.copy_(tensor)
    encoder.__dict__["named_parameters"] = lambda: [("layers.0.qkv_proj.weight", parameter)]

    loaded = encoder.load_weights([("layers.0.q_proj.weight", torch.ones(2, 2))])

    assert loaded == {"layers.0.qkv_proj.weight"}
    assert torch.equal(parameter.detach(), torch.ones(2, 2))
