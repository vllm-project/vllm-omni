from typing import Any

import torch
from torch import nn

from vllm_omni.model_executor.models.breeze_tts_2.modeling_breeze_tts_2_codec import (
    BreezeTTS2MimiCodec,
)


class _BundledTokenizer:
    def __init__(self) -> None:
        self.model = nn.Linear(2, 2)

    def named_parameters(self):
        for name, parameter in self.model.named_parameters():
            yield f"model.{name}", parameter

    def decode(self, _encoded: Any) -> tuple[list[torch.Tensor], int]:
        return [torch.zeros(4)], 24_000


def test_bundled_qwen_tokenizer_consumes_unused_root_fallback_weights(tmp_path):
    (tmp_path / "audio_tokenizer").mkdir()
    codec = object.__new__(BreezeTTS2MimiCodec)
    codec.model_path = str(tmp_path)
    codec.vllm_config = None
    codec._codec = None
    codec._audio_tokenizer = _BundledTokenizer()
    codec._loaded_local_weights = False

    weights = [
        ("backbone_model.layers.0.norm.weight", torch.ones(2)),
        ("codec_model.decoder.layers.0.conv.weight", torch.ones(2)),
        ("lm_head.weight", torch.ones(2, 2)),
    ]

    loaded = codec.load_weights(iter(weights))

    assert "_audio_tokenizer.model.weight" in loaded
    assert "_audio_tokenizer.model.bias" in loaded
    # The bundled tokenizer was preloaded, so load_weights must not try to copy
    # the root Mimi fallback into the nonexistent ``self._codec`` module.
    assert codec._codec is None


def test_codec_warmup_with_single_token_returns_empty_audio():
    codec = object.__new__(BreezeTTS2MimiCodec)
    codec._num_codebooks = 4
    codec._codebook_size = 8
    codec._sample_rate = 24_000
    codec._audio_tokenizer = None
    codec._codec = None
    codec._async_chunk = False

    output = codec.forward(torch.tensor([1], dtype=torch.long), seq_token_counts=[1])

    assert output.multimodal_outputs["model_outputs"][0].numel() == 0
    assert output.multimodal_outputs["sr"][0].item() == 24_000
