"""Config for ``THUDM/glm-4-voice-tokenizer`` (the WhisperVQ encoder used
by Kimi-Audio's input audio path).

Only the fields the encoder actually reads at inference are kept; the
training-only EMA / restart / spec-augment / pooling-type knobs are
hardcoded into the model code instead. See
``vllm_omni.model_executor.models.kimi_audio.glm4.modeling_whisper``."""

from transformers import WhisperConfig


class WhisperVQConfig(WhisperConfig):
    def __init__(
        self,
        pooling_kernel_size: int = 4,
        pooling_position: int = 16,
        quantize_vocab_size: int = 16384,
        quantize_position: int = 16,
        quantize_causal_block_size: int | None = 200,
        **kwargs,
    ):
        self.pooling_kernel_size = pooling_kernel_size
        self.pooling_position = pooling_position
        self.quantize_vocab_size = quantize_vocab_size
        self.quantize_position = quantize_position
        self.quantize_causal_block_size = quantize_causal_block_size
        super().__init__(**kwargs)
