from dataclasses import dataclass

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMForConditionalGeneration,
    MiniCPMOAudioFeatureInputs,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


_AudioCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]


@dataclass
class _FakeConv:
    weight: torch.Tensor


@dataclass
class _FakeEmbeddingPositions:
    weight: torch.Tensor


@dataclass
class _StreamingAudioEncoderOutput:
    last_hidden_state: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None
    past_key_values: _AudioCache


@dataclass
class _StreamingAudioEncoderCall:
    past_key_values: _AudioCache | None
    use_cache: bool
    output_hidden_states: bool
    attention_mask_shape: tuple[int, ...]
    use_extra_context: bool
    prefix_extra_frames: int
    suffix_extra_frames: int
    cnn_min_length: int | None


@dataclass
class _AudioConfig:
    audio_pool_step: int = 1


class _FakeStreamingAudioEncoder:
    def __init__(self) -> None:
        self.conv1 = _FakeConv(weight=torch.empty(1, dtype=torch.float32))
        self.embed_positions = _FakeEmbeddingPositions(weight=torch.empty((32, 4)))
        self.calls: list[_StreamingAudioEncoderCall] = []

    def __call__(
        self,
        input_features: torch.Tensor,
        *,
        past_key_values: _AudioCache | None,
        use_cache: bool,
        output_hidden_states: bool,
        attention_mask: torch.Tensor,
        use_extra_context: bool,
        prefix_extra_frames: int,
        suffix_extra_frames: int,
        cnn_min_length: int | None,
    ) -> _StreamingAudioEncoderOutput:
        self.calls.append(
            _StreamingAudioEncoderCall(
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states,
                attention_mask_shape=tuple(attention_mask.shape),
                use_extra_context=use_extra_context,
                prefix_extra_frames=prefix_extra_frames,
                suffix_extra_frames=suffix_extra_frames,
                cnn_min_length=cnn_min_length,
            )
        )

        current_seq_len = (input_features.shape[-1] - 1) // 2 + 1
        if use_extra_context:
            current_seq_len -= (prefix_extra_frames + 1) // 2 if prefix_extra_frames > 0 else 0
            current_seq_len -= (suffix_extra_frames + 1) // 2 if suffix_extra_frames > 0 else 0

        past_seq_len = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        total_seq_len = past_seq_len + current_seq_len
        final = torch.full((1, current_seq_len, 4), 2.0)
        earlier = torch.full((1, current_seq_len, 4), 1.0)
        cache_tensor = torch.full((1, 1, total_seq_len, 4), float(total_seq_len))
        return _StreamingAudioEncoderOutput(
            last_hidden_state=final,
            hidden_states=(earlier, final) if output_hidden_states else None,
            past_key_values=((cache_tensor, cache_tensor.clone()),),
        )


class _FakeStreamingAudioModel:
    def __init__(self, audio_encoder_layer: int) -> None:
        self.config = _AudioConfig()
        self.apm = _FakeStreamingAudioEncoder()
        self.audio_projection_layer = torch.nn.Identity()
        self.audio_avg_pooler = torch.nn.Identity()
        self.audio_encoder_layer = audio_encoder_layer
        self.audio_past_key_values: _AudioCache | None = None

    def _get_feat_extract_output_lengths(
        self,
        input_lengths: torch.LongTensor,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        input_lengths_after_cnn = (input_lengths - 1) // 2 + 1
        input_lengths_after_pooling = (
            input_lengths_after_cnn - self.config.audio_pool_step
        ) // self.config.audio_pool_step + 1
        return input_lengths_after_cnn, input_lengths_after_pooling.to(dtype=torch.int32)


def _streaming_audio_input(feature_length: int) -> MiniCPMOAudioFeatureInputs:
    return MiniCPMOAudioFeatureInputs(
        audio_features=torch.zeros((1, 80, feature_length)),
        audio_feature_lens=[torch.tensor([feature_length])],
    )


@pytest.mark.parametrize(
    ("audio_encoder_layer", "expected_value", "expected_hidden_states"),
    [(-1, 2.0, False), (0, 1.0, True)],
)
def test_streaming_audio_retains_layers_only_for_nonfinal_selection(
    audio_encoder_layer: int,
    expected_value: float,
    expected_hidden_states: bool,
) -> None:
    model = _FakeStreamingAudioModel(audio_encoder_layer)

    result = MiniCPMO45OmniLLMForConditionalGeneration.get_audio_embedding_streaming(
        model,
        _streaming_audio_input(feature_length=4),
    )

    assert len(model.apm.calls) == 1
    call = model.apm.calls[0]
    assert call.past_key_values is None
    assert call.use_cache is True
    assert call.output_hidden_states is expected_hidden_states
    assert call.attention_mask_shape == (1, 1, 2, 2)
    torch.testing.assert_close(result[0][0], torch.full((2, 4), expected_value))


def test_streaming_audio_reuses_cache_across_chunks() -> None:
    model = _FakeStreamingAudioModel(audio_encoder_layer=-1)

    first_result = MiniCPMO45OmniLLMForConditionalGeneration.get_audio_embedding_streaming(
        model,
        _streaming_audio_input(feature_length=6),
        use_extra_context=True,
        prefix_extra_frames=0,
        suffix_extra_frames=2,
        cnn_min_length=8,
    )
    first_cache = model.audio_past_key_values
    second_result = MiniCPMO45OmniLLMForConditionalGeneration.get_audio_embedding_streaming(
        model,
        _streaming_audio_input(feature_length=6),
        use_extra_context=True,
        prefix_extra_frames=2,
        suffix_extra_frames=2,
        cnn_min_length=8,
    )

    assert first_cache is not None
    assert len(model.apm.calls) == 2
    first_call, second_call = model.apm.calls
    assert first_call.past_key_values is None
    assert first_call.output_hidden_states is False
    assert first_call.attention_mask_shape == (1, 1, 2, 2)
    assert first_call.use_extra_context is True
    assert first_call.prefix_extra_frames == 0
    assert first_call.suffix_extra_frames == 2
    assert first_call.cnn_min_length == 8

    assert second_call.past_key_values is first_cache
    assert second_call.output_hidden_states is False
    assert second_call.attention_mask_shape == (1, 1, 1, 3)
    assert second_call.use_extra_context is True
    assert second_call.prefix_extra_frames == 2
    assert second_call.suffix_extra_frames == 2
    assert second_call.cnn_min_length == 8

    assert model.audio_past_key_values is not first_cache
    assert first_cache[0][0].shape[2] == 2
    assert model.audio_past_key_values[0][0].shape[2] == 3
    torch.testing.assert_close(first_result[0][0], torch.full((2, 4), 2.0))
    torch.testing.assert_close(second_result[0][0], torch.full((1, 4), 2.0))
