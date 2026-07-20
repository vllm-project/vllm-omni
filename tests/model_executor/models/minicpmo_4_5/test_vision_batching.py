from dataclasses import dataclass

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _VisionEncoderOutput:
    last_hidden_state: torch.Tensor


@dataclass
class _VisionConfig:
    vision_batch_size: int


@dataclass
class _AudioConfig:
    audio_chunk_length: float = 0
    audio_pool_step: int = 1


@dataclass
class _FakeConv:
    weight: torch.Tensor


@dataclass
class _AudioEncoderOutput:
    last_hidden_state: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None


class _FakeVisionEncoder:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(
        self,
        pixel_values: torch.Tensor,
        *,
        patch_attention_mask: torch.Tensor,
        tgt_sizes: torch.Tensor,
    ) -> _VisionEncoderOutput:
        batch_size = pixel_values.shape[0]
        self.batch_sizes.append(batch_size)
        assert patch_attention_mask.shape[0] == batch_size
        assert tgt_sizes.shape[0] == batch_size

        per_item = pixel_values.sum(dim=(1, 2, 3))
        hidden_states = per_item[:, None, None].expand(-1, 3, 4).clone()
        return _VisionEncoderOutput(last_hidden_state=hidden_states)


class _FakeResampler:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, hidden_states: torch.Tensor, tgt_sizes: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(hidden_states.shape[0])
        return hidden_states + tgt_sizes.sum(dim=1)[:, None, None]


class _FakeAudioEncoder:
    def __init__(self) -> None:
        self.conv1 = _FakeConv(weight=torch.empty(1, dtype=torch.float32))
        self.output_hidden_states: list[bool] = []

    def __call__(
        self,
        input_features: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
    ) -> _AudioEncoderOutput:
        self.output_hidden_states.append(output_hidden_states)
        assert attention_mask.shape == (1, 1, 2, 2)
        final = torch.full((1, 2, 4), 2.0)
        earlier = torch.full((1, 2, 4), 1.0)
        return _AudioEncoderOutput(
            last_hidden_state=final,
            hidden_states=(earlier, final) if output_hidden_states else None,
        )


class _FakeVisionModel:
    def __init__(self, vision_batch_size: int) -> None:
        self.config = _VisionConfig(vision_batch_size=vision_batch_size)
        self.vpm = _FakeVisionEncoder()
        self.resampler = _FakeResampler()


class _FakeAudioModel:
    def __init__(self, audio_encoder: _FakeAudioEncoder, audio_encoder_layer: int) -> None:
        self.config = _AudioConfig()
        self.apm = audio_encoder
        self.audio_projection_layer = torch.nn.Identity()
        self.audio_avg_pooler = torch.nn.Identity()
        self.audio_encoder_layer = audio_encoder_layer

    def _get_feat_extract_output_lengths(
        self,
        input_lengths: torch.LongTensor,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        input_lengths_after_cnn = (input_lengths - 1) // 2 + 1
        input_lengths_after_pooling = (
            input_lengths_after_cnn - self.config.audio_pool_step
        ) // self.config.audio_pool_step + 1
        return input_lengths_after_cnn, input_lengths_after_pooling.to(dtype=torch.int32)


def _run_vision_encoder(vision_batch_size: int) -> tuple[torch.Tensor, list[int], list[int]]:
    model = _FakeVisionModel(vision_batch_size)
    pixel_values = [
        torch.full((3, 2, length), fill_value=index + 1, dtype=torch.float32)
        for index, length in enumerate((4, 3, 2, 4, 1))
    ]
    tgt_sizes = torch.tensor([[1, 2], [1, 2], [1, 1], [2, 2], [1, 1]])

    result = MiniCPMO45OmniLLMForConditionalGeneration.get_vision_hidden_states(
        model,
        {"pixel_values": pixel_values, "tgt_sizes": tgt_sizes},
    )
    return result, model.vpm.batch_sizes, model.resampler.batch_sizes


def test_vision_encoder_batches_vpm_before_resampling() -> None:
    chunked, vision_batches, resampler_batches = _run_vision_encoder(2)
    unchunked, unchunked_vision_batches, unchunked_resampler_batches = _run_vision_encoder(16)

    assert vision_batches == [2, 2, 1]
    assert resampler_batches == [5]
    assert unchunked_vision_batches == [5]
    assert unchunked_resampler_batches == [5]
    torch.testing.assert_close(chunked, unchunked, rtol=0, atol=0)


@pytest.mark.parametrize("vision_batch_size", [0, -2])
def test_vision_encoder_clamps_nonpositive_batch_size(vision_batch_size: int) -> None:
    chunked, vision_batches, resampler_batches = _run_vision_encoder(vision_batch_size)
    unchunked, _, _ = _run_vision_encoder(16)

    assert vision_batches == [1, 1, 1, 1, 1]
    assert resampler_batches == [5]
    torch.testing.assert_close(chunked, unchunked, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("audio_encoder_layer", "expected_value", "expected_hidden_states"),
    [(-1, 2.0, False), (0, 1.0, True)],
)
def test_audio_encoder_retains_layers_only_for_nonfinal_selection(
    audio_encoder_layer: int,
    expected_value: float,
    expected_hidden_states: bool,
) -> None:
    audio_encoder = _FakeAudioEncoder()
    model = _FakeAudioModel(audio_encoder, audio_encoder_layer)

    result = MiniCPMO45OmniLLMForConditionalGeneration.get_audio_hidden_states(
        model,
        {
            "audio_features": torch.zeros((1, 80, 4)),
            "audio_feature_lens": [torch.tensor([4])],
        },
    )

    assert audio_encoder.output_hidden_states == [expected_hidden_states]
    torch.testing.assert_close(result[0], torch.full((2, 4), expected_value))
