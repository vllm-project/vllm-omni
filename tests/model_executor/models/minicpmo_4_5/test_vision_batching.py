# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMForConditionalGeneration,
    Resampler,
)

pytestmark = [pytest.mark.core_model]


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
        self.vision_packed_encode = False


class _FakeAudioModel:
    _encode_audio_features = MiniCPMO45OmniLLMForConditionalGeneration._encode_audio_features

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


@pytest.mark.cpu
def test_vision_encoder_batches_vpm_before_resampling() -> None:
    chunked, vision_batches, resampler_batches = _run_vision_encoder(2)
    unchunked, unchunked_vision_batches, unchunked_resampler_batches = _run_vision_encoder(16)

    assert vision_batches == [2, 2, 1]
    assert resampler_batches == [5]
    assert unchunked_vision_batches == [5]
    assert unchunked_resampler_batches == [5]
    torch.testing.assert_close(chunked, unchunked, rtol=0, atol=0)


@pytest.mark.cpu
@pytest.mark.parametrize("vision_batch_size", [0, -2])
def test_vision_encoder_clamps_nonpositive_batch_size(vision_batch_size: int) -> None:
    chunked, vision_batches, resampler_batches = _run_vision_encoder(vision_batch_size)
    unchunked, _, _ = _run_vision_encoder(16)

    assert vision_batches == [1, 1, 1, 1, 1]
    assert resampler_batches == [5]
    torch.testing.assert_close(chunked, unchunked, rtol=0, atol=0)


@pytest.mark.cpu
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


class _TorchResamplerAttention(torch.nn.MultiheadAttention):
    def forward(self, query, key, value, key_padding_mask):
        return super().forward(query, key, value, key_padding_mask=key_padding_mask)[0]


@pytest.mark.parametrize(
    "device,dtype",
    [
        pytest.param("cpu", torch.float32, marks=pytest.mark.cpu),
        *[
            pytest.param("cuda", dtype, marks=hardware_marks(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1))
            for dtype in (torch.float32, torch.float16, torch.bfloat16)
        ],
    ],
)
@pytest.mark.parametrize("kv_dim", [48, 72])
@pytest.mark.parametrize("padded", [False, True])
@torch.inference_mode()
def test_resampler_matches_torch_cross_attention(device, dtype, kv_dim, padded, mocker):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm required")
    torch.manual_seed(17)
    options = dict(num_queries=8, embed_dim=72, num_heads=3, kv_dim=kv_dim, max_size=(2, 2))
    model = Resampler(**options).eval().to(device=device, dtype=dtype)
    model.query.normal_()
    model.attn.in_proj_bias.normal_(std=0.1)
    reference = Resampler(**options)
    reference.attn = _TorchResamplerAttention(72, 3)
    reference = reference.eval().to(device=device, dtype=dtype)
    reference.load_state_dict(model.state_dict(), strict=True)

    sizes = [(3, 4), (2, 5), (1, 3)] if padded else [(3, 4)] * 3
    tgt_sizes = torch.tensor(sizes, device=device)
    features = torch.randn(3, 12, kv_dim, device=device, dtype=dtype)
    projection = mocker.spy(model.attn.out_proj, "forward")
    actual = model(features, tgt_sizes)
    projection.assert_called_once()
    # PyTorch multiplies by the reciprocal scale; MiniCPM keeps division.
    atol = {torch.float32: 1e-6, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(actual, reference(features, tgt_sizes), atol=atol, rtol=atol)

    # Padded patches must not contribute even when their values change.
    for i, (height, width) in enumerate(sizes):
        features[i, height * width :].fill_(100)
    torch.testing.assert_close(model(features, tgt_sizes), actual, rtol=0, atol=0)

    # Different packed/padded GEMM shapes already round differently in FP16/BF16.
    # Check layout equivalence in FP32, separately from the dtype checks above.
    if dtype == torch.float32:
        model.attn_implementation = "eager"
        packed = torch.cat([features[i, : h * w] for i, (h, w) in enumerate(sizes)])
        layouts = [(h, w, 1) for h, w in sizes]
        torch.testing.assert_close(model.forward_packed(packed, layouts), actual, atol=atol, rtol=atol)
